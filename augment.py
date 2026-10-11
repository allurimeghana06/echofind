"""Phase 1: stochastic augmentation pipeline (view generation for SSL).

Two kinds of views, one per head of the two-head encoder:

IDENTITY views (nuisance only; "the same recording through a different phone in a cafe")
  Waveform stage (simulates the acoustic channel, NAFP-style):
    1. reverb      - convolve with a room impulse response (real or synthetic)
    2. noise       - environmental noise + babble ("cafe chatter") at a random SNR
    3. EQ          - random smooth frequency tilt/bumps (phone mics, cheap speakers)
    4. gain        - random loudness
  Spectrogram stage:
    5. SpecAugment - random frequency and time masks
  Crop policy: both identity views come from the SAME moment (shifted <= 0.5 s).

STYLE view ("a different part of the song, played slightly differently")
  Crop policy: a DIFFERENT, non-overlapping segment of the same track.
    6. speed perturbation +-8% (tempo and pitch move together)
    7. mel shift +-2 bins (~1-2 semitones)
    8. light nuisance (SNR 10-30 dB, less reverb) + SpecAugment
  Style augmentations are never applied to identity views: they would destroy the
  exact-recording detail that fingerprinting relies on.

The class works on CPU or GPU, on one clip (T,) or a batch (B, T). If no noise/RIR
banks are given it uses the small bundled set in assets/, and if that is missing too it
falls back to synthetic noise and synthetic reverb, so the organizers' check
("import AugmentationPipeline, pass a dummy file") always works.
"""
import os
import math
import numpy as np
import torch
import torch.nn as nn

from features import SR, LogMel, load_audio, to_tensor

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ASSETS = os.path.join(HERE, "assets", "aug_assets.npz")


# ----------------------------------------------------------------------------- banks
def load_bank(path):
    """Ragged bank saved by notebook 02 -> list of 1-D float32 tensors."""
    z = np.load(path)
    d, o = z["data"], z["offsets"]
    return [torch.from_numpy(d[o[i]:o[i + 1]].astype(np.float32)) for i in range(len(o) - 1)]


def synthetic_rir(rng, sr=SR, rt60=None):
    """Exponentially decaying noise: a cheap, decent stand-in for a real room."""
    rt60 = rt60 if rt60 is not None else rng.uniform(0.3, 1.8)
    n = int(min(2.0, rt60 * 1.2) * sr)
    t = np.arange(n) / sr
    h = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)      # -60 dB at t = rt60
    h[0] = max(1.0, np.abs(h).max() * 2)                       # strong direct path
    return (h / np.linalg.norm(h)).astype(np.float32)


def synthetic_noise(rng, n, sr=SR):
    """Coloured noise with slow amplitude modulation (rough traffic/crowd stand-in)."""
    w = rng.standard_normal(n)
    f = np.fft.rfft(w)
    freqs = np.fft.rfftfreq(n, 1 / sr) + 1.0
    f *= freqs ** (-rng.uniform(0.5, 1.5) / 2)                   # pink-to-brown
    x = np.fft.irfft(f, n)
    env = 1 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.1, 2) * np.arange(n) / sr)
    x = x * env
    return (x / (x.std() + 1e-8)).astype(np.float32)


# ----------------------------------------------------------------------------- helpers
def _rms(x):
    return x.pow(2).mean(dim=-1, keepdim=True).sqrt()


def _random_segment(clip, length, gen):
    """Random `length`-sample window from `clip` (tiled if the clip is shorter)."""
    if clip.numel() < length:
        clip = clip.repeat(math.ceil(length / clip.numel()) + 1)
    s = int(torch.randint(0, clip.numel() - length + 1, (1,), generator=gen))
    return clip[s:s + length]


def fft_convolve(x, h):
    """Batched linear convolution via FFT, output cropped to len(x).  x:(B,T) h:(B,L)."""
    T, L = x.shape[-1], h.shape[-1]
    n = 1 << (T + L - 1).bit_length()
    with torch.autocast(device_type=x.device.type, enabled=False):
        y = torch.fft.irfft(torch.fft.rfft(x.float(), n) * torch.fft.rfft(h.float(), n), n)
    return y[..., :T]


class AugmentationPipeline(nn.Module):
    """Creates two augmented log-mel views of an audio clip.

    Main entry points
      pipeline(audio)                -> (view1, view2), each (1, 128, 431): the IDENTITY pair
                                        (same moment +-0.5 s, two independent noisy rooms)
      pipeline.view(crops)           -> identity-style view of a batch (nuisance only)
      pipeline.style_view(long_crops)-> style view: speed perturbation + mel shift + light nuisance
      pipeline.augment_wave(wav)     -> augmented waveform only (for listening tests)

    The identity head never sees style augmentations (they would break fingerprinting);
    the style head's second view always gets them.

    `audio` may be a file path, numpy array, or tensor (T,) / (B, T).
    """

    def __init__(self, noise_bank=None, babble_bank=None, rir_bank=None,
                 assets=DEFAULT_ASSETS, crop_seconds=5.0, sr=SR,
                 snr_db=(0.0, 20.0), p_reverb=0.8, p_noise=0.9, p_babble=0.5,
                 p_reverb_after_noise=0.5, p_eq=0.8, eq_db=6.0, gain_db=(-6.0, 6.0),
                 n_freq_masks=2, max_freq_mask=16, n_time_masks=2, max_time_mask=40,
                 max_shift_seconds=0.5,
                 style_speed=(0.92, 1.08), style_mel_shift=2,
                 light_snr_db=(10.0, 30.0), light_p_reverb=0.3,
                 seed=None):
        super().__init__()
        self.sr = sr
        self.crop = int(crop_seconds * sr)
        self.snr_db, self.p_reverb, self.p_noise, self.p_babble = snr_db, p_reverb, p_noise, p_babble
        self.p_rev_after = p_reverb_after_noise
        self.p_eq, self.eq_db, self.gain_db = p_eq, eq_db, gain_db
        self.fm = (n_freq_masks, max_freq_mask)
        self.tm = (n_time_masks, max_time_mask)
        self.max_shift = int(max_shift_seconds * sr)
        self.style_speed, self.style_mel_shift = style_speed, style_mel_shift
        self.light_snr_db, self.light_p_reverb = light_snr_db, light_p_reverb
        self.logmel = LogMel(sr=sr)
        self.gen = torch.Generator()
        self.gen.manual_seed(seed if seed is not None else int(np.random.SeedSequence().entropy % 2**31))
        rng = np.random.default_rng(seed)

        def as_bank(b):
            if b is None:
                return None
            if isinstance(b, str):
                return load_bank(b)
            return [to_tensor(c) for c in b]

        noise, babble, rir = as_bank(noise_bank), as_bank(babble_bank), as_bank(rir_bank)
        if (noise is None or babble is None or rir is None) and assets and os.path.exists(assets):
            z = np.load(assets)
            def unpack(prefix):
                d, o = z[prefix + "_data"], z[prefix + "_offsets"]
                return [torch.from_numpy(d[o[i]:o[i + 1]].astype(np.float32)) for i in range(len(o) - 1)]
            noise = noise or unpack("noise")
            babble = babble or unpack("babble")
            rir = rir or unpack("rir")
        self.synthetic = noise is None or rir is None
        if noise is None:
            noise = [torch.from_numpy(synthetic_noise(rng, 10 * sr)) for _ in range(32)]
        if babble is None:
            babble = [torch.from_numpy(synthetic_noise(rng, 10 * sr)) for _ in range(16)]
        if rir is None:
            rir = [torch.from_numpy(synthetic_rir(rng)) for _ in range(64)]
        self.noise, self.babble, self.rir = noise, babble, rir

    # ---------------------------------------------------------------- waveform stage
    def _batch_from_bank(self, bank, B, T, device):
        idx = torch.randint(0, len(bank), (B,), generator=self.gen).tolist()
        return torch.stack([_random_segment(bank[i], T, self.gen) for i in idx]).to(device)

    def _rirs(self, B, device):
        idx = torch.randint(0, len(self.rir), (B,), generator=self.gen).tolist()
        L = max(self.rir[i].numel() for i in idx)
        h = torch.zeros(B, L, device=self.rir[0].device)
        for b, i in enumerate(idx):
            h[b, :self.rir[i].numel()] = self.rir[i]
        return h.to(device)

    def _rand(self, B, device, lo=0.0, hi=1.0):
        return (torch.rand(B, 1, generator=self.gen) * (hi - lo) + lo).to(device)

    def _reverb(self, x, mask):
        wet = fft_convolve(x, self._rirs(x.shape[0], x.device))
        # keep loudness comparable to the dry signal
        wet = wet * (_rms(x) / (_rms(wet) + 1e-8))
        return torch.where(mask, wet, x)

    def augment_wave(self, wav, light=False):
        """wav: (T,) or (B, T) float tensor -> augmented waveform, same shape.
        light=True: milder noise (SNR 10-30 dB) and less reverb, used for the style view."""
        squeeze = wav.dim() == 1
        x = wav.unsqueeze(0) if squeeze else wav
        x = x.float()
        B, T = x.shape
        dev = x.device
        x = x - x.mean(dim=-1, keepdim=True)

        p_rev = self.light_p_reverb if light else self.p_reverb
        snr_range = self.light_snr_db if light else self.snr_db
        do_rev = self._rand(B, dev) < p_rev
        rev_after = self._rand(B, dev) < self.p_rev_after
        x = self._reverb(x, do_rev & ~rev_after)                 # reverb the music only

        # noise: environmental always (if active) + babble sometimes
        noise = self._batch_from_bank(self.noise, B, T, dev)
        bab = self._batch_from_bank(self.babble, B, T, dev)
        use_bab = (self._rand(B, dev) < self.p_babble).float()
        w = self._rand(B, dev, 0.3, 1.0)                         # balance of the two noises
        n = noise * (1 - use_bab * (1 - w)) + bab * use_bab * w
        snr = self._rand(B, dev, *snr_range)
        # SNR mixing: scale noise so 20*log10(rms(signal)/rms(noise)) = snr
        n = n * (_rms(x) / (_rms(n) + 1e-8)) * 10 ** (-snr / 20)
        add = (self._rand(B, dev) < self.p_noise).float()
        x = x + add * n

        x = self._reverb(x, do_rev & rev_after)                  # or reverb the whole mix

        # EQ: smooth random gain curve over log-frequency, applied in the FFT domain
        if self.p_eq > 0:
            n_fft = 1 << (T - 1).bit_length()
            with torch.autocast(device_type=dev.type, enabled=False):
                X = torch.fft.rfft(x.float(), n_fft)
                knots = (torch.rand(B, 6, generator=self.gen) * 2 - 1).to(dev) * self.eq_db
                pos = torch.linspace(0, 1, X.shape[-1], device=dev)
                logf = torch.log1p(pos * 100) / math.log(101)        # log-ish frequency axis
                curve = torch.nn.functional.interpolate(knots.unsqueeze(1), size=X.shape[-1],
                                                        mode="linear", align_corners=True).squeeze(1)
                # warp knots onto log axis
                idx = (logf * (X.shape[-1] - 1)).long().clamp(0, X.shape[-1] - 1)
                curve = curve[:, idx]
                do_eq = (self._rand(B, dev) < self.p_eq).float()
                g = 10 ** (curve * do_eq / 20)
                x = torch.fft.irfft(X * g, n_fft)[..., :T]

        x = x * 10 ** (self._rand(B, dev, *self.gain_db) / 20)
        x = x / (x.abs().amax(dim=-1, keepdim=True).clamp(min=1.0))  # avoid clipping
        return x.squeeze(0) if squeeze else x

    # ---------------------------------------------------------------- spectrogram stage
    def spec_augment(self, spec):
        """spec: (B, 1, M, F), already instance-normalised -> masks filled with 0 (= mean)."""
        B, _, M, F = spec.shape
        dev = spec.device
        keep = torch.ones(B, M, F, dtype=torch.bool, device=dev)
        for size, n_masks, max_w in ((M, *self.fm), (F, *self.tm)):
            ar = torch.arange(size, device=dev)
            for _ in range(n_masks):
                w = torch.randint(0, max_w + 1, (B, 1), generator=self.gen).to(dev)
                s = (torch.rand(B, 1, generator=self.gen).to(dev) * (size - w).clamp(min=0)).long()
                m = (ar >= s) & (ar < s + w)                      # (B, size)
                keep &= ~(m[:, :, None] if size == M else m[:, None, :])
        return spec * keep.unsqueeze(1)

    def banks_to(self, device):
        """Move noise/babble/RIR banks to the GPU once, so augmentation never touches the CPU."""
        self.noise = [c.to(device) for c in self.noise]
        self.babble = [c.to(device) for c in self.babble]
        self.rir = [c.to(device) for c in self.rir]
        self.logmel = self.logmel.to(device)
        return self

    def view(self, wav):
        """Identity view (nuisance augmentations only): waveform (B, T) -> log-mel (B, 1, M, F)."""
        return self.spec_augment(self.logmel.to(wav.device)(self.augment_wave(wav)))

    # ---------------------------------------------------------------- style view
    def speed_perturb(self, wav, T):
        """Resample each clip by a random factor (changes tempo AND pitch together, like a
        record played slightly fast/slow). wav: (B, L) with L >= T * max factor -> (B, T)."""
        B, L = wav.shape
        lo, hi = self.style_speed
        r = (torch.rand(B, generator=self.gen) * (hi - lo) + lo).tolist()
        out = []
        for b in range(B):
            n_new = max(T, int(round(L / r[b])))
            y = torch.nn.functional.interpolate(wav[b].view(1, 1, -1).float(), size=n_new,
                                                mode="linear", align_corners=False)[0, 0]
            out.append(y[:T])
        return torch.stack(out)

    def mel_shift(self, spec):
        """Shift each spectrogram up/down by up to `style_mel_shift` mel bins (~1-2 semitones
        in the mid range); edges are filled by repeating the border bin."""
        B, _, M, F = spec.shape
        k = torch.randint(-self.style_mel_shift, self.style_mel_shift + 1, (B,), generator=self.gen)
        idx = (torch.arange(M).view(1, M) - k.view(B, 1)).clamp(0, M - 1).to(spec.device)
        return torch.gather(spec, 2, idx.view(B, 1, M, 1).expand(B, 1, M, F))

    def style_view(self, wav_long):
        """Style view: speed perturbation -> light nuisance -> log-mel -> mel shift -> SpecAugment.
        wav_long: (B, L) crops at least 1.1 x crop length (room for the speed change)."""
        x = self.speed_perturb(wav_long, self.crop)
        x = self.augment_wave(x, light=True)
        return self.spec_augment(self.mel_shift(self.logmel.to(x.device)(x)))

    # ---------------------------------------------------------------- public API
    def random_crops(self, wav):
        """Identity pair: one random crop and a second one shifted by at most +-0.5 s."""
        T = wav.shape[-1]
        if T <= self.crop:
            pad = torch.zeros(self.crop - T)
            c = torch.cat([wav, pad])
            return c, c.clone()
        s = int(torch.randint(0, T - self.crop + 1, (1,), generator=self.gen))
        d = int(torch.randint(-self.max_shift, self.max_shift + 1, (1,), generator=self.gen))
        s2 = min(max(0, s + d), T - self.crop)
        return wav[s:s + self.crop], wav[s2:s2 + self.crop]

    @torch.no_grad()
    def views_from_crops(self, c1, c2):
        """c1, c2: (B, T) raw crops (any device) -> two batches of augmented log-mels."""
        return self.view(c1), self.view(c2)

    @torch.no_grad()
    def forward(self, audio):
        """audio: path | numpy | tensor (T,) -> (view1, view2), each (1, 128, frames)."""
        if isinstance(audio, str):
            audio = load_audio(audio, self.sr)
        wav = to_tensor(audio)
        if wav.dim() == 2:                       # already a batch of crops
            return self.views_from_crops(wav, wav)
        c1, c2 = self.random_crops(wav)
        v1, v2 = self.views_from_crops(c1.unsqueeze(0), c2.unsqueeze(0))
        return v1[0], v2[0]


# ----------------------------------------------------------------------------- eval queries
def simulate_query(clean, noise_clips, babble_clips, rir_clips, rng, snr_db=5.0,
                   babble_share=0.5, sr=SR):
    """Build ONE organiser-style mystery query in numpy (used for dev sets only).

    clean: float32 waveform (already the 5 s snippet).  Noise = traffic/environment +
    cafe babble, mixed to `snr_db`, then the whole mixture is put through a heavy room.
    """
    from scipy.signal import fftconvolve
    T = len(clean)

    def seg(c):
        c = np.asarray(c, np.float32)
        if len(c) < T:
            c = np.tile(c, int(np.ceil(T / len(c))) + 1)
        s = rng.integers(0, len(c) - T + 1)
        return c[s:s + T]

    env = seg(noise_clips[rng.integers(len(noise_clips))])
    bab = seg(babble_clips[rng.integers(len(babble_clips))])
    rms = lambda v: np.sqrt(np.mean(v ** 2)) + 1e-8
    noise = (1 - babble_share) * env / rms(env) + babble_share * bab / rms(bab)
    noise = noise / rms(noise) * rms(clean) * 10 ** (-snr_db / 20)
    mix = clean + noise
    h = np.asarray(rir_clips[rng.integers(len(rir_clips))], np.float32)
    wet = fftconvolve(mix, h)[:T]
    wet = wet * rms(mix) / rms(wet)
    return (wet / max(1.0, np.abs(wet).max())).astype(np.float32)
