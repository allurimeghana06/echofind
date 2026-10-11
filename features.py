"""Phase 1 front-end: audio loading + differentiable log-mel spectrogram.

Everything here is a torch op, so gradients can flow through it (the spec asks for a
"differentiable function"), and it runs on CPU or GPU, one clip or a batch.
"""
import numpy as np
import torch
import torch.nn as nn
import torchaudio

SR = 22050          # sample rate used everywhere in the project
N_FFT = 1024        # ~46 ms analysis window
HOP = 256           # ~11.6 ms hop  -> 5 s of audio = 431 frames
N_MELS = 128
F_MIN, F_MAX = 20.0, SR / 2


def load_audio(path, sr=SR, max_seconds=None):
    """Load any audio file (wav/mp3/flac/ogg) as mono float32 numpy at `sr`."""
    import librosa
    x, _ = librosa.load(path, sr=sr, mono=True, duration=max_seconds)
    return x.astype(np.float32)


def to_tensor(audio):
    """numpy / list / tensor  ->  float32 torch tensor (int16 is rescaled to [-1, 1])."""
    if isinstance(audio, torch.Tensor):
        return audio.float() if audio.dtype != torch.int16 else audio.float() / 32768.0
    a = np.asarray(audio)
    if a.dtype == np.int16:
        return torch.from_numpy(a.astype(np.float32) / 32768.0)
    return torch.from_numpy(a.astype(np.float32))


def instance_norm(spec, eps=1e-5):
    """Normalise each spectrogram to zero mean / unit std over (mel, time).

    Removes overall loudness differences, so a quiet phone recording and a loud
    studio master of the same song start from the same scale.
    spec: (..., M, F)
    """
    m = spec.mean(dim=(-2, -1), keepdim=True)
    s = spec.std(dim=(-2, -1), keepdim=True)
    return (spec - m) / (s + eps)


class LogMel(nn.Module):
    """waveform (T,) or (B, T)  ->  log-mel (B, 1, N_MELS, frames).

    frames = 1 + T // HOP   (5 s -> 431, 30 s -> 2584)
    """

    def __init__(self, sr=SR, n_fft=N_FFT, hop=HOP, n_mels=N_MELS, normalize=True):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels,
            f_min=F_MIN, f_max=F_MAX, power=2.0, center=True, norm="slaney", mel_scale="slaney")
        self.normalize = normalize

    def forward(self, wav):
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        # the STFT is done in float32 even under mixed precision (fp16 overflows here)
        with torch.autocast(device_type=wav.device.type, enabled=False):
            spec = self.mel(wav.float())
            spec = torch.log(spec + 1e-6)
        spec = spec.unsqueeze(1)                    # (B, 1, M, F)
        return instance_norm(spec) if self.normalize else spec
