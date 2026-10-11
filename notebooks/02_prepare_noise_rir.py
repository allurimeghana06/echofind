# %% [markdown]
# # EchoFind · 02 — Noise, babble, room-impulse-response banks (train vs. eval, no leakage)
#
# **Kaggle settings:** Accelerator = None (CPU) · Internet = On · run as a commit.
#
# Key design rule: anything used to build evaluation queries or OOD tests is NEVER
# seen by the augmentation pipeline during training. Otherwise your retrieval and
# OOD numbers are optimistic and a reviewer will (rightly) discount them.
#
#   source                         TRAIN augmentation          EVAL queries / OOD
#   ESC-50 environmental sound     folds 1-4                   fold 5
#   babble ("cafe chatter")        LibriSpeech dev-clean       test-clean, speakers A
#   clean speech (OOD, Ext. B)     —                           test-clean, speakers B
#   room impulse responses         simulated + 50% real RIRs   other 50% real RIRs
#   real isotropic background      yes                         —
#
# Output → /kaggle/working/noise_rir/*.npz, each a ragged bank:
#   data (float16, concatenated clips at 22,050 Hz), offsets (int), optional labels.
# Noise and babble clips are unit-RMS, so SNR mixing later is just a gain.
# RIRs are aligned to the direct-path peak, truncated to 2 s, and unit-energy.

# %%
import os, io, json, zipfile, tarfile, subprocess, time, collections
import numpy as np
import pandas as pd
import soundfile as sf
import librosa

SR = 22050
OUT = "/kaggle/working/noise_rir"
DL = "/tmp/nr_dl"
os.makedirs(OUT, exist_ok=True)
os.makedirs(DL, exist_ok=True)
rng = np.random.default_rng(1234)

URLS = {
    "esc50": "https://github.com/karolpiczak/ESC-50/archive/master.zip",          # ~600 MB
    "ls_dev": "https://www.openslr.org/resources/12/dev-clean.tar.gz",             # ~340 MB
    "ls_test": "https://www.openslr.org/resources/12/test-clean.tar.gz",           # ~350 MB
    "rirs": "https://www.openslr.org/resources/28/rirs_noises.zip",                # ~1.3 GB
}

def download(url, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst
    print("downloading", url)
    r = subprocess.run(["wget", "-c", "--progress=dot:giga", "-O", dst, url])
    assert r.returncode == 0, f"download failed: {url}"
    return dst

def read_audio_bytes(b):
    x, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
    x = x[:, 0]                       # first channel only (averaging RIR channels would comb-filter)
    if sr != SR:
        x = librosa.resample(x, orig_sr=sr, target_sr=SR, res_type="soxr_hq")
    return x.astype(np.float32)

def unit_rms(x, eps=1e-8):
    return (x / (np.sqrt(np.mean(x ** 2)) + eps)).astype(np.float32)

def save_bank(name, clips, **labels):
    lens = np.array([len(c) for c in clips], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lens)])
    data = np.concatenate(clips).astype(np.float16)
    np.savez(f"{OUT}/{name}.npz", data=data, offsets=offsets, **labels)
    print(f"{name:18s} {len(clips):5d} clips  {data.size / SR / 60:7.1f} min  {data.nbytes / 1e6:6.0f} MB")

def load_bank(path):
    z = np.load(path)
    d, o = z["data"], z["offsets"]
    return [d[o[i]:o[i + 1]] for i in range(len(o) - 1)]

# %% ESC-50: environmental sounds (traffic, engines, horns, footsteps, doors, ...)
esc_zip = zipfile.ZipFile(download(URLS["esc50"], f"{DL}/esc50.zip"))
esc_meta = pd.read_csv(io.BytesIO(esc_zip.read("ESC-50-master/meta/esc50.csv")))

def load_esc(rows):
    clips, cats = [], []
    for r in rows.itertuples():
        x = read_audio_bytes(esc_zip.read(f"ESC-50-master/audio/{r.filename}"))
        x, _ = librosa.effects.trim(x, top_db=50)   # ESC-50 clips are zero-padded to 5 s
        if len(x) < SR:                             # skip clips with <1 s of actual sound
            continue
        clips.append(unit_rms(x))
        cats.append(r.category)
    return clips, np.array(cats)

t = time.time()
esc_train, esc_train_cat = load_esc(esc_meta[esc_meta.fold.isin([1, 2, 3, 4])])
esc_eval, esc_eval_cat = load_esc(esc_meta[esc_meta.fold == 5])
print(f"ESC-50: {len(esc_train)} train / {len(esc_eval)} eval clips ({time.time() - t:.0f} s)")

# %% LibriSpeech → synthetic babble (3-7 overlapping talkers ≈ cafe chatter)
def read_librispeech(tar_path, per_speaker=8):
    by_spk = collections.defaultdict(list)
    with tarfile.open(tar_path, "r:gz") as tf:
        for m in tf:
            if not m.name.endswith(".flac"):
                continue
            spk = m.name.split("/")[-3]     # LibriSpeech/<subset>/<speaker>/<chapter>/<file>.flac
            if len(by_spk[spk]) < per_speaker:
                by_spk[spk].append(read_audio_bytes(tf.extractfile(m).read()))
    return dict(by_spk)

def make_babble(by_spk, n_clips, dur_s=10.0, talkers=(3, 8)):
    spks, L, out = list(by_spk), int(dur_s * SR), []
    for _ in range(n_clips):
        k = int(rng.integers(*talkers))
        mix = np.zeros(L, np.float32)
        for spk in rng.choice(spks, size=min(k, len(spks)), replace=False):
            u = by_spk[spk][rng.integers(len(by_spk[spk]))]
            u = np.tile(u, int(np.ceil(L / len(u))) + 1)
            s = int(rng.integers(0, len(u) - L))
            mix += unit_rms(u[s:s + L]) * 10 ** (rng.uniform(-6, 0) / 20)
        out.append(unit_rms(mix))
    return out

t = time.time()
dev = read_librispeech(download(URLS["ls_dev"], f"{DL}/dev-clean.tar.gz"))
test = read_librispeech(download(URLS["ls_test"], f"{DL}/test-clean.tar.gz"))
test_spk = sorted(test)
rng.shuffle(test_spk)
spk_babble_eval = test_spk[: len(test_spk) // 2]
spk_ood = test_spk[len(test_spk) // 2:]          # disjoint speakers → clean-speech OOD set

babble_train = make_babble(dev, 800)
babble_eval = make_babble({s: test[s] for s in spk_babble_eval}, 200)
ood_speech = [unit_rms(u) for s in spk_ood for u in test[s]]
print(f"LibriSpeech: {len(dev)} dev speakers, {len(test)} test speakers ({time.time() - t:.0f} s)")

# %% OpenSLR-28: room impulse responses + real recorded background noise
rir_zip = zipfile.ZipFile(download(URLS["rirs"], f"{DL}/rirs_noises.zip"))
wavs = [n for n in rir_zip.namelist() if n.endswith(".wav")]
sim = {room: [n for n in wavs if f"simulated_rirs/{room}/" in n]
       for room in ["smallroom", "mediumroom", "largeroom"]}
real_dir = [n for n in wavs if "real_rirs_isotropic_noises/" in n]
real_rir = sorted(n for n in real_dir if "noise" not in os.path.basename(n).lower())
real_noise = sorted(n for n in real_dir if "noise" in os.path.basename(n).lower())
print({k: len(v) for k, v in sim.items()}, "| real RIRs:", len(real_rir), "| real noise files:", len(real_noise))

def prep_rir(x, max_s=2.0):
    peak = int(np.argmax(np.abs(x)))
    x = x[max(0, peak - int(0.0025 * SR)):][: int(max_s * SR)]   # start 2.5 ms before direct path
    return (x / (np.linalg.norm(x) + 1e-8)).astype(np.float32)

# Over-sample bigger rooms: the official test set uses heavy reverb
N_SIM = {"smallroom": 400, "mediumroom": 800, "largeroom": 1200}
rir_train, rir_train_kind = [], []
for room, k in N_SIM.items():
    for n in rng.choice(sim[room], size=min(k, len(sim[room])), replace=False):
        rir_train.append(prep_rir(read_audio_bytes(rir_zip.read(n))))
        rir_train_kind.append(room)

perm = rng.permutation(len(real_rir))
half = len(perm) // 2
for i in perm[:half]:
    rir_train.append(prep_rir(read_audio_bytes(rir_zip.read(real_rir[i]))))
    rir_train_kind.append("real")
rir_eval = [prep_rir(read_audio_bytes(rir_zip.read(real_rir[i]))) for i in perm[half:]]
# Caveat: a random split by file can put two RIRs from the same physical room on both
# sides. Mention this in the write-up, or split by room name if you want it airtight.

iso = []
for n in real_noise:
    x = read_audio_bytes(rir_zip.read(n))
    for s in range(0, len(x) - 10 * SR, 10 * SR):
        iso.append(unit_rms(x[s:s + 10 * SR]))
iso = [iso[i] for i in rng.permutation(len(iso))[:300]]

# %% Save all banks
save_bank("noise_env_train", esc_train + iso,
          category=np.concatenate([esc_train_cat, np.array(["isotropic_bg"] * len(iso))]))
save_bank("noise_env_eval", esc_eval, category=esc_eval_cat)
save_bank("babble_train", babble_train)
save_bank("babble_eval", babble_eval)
save_bank("rir_train", rir_train, kind=np.array(rir_train_kind))
save_bank("rir_eval", rir_eval)
save_bank("ood_speech", ood_speech)

json.dump({
    "sr": SR,
    "splits": {
        "noise_env_train": "ESC-50 folds 1-4 + OpenSLR-28 real isotropic noise (10 s chunks)",
        "noise_env_eval": "ESC-50 fold 5",
        "babble_train": "LibriSpeech dev-clean, 3-7 talkers, 10 s",
        "babble_eval": "LibriSpeech test-clean (speaker half A), 3-7 talkers, 10 s",
        "ood_speech": "LibriSpeech test-clean (speaker half B), single clean utterances",
        "rir_train": "OpenSLR-28 simulated (small 400 / medium 800 / large 1200) + 50% real",
        "rir_eval": "OpenSLR-28 other 50% real RIRs",
    },
    "seed": 1234,
}, open(f"{OUT}/README.json", "w"), indent=2)

# %% Listen: babble alone, then babble through an unseen real room
from IPython.display import Audio, display
from scipy.signal import fftconvolve

b = babble_eval[0]
display(Audio(b, rate=SR))
wet = fftconvolve(b, rir_eval[0])[: len(b)]
display(Audio(wet / (np.abs(wet).max() + 1e-8), rate=SR))
