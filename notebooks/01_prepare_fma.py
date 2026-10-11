# %% [markdown]
# # EchoFind · 01 — Decode FMA-small into one fast waveform array
#
# **Kaggle settings:** Accelerator = None (CPU session → doesn't touch GPU quota) · Internet = On
#
# Why: decoding MP3s during training on Kaggle's few CPU cores would starve the GPU.
# We decode all 8,000 tracks ONCE (mono, 22,050 Hz, int16) into a single memory-mapped
# .npy. Later notebooks read random 5 s crops straight from disk in microseconds.
#
# Output → /kaggle/working/fma_small_wav/
#   audio_int16.npy  shape (8000, 661500) int16, ~10.6 GB → np.load(path, mmap_mode="r")
#   meta.csv         row, track_id, genre, split, n_samples, ok
#
# Run it as a commit: Save Version → "Save & Run All (Commit)" (outputs of interactive
# runs are not kept). In later notebooks: Add Input → Your Work → this notebook.
#
# If a Kaggle dataset with fma_small/ and tracks.csv is attached as input, it is used
# automatically; otherwise the official files are downloaded (~7.6 GB, can be slow).

# %%
import os, zipfile, subprocess, shutil, time
import numpy as np
import pandas as pd
from multiprocessing import Pool

SR = 22050
CLIP_SEC = 30
N_SAMP = SR * CLIP_SEC                 # 661,500 samples per track
MIN_OK_SAMPLES = 5 * SR                # need at least one full 5 s training crop
OUT = "/kaggle/working/fma_small_wav"
DL = "/tmp/fma_dl"                     # downloads to /tmp, NOT /kaggle/working (20 GB output cap)
os.makedirs(OUT, exist_ok=True)
os.makedirs(DL, exist_ok=True)

FMA_SMALL_URL = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
FMA_META_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"

def free_gb(path):
    return shutil.disk_usage(path).free / 1e9

print(f"/tmp free: {free_gb('/tmp'):.1f} GB | /kaggle/working free: {free_gb('/kaggle/working'):.1f} GB")
print("CPU cores:", os.cpu_count(), "| ffmpeg:", shutil.which("ffmpeg"))
assert shutil.which("ffmpeg"), "ffmpeg missing — run: !apt-get -y install ffmpeg"

# %% Locate or download sources
def find_dir(name, ok=lambda dirs, files: True):
    for root, dirs, files in os.walk("/kaggle/input"):
        if os.path.basename(root) == name and ok(dirs, files):
            return root
    return None

def find_file(name):
    for root, dirs, files in os.walk("/kaggle/input"):
        if name in files:
            return os.path.join(root, name)
    return None

def download(url, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print("already downloaded:", dst)
        return dst
    print("downloading", url)
    t = time.time()
    r = subprocess.run(["wget", "-c", "--progress=dot:giga", "-O", dst, url])
    assert r.returncode == 0, f"download failed: {url}"
    print(f"done in {(time.time() - t) / 60:.1f} min")
    return dst

local = find_dir("fma_small", lambda d, f: any(x.isdigit() and len(x) == 3 for x in d))
if local:
    SRC = ("dir", local)
else:
    SRC = ("zip", download(FMA_SMALL_URL, f"{DL}/fma_small.zip"))
print("audio source:", SRC)

tracks_csv = find_file("tracks.csv")
if tracks_csv is None:
    mz = download(FMA_META_URL, f"{DL}/fma_metadata.zip")
    with zipfile.ZipFile(mz) as z:
        z.extract("fma_metadata/tracks.csv", DL)
    tracks_csv = f"{DL}/fma_metadata/tracks.csv"
print("metadata:", tracks_csv)

# %% Metadata: track ids, top genre, official split
tracks = pd.read_csv(tracks_csv, index_col=0, header=[0, 1], low_memory=False)
small = tracks[tracks[("set", "subset")] == "small"]
meta = pd.DataFrame({
    "track_id": small.index.astype(int),
    "genre": small[("track", "genre_top")].astype(str).values,
    "split": small[("set", "split")].astype(str).values,
}).sort_values("track_id").reset_index(drop=True)
meta["row"] = np.arange(len(meta))
assert len(meta) == 8000, f"expected 8000 small tracks, got {len(meta)} — wrong tracks.csv?"
print(meta.genre.value_counts().to_string())
print(meta.split.value_counts().to_string())
# NOTE: no Classical, no Metal in FMA-small. Genre labels are only for Phase 4 / analysis,
# never for pretraining.

# %% Parallel decoding (each worker opens its own handle to the zip)
def rel_path(tid):
    s = f"{tid:06d}"
    return f"{s[:3]}/{s}.mp3"

_src, _zip = None, None

def _init(src):
    global _src, _zip
    _src = src
    if src[0] == "zip":
        _zip = zipfile.ZipFile(src[1])

def _read_bytes(tid):
    if _src[0] == "zip":
        return _zip.read("fma_small/" + rel_path(tid))
    with open(os.path.join(_src[1], rel_path(tid)), "rb") as f:
        return f.read()

def _decode(job):
    row, tid = job
    try:
        b = _read_bytes(tid)
        # ffmpeg: MP3 bytes in via stdin -> mono, 22,050 Hz, 16-bit PCM out via stdout
        p = subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-i", "pipe:0",
             "-f", "s16le", "-ac", "1", "-ar", str(SR), "pipe:1"],
            input=b, capture_output=True, timeout=120)
        x = np.frombuffer(p.stdout, dtype=np.int16)
        if x.size < SR:                      # <1 s decoded → treat as broken
            return row, None, int(x.size)
        return row, x[:N_SAMP].copy(), int(x.size)
    except Exception:
        return row, None, 0

audio = np.lib.format.open_memmap(f"{OUT}/audio_int16.npy", mode="w+",
                                  dtype=np.int16, shape=(len(meta), N_SAMP))
n_samples = np.zeros(len(meta), dtype=np.int64)
jobs = list(zip(meta.row.tolist(), meta.track_id.tolist()))

t0 = time.time()
with Pool(os.cpu_count(), initializer=_init, initargs=(SRC,)) as pool:
    for i, (row, x, n) in enumerate(pool.imap_unordered(_decode, jobs, chunksize=8), 1):
        n_samples[row] = n
        if x is not None:
            audio[row, :x.size] = x
            if x.size < N_SAMP:
                audio[row, x.size:] = 0      # explicit zero-pad for short tracks
        if i % 500 == 0 or i == len(jobs):
            el = time.time() - t0
            print(f"{i}/{len(jobs)}  {el / 60:.1f} min elapsed, ETA {el / i * (len(jobs) - i) / 60:.1f} min")
audio.flush()
del audio

meta["n_samples"] = n_samples
meta["ok"] = n_samples >= MIN_OK_SAMPLES
meta.to_csv(f"{OUT}/meta.csv", index=False)
bad = meta[~meta.ok]
print(f"\nusable tracks: {meta.ok.sum()} / {len(meta)}")
print("broken or too-short tracks (excluded downstream):")
print(bad[["track_id", "genre", "split", "n_samples"]].to_string(index=False))

# %% Sanity checks: listen to a few tracks, and time random-crop reads
from IPython.display import Audio, display

A = np.load(f"{OUT}/audio_int16.npy", mmap_mode="r")
ok_rows = meta.row[meta.ok].values
rng = np.random.default_rng(0)
for r in rng.choice(ok_rows, 3, replace=False):
    print(meta.loc[r, ["track_id", "genre"]].to_dict())
    display(Audio(A[r, : 10 * SR].astype(np.float32) / 32768, rate=SR))

# Simulates the training dataloader: 256 random 5 s crops
t = time.time()
crop = 5 * SR
for _ in range(256):
    r = rng.choice(ok_rows)
    s = rng.integers(0, min(meta.n_samples[r], N_SAMP) - crop)
    _ = np.asarray(A[r, s:s + crop])
print(f"256 random 5 s crops in {time.time() - t:.2f} s (first pass is slower while the disk cache warms)")

# %% Free /tmp (the zip is not part of the saved output anyway)
shutil.rmtree(DL, ignore_errors=True)
print("output size:", sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT)) / 1e9, "GB")
