# %% [markdown]
# # EchoFind · 03 — Practice ("dev") test sets + the small augmentation asset pack
#
# **Kaggle settings:** Accelerator = None (CPU) · Internet = On · run as a commit.
# **Inputs to add:** the output of notebook 01 (FMA) and notebook 02 (noise/RIR banks).
#
# Builds everything later notebooks are scored on, ONCE, with a fixed seed:
#   dev_queries.npz      500 organiser-style queries from VALIDATION tracks
#                        (5 s, traffic + cafe babble at ~5 dB, heavy unseen reverb)
#   dev_snr_sweep.npz    the same 500 snippets at SNR 20 / 10 / 5 / 0 dB and clean
#   dev_ood.npz          non-music clips for Extension B (env. sounds, speech, babble, noise)
#   dev_lengths.npz      clips of 2-60 s for Extension C
#   dev_wav/             50 queries as .wav + truth.csv, to test submission.py end to end
#   aug_assets.npz       ~12 MB of TRAIN noise/babble/RIRs to commit into the repo's assets/
#
# Only *_eval banks are used for queries and OOD. Training never sees them.

# %% Setup: get the code
REPO = "https://github.com/allurimeghana06/echofind.git"
import os, sys, subprocess
if not os.path.exists("/tmp/echofind"):
    url = REPO
    try:                                    # private repo? add a Kaggle Secret named GITHUB_TOKEN
        from kaggle_secrets import UserSecretsClient
        url = REPO.replace("https://", f"https://{UserSecretsClient().get_secret('GITHUB_TOKEN')}@")
    except Exception:
        pass
    subprocess.run(["git", "clone", "--depth", "1", url, "/tmp/echofind"], check=True)
sys.path.insert(0, "/tmp/echofind")

import numpy as np, pandas as pd, soundfile as sf
from features import SR
from augment import simulate_query
from data import FMAStore
from kaggle_utils import find_file, load_ragged, save_ragged

OUT = "/kaggle/working/dev_sets"
os.makedirs(f"{OUT}/dev_wav", exist_ok=True)
rng = np.random.default_rng(2026)

store = FMAStore(find_file("audio_int16.npy"), find_file("meta.csv"))
NR = os.path.dirname(find_file("rir_eval.npz"))
env_eval, env_lab = load_ragged(f"{NR}/noise_env_eval.npz")
bab_eval, _ = load_ragged(f"{NR}/babble_eval.npz")
rir_eval, _ = load_ragged(f"{NR}/rir_eval.npz")
speech, _ = load_ragged(f"{NR}/ood_speech.npz")
print(len(env_eval), "env clips |", len(bab_eval), "babble |", len(rir_eval), "RIRs |", len(speech), "speech")

# %% "Heavy reverb": keep the longer-decay half of the unseen real rooms
def rt60(h):
    """Rough RT60 from the Schroeder decay curve (T20 x 3)."""
    e = np.cumsum((h ** 2)[::-1])[::-1]
    db = 10 * np.log10(e / e[0] + 1e-12)
    i5, i25 = np.argmax(db < -5), np.argmax(db < -25)
    return 3 * (i25 - i5) / SR if i25 > i5 else 0.0

rts = np.array([rt60(h) for h in rir_eval])
heavy = [h for h, r in zip(rir_eval, rts) if r >= np.median(rts)]
print(f"eval RIR RT60: median {np.median(rts):.2f}s, using {len(heavy)} with RT60 >= median")

# Traffic-like categories from the held-out ESC-50 fold
TRAFFIC = {"car_horn", "engine", "siren", "train", "airplane", "helicopter", "footsteps",
           "door_wood_knock", "church_bells", "wind", "rain"}
cats = env_lab["category"]
traffic = [c for c, k in zip(env_eval, cats) if k in TRAFFIC] or env_eval
print(len(traffic), "traffic/street clips")

# %% Dev queries: 500 validation tracks, one random 5 s snippet each
Q = 5 * SR
val_rows = store.rows(["validation"])
q_rows = rng.choice(val_rows, size=min(500, len(val_rows)), replace=False)
starts = np.array([rng.integers(0, store.n_samples[r] - Q + 1) for r in q_rows])
clean = np.stack([store.crop(r, s, Q) for r, s in zip(q_rows, starts)])

def make(snr, seed):
    g = np.random.default_rng(seed)
    out = []
    for x in clean:
        out.append(simulate_query(x, traffic, bab_eval, heavy, g,
                                  snr_db=snr + g.uniform(-2, 2), babble_share=g.uniform(0.3, 0.7)))
    return np.stack(out)

queries = make(5.0, 1)
np.savez(f"{OUT}/dev_queries.npz", wave=queries.astype(np.float16), row=q_rows, start=starts,
         track_id=store.meta.track_id.values[q_rows])
sweep = {"clean": clean.astype(np.float16)}
for snr in (20, 10, 5, 0):
    sweep[f"snr{snr}"] = make(float(snr), 1).astype(np.float16)     # same noise draws per SNR
np.savez(f"{OUT}/dev_snr_sweep.npz", row=q_rows, **sweep)
print("queries:", queries.shape)

# %% Listen to two of them next to the clean snippet
from IPython.display import Audio, display
for i in range(2):
    display(Audio(clean[i], rate=SR)); display(Audio(queries[i], rate=SR))

# %% 50 queries as .wav files (exactly what the organisers will hand predict_track)
truth = []
for i in range(min(50, len(q_rows))):
    tid = int(store.meta.track_id.values[q_rows[i]])
    name = f"query_{i:03d}.wav"
    sf.write(f"{OUT}/dev_wav/{name}", queries[i], SR)
    truth.append((name, tid))
pd.DataFrame(truth, columns=["file", "track_id"]).to_csv(f"{OUT}/dev_wav/truth.csv", index=False)

# %% Extension B: out-of-distribution (non-music) set
def fix_len(x, L=Q):
    x = np.tile(x, int(np.ceil(L / len(x)))) if len(x) < L else x
    s = rng.integers(0, len(x) - L + 1)
    return x[s:s + L]

ood, ood_kind = [], []
for c, k in zip(env_eval, cats):
    ood.append(fix_len(c)); ood_kind.append(f"env:{k}")
for c in rng.permutation(len(speech))[:300]:
    ood.append(fix_len(speech[c])); ood_kind.append("speech")
for c in bab_eval[:100]:
    ood.append(fix_len(c)); ood_kind.append("babble")
for _ in range(50):
    ood.append(rng.standard_normal(Q).astype(np.float32) * 0.1); ood_kind.append("white_noise")
for _ in range(50):                        # very quiet room tone
    ood.append((rng.standard_normal(Q) * 1e-3).astype(np.float32)); ood_kind.append("near_silence")
# peak-normalise everything so loudness is not a giveaway
ood = [x / (np.abs(x).max() + 1e-8) * 0.5 for x in ood]
save_ragged(f"{OUT}/dev_ood.npz", ood, kind=np.array(ood_kind))
print(pd.Series([k.split(":")[0] for k in ood_kind]).value_counts().to_string())

# %% Extension C: length test (100 validation tracks; 60 s = the 30 s track looped twice)
LENS = [2, 3, 5, 10, 20, 30, 60]
len_rows = rng.choice(val_rows, size=min(100, len(val_rows)), replace=False)
clips, noisy, meta = [], [], []
g = np.random.default_rng(7)
for r in len_rows:
    full = store.track(r)
    for L in LENS:
        n = L * SR
        src = np.tile(full, 3)[:n] if n > len(full) else full
        s = 0 if n >= len(full) else rng.integers(0, len(full) - n + 1)
        x = src[s:s + n]
        clips.append(x)
        noisy.append(simulate_query(x, traffic, bab_eval, heavy, g, snr_db=5.0))
        meta.append((r, L, s))
meta = np.array(meta)
save_ragged(f"{OUT}/dev_lengths_clean.npz", clips, row=meta[:, 0], seconds=meta[:, 1], start=meta[:, 2])
save_ragged(f"{OUT}/dev_lengths_noisy.npz", noisy, row=meta[:, 0], seconds=meta[:, 1], start=meta[:, 2])

# %% Asset pack for the repo (from TRAIN banks only) -> commit as assets/aug_assets.npz
env_tr, _ = load_ragged(f"{NR}/noise_env_train.npz")
bab_tr, _ = load_ragged(f"{NR}/babble_train.npz")
rir_tr, rir_lab = load_ragged(f"{NR}/rir_train.npz")

def pack(clips, n, max_s):
    idx = rng.choice(len(clips), size=min(n, len(clips)), replace=False)
    return [clips[i][: int(max_s * SR)] for i in idx]

noise_p, bab_p, rir_p = pack(env_tr, 24, 6), pack(bab_tr, 12, 6), pack(rir_tr, 40, 1.5)
pk = {}
for name, cl in (("noise", noise_p), ("babble", bab_p), ("rir", rir_p)):
    lens = np.array([len(c) for c in cl])
    pk[f"{name}_data"] = np.concatenate(cl).astype(np.float16)
    pk[f"{name}_offsets"] = np.concatenate([[0], np.cumsum(lens)])
np.savez(f"{OUT}/aug_assets.npz", **pk)
print("aug_assets.npz:", os.path.getsize(f"{OUT}/aug_assets.npz") / 1e6, "MB  -> download it and put it in repo/assets/")

for f in sorted(os.listdir(OUT)):
    p = os.path.join(OUT, f)
    if os.path.isfile(p):
        print(f"{f:28s} {os.path.getsize(p) / 1e6:8.1f} MB")
