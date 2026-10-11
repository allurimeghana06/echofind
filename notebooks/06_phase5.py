# %% [markdown]
# # EchoFind · 06 — Phase 5 experiments: B (OOD), C (variable length), A (latent synthesis)
#
# **Kaggle settings:** Accelerator = GPU T4 x1 · Internet = On · run as a commit.
# **Inputs:** outputs of 01, 02, 03, 04 (main run and, if done, the no-noise run) and 05.
#
# Order = priority. B is the main experiment, C is cheap, A is the stretch goal.
# Each section ends by writing a results table; the report is built from those tables.

# %% Setup
RUNS = {"main": "bt_main", "no_noise_aug": "bt_nonoise"}    # second one optional
REPO = "https://github.com/YOUR_USERNAME/echofind.git"     # <-- change this
import os, sys, subprocess
if not os.path.exists("/tmp/echofind"):
    url = REPO
    try:
        from kaggle_secrets import UserSecretsClient
        url = REPO.replace("https://", f"https://{UserSecretsClient().get_secret('GITHUB_TOKEN')}@")
    except Exception:
        pass
    subprocess.run(["git", "clone", "--depth", "1", url, "/tmp/echofind"], check=True)
sys.path.insert(0, "/tmp/echofind")

import glob, json, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F, matplotlib.pyplot as plt
from features import SR, LogMel
from data import FMAStore
from model import AudioEncoder, SpecDecoder
from evaluate import (embed_waves, retrieval_topk, knn_ood_scores, mahalanobis_scores,
                      ood_metrics, ten_percent_rows, linear_probe)
from kaggle_utils import find_file, load_ragged

dev = "cuda" if torch.cuda.is_available() else "cpu"
R = "/kaggle/working/results_phase5"
os.makedirs(R, exist_ok=True)

def load_run(run, which="best_encoder.pth"):
    hits = sorted(glob.glob(f"/kaggle/input/**/ckpt/{run}/{which}", recursive=True))
    if not hits:
        return None
    e = AudioEncoder(); e.load_state_dict(torch.load(hits[-1], map_location="cpu"))
    return e.to(dev).eval()

encoders = {k: load_run(v) for k, v in RUNS.items()}
encoders = {k: v for k, v in encoders.items() if v is not None}
encoders["random_init"] = AudioEncoder().to(dev).eval()
enc = encoders["main"]
print("encoders:", list(encoders))

store = FMAStore(find_file("audio_int16.npy"), find_file("meta.csv"))
meta = store.meta
tr_rows, val_rows = store.rows(["training"]), store.rows(["validation"])
rng = np.random.default_rng(0)
bank_rows = rng.choice(tr_rows, min(2000, len(tr_rows)), replace=False)          # in-distribution reference
dq = np.load(find_file("dev_queries.npz"))
q_waves = [w.astype(np.float32) for w in dq["wave"]]
sweep = np.load(find_file("dev_snr_sweep.npz"))
ood_clips, ood_lab = load_ragged(find_file("dev_ood.npz"))
ood_kind = np.array([k.split(":")[0] for k in ood_lab["kind"]])

def embed_rows(model, rws, mode="windows", chunk=256):
    return np.concatenate([embed_waves(model, [store.track(r) for r in rws[s:s + chunk]], dev, mode=mode)
                           for s in range(0, len(rws), chunk)])

# %% [markdown]
# ## Extension B — does the encoder know what is NOT music?
# Question: we trained the encoder to IGNORE cafe noise and traffic. Does that make it
# blind to non-music (bad for B), or does non-music land in an empty region (good)?
# In-distribution = noisy music at each SNR (the real use case), so a detector that
# just flags "noisy" is punished. Report per OOD type, kNN vs Mahalanobis, each encoder.

# %%
rows = []
for name, m in encoders.items():
    bank = embed_rows(m, bank_rows)
    oe = embed_waves(m, ood_clips, dev, mode="full")
    for cond in ("clean", "snr10", "snr5", "snr0"):
        qe = embed_waves(m, [w.astype(np.float32) for w in sweep[cond]], dev, mode="full")
        for method, fn in (("knn10", lambda X: knn_ood_scores(bank, X, 10)),
                           ("mahalanobis", lambda X: mahalanobis_scores(bank, X))):
            s_in, s_out = fn(qe), fn(oe)
            for kind in ["ALL"] + sorted(set(ood_kind)):
                mask = np.ones(len(oe), bool) if kind == "ALL" else ood_kind == kind
                om = ood_metrics(s_in, s_out[mask])
                rows.append({"encoder": name, "music_condition": cond, "method": method,
                             "ood_type": kind, **{k: round(v, 4) for k, v in om.items()}})
B = pd.DataFrame(rows); B.to_csv(f"{R}/extB_ood.csv", index=False)
print(B[(B.ood_type == "ALL")].pivot_table(index=["encoder", "method"], columns="music_condition",
                                           values="auroc").round(3).to_string())
print(B[(B.music_condition == "snr5") & (B.method == "knn10")]
      .pivot_table(index="ood_type", columns="encoder", values="auroc").round(3).to_string())

# %% Where do OOD clips sit? Score distributions for the main encoder
bank = embed_rows(enc, bank_rows)
oe = embed_waves(enc, ood_clips, dev, mode="full")
plt.figure(figsize=(8, 4))
for cond in ("clean", "snr5", "snr0"):
    qe = embed_waves(enc, [w.astype(np.float32) for w in sweep[cond]], dev, mode="full")
    plt.hist(knn_ood_scores(bank, qe, 10), bins=50, alpha=0.5, density=True, label=f"music {cond}")
for kind in ("speech", "env", "white_noise"):
    plt.hist(knn_ood_scores(bank, oe[ood_kind == kind], 10), bins=50, alpha=0.5, density=True,
             label=f"OOD {kind}", histtype="step", linewidth=2)
plt.xlabel("k-NN cosine distance (higher = more anomalous)"); plt.legend(); plt.tight_layout()
plt.savefig(f"{R}/extB_score_hist.png", dpi=130); plt.show()

# %% [markdown]
# ## Extension C — variable-length inference
# (1) Retrieval accuracy vs query length, clean and at 5 dB.
# (2) The spec's own test: cosine(5 s clip, 30 s clip) of the SAME track vs of DIFFERENT tracks.

# %%
lc, lc_lab = load_ragged(find_file("dev_lengths_clean.npz"))
ln, _ = load_ragged(find_file("dev_lengths_noisy.npz"))
rows_l, secs = lc_lab["row"], lc_lab["seconds"]
db = embed_waves(enc, [store.track(r) for r in val_rows], dev, mode="windows")
val_ids = meta.track_id.values[val_rows]
res = []
for name, clips in (("clean", lc), ("noisy_5dB", ln)):
    E = embed_waves(enc, clips, dev, mode="full")
    for L in sorted(set(secs)):
        m = secs == L
        r = retrieval_topk(E[m], meta.track_id.values[rows_l[m]], db, val_ids)
        res.append({"condition": name, "seconds": int(L), "top1": r["top1"], "n": int(m.sum())})
C = pd.DataFrame(res); C.to_csv(f"{R}/extC_length_curve.csv", index=False)
print(C.pivot(index="seconds", columns="condition", values="top1").round(3).to_string())

E = embed_waves(enc, lc, dev, mode="full")
e5 = E[secs == 5]; e30 = E[secs == 30]                 # same track order in both
S = e5 @ e30.T
same, diff = np.diag(S), S[~np.eye(len(S), dtype=bool)]
print(f"cos(5 s, 30 s) same track: {same.mean():.3f} ± {same.std():.3f}   "
      f"different tracks: {diff.mean():.3f} ± {diff.std():.3f}")
plt.figure(figsize=(6, 3.5))
plt.hist(diff, bins=60, alpha=0.6, density=True, label="different tracks")
plt.hist(same, bins=30, alpha=0.6, density=True, label="same track")
plt.xlabel("cosine similarity of 5 s vs 30 s embedding"); plt.legend(); plt.tight_layout()
plt.savefig(f"{R}/extC_5s_vs_30s.png", dpi=130); plt.show()
json.dump({"same_mean": float(same.mean()), "same_std": float(same.std()),
           "diff_mean": float(diff.mean()), "diff_std": float(diff.std())},
          open(f"{R}/extC_5s_vs_30s.json", "w"), indent=2)

# %% [markdown]
# ## Extension A — latent synthesis (stretch goal, honest version)
# Two decoders trained on the FROZEN encoder, both predicting the normalised log-mel of
# clean 5 s crops:
#   * "vector": from the pooled 512-d h (what the spec asks; no timing left)
#   * "map":    from the 512 x 4 x 14 map before pooling (keeps time)
# Then: morph two tracks of different genres, and compare against a plain spectrogram
# crossfade. Does the latent morph do anything a crossfade cannot?

# %%
DEC_EPOCHS, DEC_STEPS, DEC_BATCH = json.loads(os.environ.get("ECHOFIND_DEC", "[30, 100, 64]"))
lm = LogMel(normalize=False).to(dev)

def crop_batch(rws, n=DEC_BATCH):
    r = rng.choice(rws, n)
    out = []
    for x in r:
        s = rng.integers(0, store.n_samples[x] - 5 * SR)
        out.append(store.crop(x, s, 5 * SR))
    return torch.from_numpy(np.stack(out)).to(dev)

@torch.no_grad()
def targets(w):
    raw = lm(w)                                            # log-mel, not normalised
    mu = raw.mean(dim=(2, 3), keepdim=True); sd = raw.std(dim=(2, 3), keepdim=True)
    maps = enc.feature_maps(w)["layer4"]
    return (raw - mu) / (sd + 1e-5), maps, maps.mean(dim=(2, 3)), mu, sd

decoders = {"vector": SpecDecoder(from_vector=True).to(dev), "map": SpecDecoder(from_vector=False).to(dev)}
dec_log = []
for kind, D in decoders.items():
    opt = torch.optim.AdamW(D.parameters(), 1e-3, weight_decay=1e-4)
    for ep in range(DEC_EPOCHS):
        D.train(); tot = 0
        for _ in range(DEC_STEPS):
            y, maps, h, _, _ = targets(crop_batch(tr_rows))
            pred = D(h if kind == "vector" else maps)
            loss = F.l1_loss(pred, y)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        D.eval()
        with torch.no_grad():
            y, maps, h, _, _ = targets(crop_batch(val_rows, 128))
            vl = F.l1_loss(D(h if kind == "vector" else maps), y).item()
        dec_log.append({"decoder": kind, "epoch": ep + 1, "train_l1": tot / DEC_STEPS, "val_l1": vl})
        print(dec_log[-1])
pd.DataFrame(dec_log).to_csv(f"{R}/extA_decoder_training.csv", index=False)
torch.save({k: d.state_dict() for k, d in decoders.items()}, f"{R}/decoders.pt")
# Reference point: predicting all zeros (= the average normalised spectrogram) gives val L1 of
with torch.no_grad():
    y, *_ = targets(crop_batch(val_rows, 128))
    print("zero-prediction L1:", y.abs().mean().item())

# %% Morph two tracks of different genres
import librosa, soundfile as sf
from IPython.display import Audio, display
GENRE_A, GENRE_B = "Hip-Hop", "Folk"            # FMA-small has no Classical/Metal
pa = meta[(meta.split == "validation") & meta.ok & (meta.genre == GENRE_A)].row.values[0]
pb = meta[(meta.split == "validation") & meta.ok & (meta.genre == GENRE_B)].row.values[0]
wa = torch.from_numpy(store.crop(pa, 10 * SR, 5 * SR))[None].to(dev)
wb = torch.from_numpy(store.crop(pb, 10 * SR, 5 * SR))[None].to(dev)
ya, ma, ha, mua, sda = targets(wa)
yb, mb, hb, mub, sdb = targets(wb)

def to_audio(norm_spec, mu, sd):
    logmel = (norm_spec * sd + mu)[0, 0].float().cpu().numpy()
    mel = np.exp(logmel) - 1e-6
    return librosa.feature.inverse.mel_to_audio(np.maximum(mel, 0), sr=SR, n_fft=1024, hop_length=256,
                                                power=2.0, n_iter=64, fmin=20, fmax=SR / 2, htk=False, norm="slaney")

probe_rows = ten_percent_rows(meta, 0.10, seed=0)
clf = linear_probe(embed_rows(enc, probe_rows), meta.genre.values[probe_rows],
                   embed_rows(enc, val_rows[:50]), meta.genre.values[val_rows[:50]])["clf"]
classes = list(clf.classes_)

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
morph_rows = []
os.makedirs(f"{R}/morph_wavs", exist_ok=True)
fig, axs = plt.subplots(3, len(ALPHAS), figsize=(3 * len(ALPHAS), 7))
with torch.no_grad():
    for j, a in enumerate(ALPHAS):
        mu, sd = a * mua + (1 - a) * mub, a * sda + (1 - a) * sdb
        outs = {"latent_vector": decoders["vector"](a * ha + (1 - a) * hb),
                "latent_map": decoders["map"](a * ma + (1 - a) * mb),
                "crossfade": a * ya + (1 - a) * yb}
        for i, (name, spec) in enumerate(outs.items()):
            audio = to_audio(spec, mu, sd)
            sf.write(f"{R}/morph_wavs/{name}_alpha{a:.2f}.wav", audio, SR)
            e = embed_waves(enc, [audio.astype(np.float32)], dev, mode="full")
            p = clf.predict_proba(e)[0]
            morph_rows.append({"method": name, "alpha_A": a, f"p_{GENRE_A}": p[classes.index(GENRE_A)],
                               f"p_{GENRE_B}": p[classes.index(GENRE_B)]})
            axs[i, j].imshow(spec[0, 0].cpu(), origin="lower", aspect="auto")
            axs[i, j].set_title(f"{name}\nα={a}", fontsize=8); axs[i, j].axis("off")
plt.tight_layout(); plt.savefig(f"{R}/extA_morph_specs.png", dpi=120); plt.show()
M = pd.DataFrame(morph_rows); M.to_csv(f"{R}/extA_morph_probe.csv", index=False)
print(M.round(3).to_string(index=False))
for name in ("latent_vector", "latent_map", "crossfade"):
    print(name); display(Audio(f"{R}/morph_wavs/{name}_alpha0.50.wav"))
