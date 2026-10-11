# %% [markdown]
# # EchoFind · 05 — Phases 3 & 4: retrieval, linear probe, t-SNE/UMAP, build weights/
#
# **Kaggle settings:** Accelerator = GPU T4 x1 · Internet = On · run as a commit.
# **Inputs:** outputs of 01, 02, 03 and the training notebook 04.
#
# Produces /kaggle/working/weights/ = exactly what goes into the repo's weights/ folder:
#   encoder.pth, config.json, database.npz (clean FMA-small TEST set), ood_bank.npy
# plus results/*.csv tables and figures for the report.
#
# Rule: every decision (which database mode, which layer, which threshold) is made on
# VALIDATION data. The test split is touched once, at the end, for the final numbers.

# %% Setup
RUN_NAME = "bt_main"
CHECKPOINT = "best_encoder.pth"          # or "encoder.pth" (= last epoch)
REPO = "https://github.com/allurimeghana06/echofind.git"
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

import glob, json, shutil, time
import numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
from features import SR
from data import FMAStore
from model import AudioEncoder
from evaluate import (embed_waves, raw_mel_embed, retrieval_topk, wilson_interval,
                      ten_percent_rows, linear_probe, knn_ood_scores, ood_metrics)
from kaggle_utils import find_file, load_ragged

dev = "cuda" if torch.cuda.is_available() else "cpu"
W, R = "/kaggle/working/weights", "/kaggle/working/results"
os.makedirs(W, exist_ok=True); os.makedirs(R, exist_ok=True)

ckpt = sorted(glob.glob(f"/kaggle/input/**/ckpt/{RUN_NAME}/{CHECKPOINT}", recursive=True))[-1]
enc = AudioEncoder()
enc.load_state_dict(torch.load(ckpt, map_location="cpu"))
enc = enc.to(dev).eval()
rand_enc = AudioEncoder().to(dev).eval()          # baseline: same network, no training
print("loaded", ckpt)

store = FMAStore(find_file("audio_int16.npy"), find_file("meta.csv"))
meta = store.meta
val_rows, test_rows = store.rows(["validation"]), store.rows(["test"])
val_waves = [store.track(r) for r in val_rows]
val_ids = meta.track_id.values[val_rows]
dq = np.load(find_file("dev_queries.npz"))
q_waves, q_true = [w.astype(np.float32) for w in dq["wave"]], dq["track_id"]
sweep = np.load(find_file("dev_snr_sweep.npz"))

def embed_rows(model, rws, mode, layer="layer4", chunk=256):
    """Embed tracks in chunks so 6,400 x 30 s of audio never sits in RAM at once."""
    return np.concatenate([embed_waves(model, [store.track(r) for r in rws[s:s + chunk]], dev,
                                       mode=mode, layer=layer) for s in range(0, len(rws), chunk)])

def ci(p, n):
    lo, hi = wilson_interval(p, n)
    return f"{100 * p:.1f}% [{100 * lo:.0f}-{100 * hi:.0f}]"

# %% Phase 3 on the dev set: which way of embedding database tracks works best?
rows = []
for name, model in (("trained", enc), ("random_init", rand_enc)):
    q = embed_waves(model, q_waves, dev, mode="full")
    for mode in ("full", "windows", "multi"):
        db = embed_waves(model, val_waves, dev, mode=mode)
        r = retrieval_topk(q, q_true, db, val_ids)
        rows.append({"encoder": name, "db_mode": mode, "top1": r["top1"], "top5": r["top5"],
                     "top1_ci": ci(r["top1"], len(q_true))})
q = raw_mel_embed(q_waves); db = raw_mel_embed(val_waves)
r = retrieval_topk(q, q_true, db, val_ids)
rows.append({"encoder": "raw_mel", "db_mode": "full", "top1": r["top1"], "top5": r["top5"],
             "top1_ci": ci(r["top1"], len(q_true))})
ret = pd.DataFrame(rows); ret.to_csv(f"{R}/phase3_dev_modes.csv", index=False)
print(ret.to_string(index=False))

best = ret[(ret.encoder == "trained") & (ret.db_mode != "multi")].sort_values("top1").iloc[-1]
DB_MODE = best.db_mode          # multi-vector is reported but not shipped (spec: one vector per id)
print("chosen database mode:", DB_MODE)

# %% Phase 3: accuracy vs SNR (same snippets, same noise, only the level changes)
db = embed_waves(enc, val_waves, dev, mode=DB_MODE)
true = sweep["row"]
true_ids = meta.track_id.values[true]
snr_rows = []
for key in ("clean", "snr20", "snr10", "snr5", "snr0"):
    q = embed_waves(enc, [w.astype(np.float32) for w in sweep[key]], dev, mode="full")
    r = retrieval_topk(q, true_ids, db, val_ids)
    snr_rows.append({"condition": key, "top1": r["top1"], "top5": r["top5"]})
snr = pd.DataFrame(snr_rows); snr.to_csv(f"{R}/phase3_snr_sweep.csv", index=False)
print(snr.to_string(index=False))

# %% Phase 4: linear probe with 10% of the training labels (5 different 10% draws)
probe_cache = {}
def track_emb(rws, layer="layer4", mode="windows"):
    key = (np.asarray(rws).tobytes(), layer, mode)
    if key not in probe_cache:
        probe_cache[key] = embed_rows(enc, rws, mode, layer)
    return probe_cache[key]

abl = []
for layer in ("layer3", "layer4"):
    for mode in ("full", "windows"):
        Xv = track_emb(val_rows, layer, mode)
        f1s = []
        for seed in range(5):
            pr = ten_percent_rows(meta, 0.10, seed=seed)
            res = linear_probe(track_emb(pr, layer, mode), meta.genre.values[pr],
                               Xv, meta.genre.values[val_rows], seed=seed)
            f1s.append(res["f1_macro"])
        abl.append({"layer": layer, "pooling": mode, "val_f1_mean": np.mean(f1s), "val_f1_std": np.std(f1s)})
        print(abl[-1])
abl = pd.DataFrame(abl); abl.to_csv(f"{R}/phase4_layer_pooling_ablation.csv", index=False)
bp = abl.sort_values("val_f1_mean").iloc[-1]
PROBE_LAYER, PROBE_MODE = bp.layer, bp.pooling

# %% Phase 4 final: train on 10% (seed 0), report on the TEST split once
pr = ten_percent_rows(meta, 0.10, seed=0)
res = linear_probe(track_emb(pr, PROBE_LAYER, PROBE_MODE), meta.genre.values[pr],
                   track_emb(test_rows, PROBE_LAYER, PROBE_MODE), meta.genre.values[test_rows])
print(f"TEST macro-F1 = {res['f1_macro']:.3f}   accuracy = {res['acc']:.3f}   ({PROBE_LAYER}, {PROBE_MODE})")
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
print(classification_report(meta.genre.values[test_rows], res["pred"], digits=3))
ConfusionMatrixDisplay.from_predictions(meta.genre.values[test_rows], res["pred"],
                                        xticks_rotation=45, normalize="true", values_format=".2f")
plt.title("Linear probe, test split"); plt.tight_layout(); plt.savefig(f"{R}/confusion.png", dpi=130); plt.show()
import joblib; joblib.dump(res["clf"], f"{W}/genre_probe.joblib")

# %% Phase 4: t-SNE and UMAP of validation embeddings, coloured by genre
from sklearn.manifold import TSNE
Xv = track_emb(val_rows, PROBE_LAYER, PROBE_MODE)
gv = meta.genre.values[val_rows]
proj = {"t-SNE": TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(Xv)}
try:
    import umap
    proj["UMAP"] = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=0).fit_transform(Xv)
except ImportError:
    print("pip install umap-learn for the UMAP plot")
fig, axs = plt.subplots(1, len(proj), figsize=(7 * len(proj), 6))
axs = np.atleast_1d(axs)
for ax, (name, P) in zip(axs, proj.items()):
    for gname in sorted(set(gv)):
        m = gv == gname
        ax.scatter(P[m, 0], P[m, 1], s=8, label=gname, alpha=0.7)
    ax.set_title(f"{name} of validation embeddings"); ax.set_xticks([]); ax.set_yticks([])
axs[0].legend(markerscale=3, fontsize=8)
plt.tight_layout(); plt.savefig(f"{R}/latent_space.png", dpi=130); plt.show()

# %% Extension B prep: OOD bank (training-split music) + threshold calibrated on noisy music
tr_rows = store.rows(["training"])
bank = embed_rows(enc, tr_rows, DB_MODE)
np.save(f"{W}/ood_bank.npy", bank.astype(np.float16))
q_dev = embed_waves(enc, q_waves, dev, mode="full")
ood_clips, ood_lab = load_ragged(find_file("dev_ood.npz"))
oe = embed_waves(enc, ood_clips, dev, mode="full")
s_in, s_out = knn_ood_scores(bank, q_dev, 10), knn_ood_scores(bank, oe, 10)
om = ood_metrics(s_in, s_out)
print({k: round(v, 3) for k, v in om.items()})

# %% Build the TEST-set database the organisers will query, and the config
test_emb = embed_rows(enc, test_rows, DB_MODE)
np.savez(f"{W}/database.npz", ids=meta.track_id.values[test_rows].astype(np.int64),
         emb=test_emb.astype(np.float32))
shutil.copy(ckpt, f"{W}/encoder.pth")
config = {"db_mode": DB_MODE, "query_mode": "full", "win_sec": 5.0, "hop_sec": 2.5,
          "ood_k": 10, "ood_threshold": om["thr_accept99"],
          "probe_layer": PROBE_LAYER, "probe_pooling": PROBE_MODE,
          "trained_run": RUN_NAME, "checkpoint": CHECKPOINT}
json.dump(config, open(f"{W}/config.json", "w"), indent=2)
print(config)
for f in sorted(os.listdir(W)):
    print(f"{f:22s} {os.path.getsize(os.path.join(W, f)) / 1e6:7.1f} MB")

# %% End-to-end check with the real submission.py on the 50 dev .wav queries
import importlib
os.makedirs("/tmp/echofind/weights", exist_ok=True)
for f in os.listdir(W):
    shutil.copy(os.path.join(W, f), "/tmp/echofind/weights/")
import submission; importlib.reload(submission)
val_db = {int(t): v for t, v in zip(val_ids, embed_waves(enc, val_waves, dev, mode=DB_MODE))}
wav_dir = os.path.dirname(find_file("truth.csv"))
truth = pd.read_csv(f"{wav_dir}/truth.csv")
t = time.time()
pred = [submission.predict_track(f"{wav_dir}/{f}", val_db) for f in truth.file]
acc = np.mean(np.array(pred) == truth.track_id.values)
print(f"submission.predict_track on 50 dev wavs: {ci(acc, 50)}  ({(time.time() - t) / 50:.2f} s/query)")
