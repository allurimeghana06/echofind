# %% [markdown]
# # EchoFind · 05 — Compare all conditions (hypothesis tests) + package the submission
#
# **Kaggle settings:** Accelerator = GPU T4 x2 · Internet = On · run as a commit.
# **Inputs:** outputs of 01, 02, 03 and every training run of notebook 04 you want compared
# (add each 04 version's output; runs are found by name under ckpt/).
#
# Answers, on VALIDATION data (500 dev queries, 800 tracks):
#   Q1  identity subspace retrieves better than style subspace?
#   Q2  style subspace classifies genre better than identity subspace?
#   Q3  HEADLINE: split model vs the two specialists (M-id for retrieval, M-st for genre)
#   Q4  effect of the split ratio (S-50 / S-70 / S-90)
#   Q5  HEADLINE: does the orthogonality penalty help? (S-50+orth vs S-50)
#   Q6  are the subspaces really different? (CKA between e_id and e_st)
# Then, for FINAL_CONDITION only, touches the TEST split once and builds weights/.

# %% Setup
CONDITIONS = ["M-id", "M-st", "S-50", "S-70", "S-90", "S-50+orth"]
SEEDS = [0, 1]                     # missing runs are skipped
FINAL_CONDITION = "S-50"           # set after reading the tables (S-best or S-best+orth)
FINAL_SEED = 0
PROBE_DRAWS = 5

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
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "faiss-cpu"], check=False)

import glob, json, shutil, time
import numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from data import FMAStore
from model import AudioEncoder, linear_cka
from evaluate import (embed_windows, track_vectors, raw_mel_windows, MultiVectorIndex,
                      retrieval_eval, wilson_interval, mcnemar, ten_percent_rows,
                      linear_probe, probe_repeated)
from kaggle_utils import find_file

dev = "cuda" if torch.cuda.is_available() else "cpu"
W, R = "/kaggle/working/weights", "/kaggle/working/results"
os.makedirs(W, exist_ok=True); os.makedirs(R, exist_ok=True)

def run_name(cond, seed):
    return f"{cond.replace('+', '_')}_s{seed}"

def load_run(cond, seed):
    hits = sorted(glob.glob(f"/kaggle/input/**/ckpt/{run_name(cond, seed)}/encoder.pth", recursive=True)
                  + glob.glob(f"/kaggle/working/ckpt/{run_name(cond, seed)}/encoder.pth"))
    if not hits:
        return None, None
    cfg = json.load(open(os.path.join(os.path.dirname(hits[-1]), "encoder_config.json")))
    e = AudioEncoder(cfg["d_id"], cfg["d_st"], cfg.get("trunk_dim", 512))
    e.load_state_dict(torch.load(hits[-1], map_location="cpu"))
    return e.to(dev).eval(), hits[-1]

store = FMAStore(find_file("audio_int16.npy"), find_file("meta.csv"))
meta = store.meta
val_rows, test_rows = store.rows(["validation"]), store.rows(["test"])
val_ids = meta.track_id.values[val_rows]
val_waves = [store.track(r) for r in val_rows]
dq = np.load(find_file("dev_queries.npz"))
q_waves, q_true = [w.astype(np.float32) for w in dq["wave"]], dq["track_id"]
NQ = len(q_true)
probe_rows_all = np.unique(np.concatenate([ten_percent_rows(meta, 0.10, s) for s in range(PROBE_DRAWS)]))

def ci(p):
    lo, hi = wilson_interval(p, NQ)
    return f"{100 * p:.1f} [{100 * lo:.0f}-{100 * hi:.0f}]"

# %% Embed every available run once (validation tracks, dev queries, probe tracks)
cache = {}
for cond in CONDITIONS:
    for seed in SEEDS:
        enc, path = load_run(cond, seed)
        if enc is None:
            continue
        t = time.time()
        sp = [s for s in enc.spaces if s != "full"]
        vw = embed_windows(enc, val_waves, dev, spaces=sp, hop_sec=1.0)
        qw = embed_windows(enc, q_waves, dev, spaces=sp, hop_sec=1.0)
        pw = embed_windows(enc, [store.track(r) for r in probe_rows_all], dev, spaces=sp, hop_sec=2.5)
        cache[(cond, seed)] = {"vw": vw, "qw": qw,
                               "pv": {s: dict(zip(probe_rows_all, track_vectors(pw[s]))) for s in sp},
                               "vv": {s: track_vectors(vw[s]) for s in sp}}
        print(f"embedded {cond} seed {seed} ({path}) in {time.time() - t:.0f}s")
        del enc; torch.cuda.empty_cache()
assert cache, "no trained runs found — add the notebook-04 outputs as inputs"

# Baselines with no learning / no training
rand = AudioEncoder(256, 256).to(dev).eval()
cache[("random-init S-50", 0)] = {
    "vw": embed_windows(rand, val_waves, dev, spaces=["id", "st"]),
    "qw": embed_windows(rand, q_waves, dev, spaces=["id", "st"])}
cache[("raw log-mel", 0)] = {"vw": {"raw": raw_mel_windows(val_waves)}, "qw": {"raw": raw_mel_windows(q_waves)}}

# %% Per-run, per-subspace scores (Q1, Q2, Q4, Q6)
rows, correct = [], {}
for (cond, seed), c in cache.items():
    for sp in c["vw"]:
        r = retrieval_eval(MultiVectorIndex(c["vw"][sp], val_ids), c["qw"][sp], q_true)
        correct[(cond, seed, sp)] = r["correct"]
        row = {"condition": cond, "seed": seed, "space": sp, "top1": r["top1"], "top5": r["top5"]}
        if "pv" in c:
            get = lambda rr, s=sp, cc=c: np.stack([cc["pv"][s][x] for x in rr]) if rr is not val_rows else cc["vv"][s]
            f1m, f1s, _ = probe_repeated(get, meta, val_rows, PROBE_DRAWS)
            row.update({"f1": f1m, "f1_sd_draws": f1s})
        rows.append(row)
    if "pv" in c and len(c["vw"]) == 2:
        A, B = np.concatenate(c["vw"]["id"]), np.concatenate(c["vw"]["st"])
        sel = np.random.default_rng(0).choice(len(A), size=min(8000, len(A)), replace=False)
        rows.append({"condition": cond, "seed": seed, "space": "CKA(id,st)", "top1": np.nan,
                     "f1": np.nan, "cka": linear_cka(A[sel], B[sel])})
T = pd.DataFrame(rows)
if "cka" not in T:
    T["cka"] = np.nan
if "f1" not in T:
    T["f1"] = np.nan
T.to_csv(f"{R}/per_run_per_space.csv", index=False)
summary = (T.groupby(["condition", "space"]).agg(top1=("top1", "mean"), top1_sd=("top1", "std"),
                                                f1=("f1", "mean"), f1_sd=("f1", "std"),
                                                cka=("cka", "mean"),
                                                n_seeds=("seed", "nunique")).round(4))
summary.to_csv(f"{R}/summary_by_condition.csv")
print(summary.to_string())

# %% Q3 HEADLINE: split models vs specialists, and specialists at matched size (PCA)
def mean_metric(cond, sp, col):
    d = T[(T.condition == cond) & (T.space == sp)][col]
    return (d.mean(), d.std()) if len(d) else (np.nan, np.nan)

head = []
spec_ret, spec_f1 = mean_metric("M-id", "id", "top1"), mean_metric("M-st", "st", "f1")
head.append({"system": "Two specialists (M-id + M-st)", "retrieval_top1": spec_ret[0], "genre_f1": spec_f1[0],
             "dims_per_task": "512 + 512", "trunks": 2})
for cond in [c for c in CONDITIONS if c.startswith("S-")]:
    r, f = mean_metric(cond, "id", "top1"), mean_metric(cond, "st", "f1")
    if not np.isnan(r[0]):
        dims = {"S-50": "256 + 256", "S-70": "358 + 154", "S-90": "461 + 51", "S-50+orth": "256 + 256"}
        head.append({"system": cond, "retrieval_top1": r[0], "genre_f1": f[0],
                     "dims_per_task": dims.get(cond, "?"), "trunks": 1})

# Specialists cut to 256 dims with PCA fitted on unlabelled validation windows (no retraining)
for cond, sp in (("M-id", "id"), ("M-st", "st")):
    if (cond, 0) not in cache:
        continue
    c = cache[(cond, 0)]
    pca = PCA(256, random_state=0).fit(np.concatenate(c["vw"][sp]))
    norm = lambda X: (X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
    if sp == "id":
        vw = [norm(pca.transform(w)) for w in c["vw"][sp]]
        qw = [norm(pca.transform(w)) for w in c["qw"][sp]]
        head.append({"system": "M-id cut to 256 (PCA)", "retrieval_top1":
                     retrieval_eval(MultiVectorIndex(vw, val_ids), qw, q_true)["top1"], "genre_f1": np.nan,
                     "dims_per_task": "256", "trunks": 1})
    else:
        get = lambda rr: norm(pca.transform(np.stack([c["pv"]["st"][x] for x in rr]))) if rr is not val_rows \
            else norm(pca.transform(c["vv"]["st"]))
        head.append({"system": "M-st cut to 256 (PCA)", "retrieval_top1": np.nan,
                     "genre_f1": probe_repeated(get, meta, val_rows, PROBE_DRAWS)[0],
                     "dims_per_task": "256", "trunks": 1})
H = pd.DataFrame(head); H.to_csv(f"{R}/headline_vs_specialists.csv", index=False)
print(H.round(4).to_string(index=False))

# %% Significance: McNemar on the same 500 queries (seed 0)
tests = []
def add_test(name, a, b):
    if a in correct and b in correct:
        bb, cc, p = mcnemar(correct[a], correct[b])
        tests.append({"comparison": name, "only_first_correct": bb, "only_second_correct": cc, "p_value": p})
add_test("S-50 e_id vs M-id (retrieval)", ("S-50", 0, "id"), ("M-id", 0, "id"))
add_test("S-50+orth e_id vs S-50 e_id (retrieval)  [HEADLINE: orthogonality]",
         ("S-50+orth", 0, "id"), ("S-50", 0, "id"))
add_test("S-70 e_id vs M-id (retrieval)", ("S-70", 0, "id"), ("M-id", 0, "id"))
add_test("S-90 e_id vs M-id (retrieval)", ("S-90", 0, "id"), ("M-id", 0, "id"))
add_test("M-id vs raw log-mel (sanity)", ("M-id", 0, "id"), ("raw log-mel", 0, "raw"))
Tst = pd.DataFrame(tests); Tst.to_csv(f"{R}/mcnemar_tests.csv", index=False)
print(Tst.to_string(index=False))
print("Genre F1 differences: compare means against f1_sd in the summary table "
      "(variation over seeds and 10% draws); report both.")

# %% Q5 HEADLINE table: orthogonality on vs off, both tasks, all spaces
orth = summary.reset_index()
orth = orth[orth.condition.isin(["S-50", "S-50+orth"])]
orth.to_csv(f"{R}/headline_orthogonality.csv", index=False)
print(orth.to_string(index=False))

# %% FINAL condition: SNR robustness, t-SNE of both subspaces, test split once, weights/
enc, path = load_run(FINAL_CONDITION, FINAL_SEED)
cfg = json.load(open(os.path.join(os.path.dirname(path), "encoder_config.json")))
fc = cache[(FINAL_CONDITION, FINAL_SEED)]
sweep = np.load(find_file("dev_snr_sweep.npz"))
idx_id = MultiVectorIndex(fc["vw"]["id"], val_ids)
snr = []
for key in ("clean", "snr20", "snr10", "snr5", "snr0"):
    qw = embed_windows(enc, [w.astype(np.float32) for w in sweep[key]], dev, spaces=["id"])["id"]
    snr.append({"condition": key, "top1": retrieval_eval(idx_id, qw, meta.track_id.values[sweep["row"]])["top1"]})
pd.DataFrame(snr).to_csv(f"{R}/final_snr_sweep.csv", index=False)
print(pd.DataFrame(snr).to_string(index=False))

from sklearn.manifold import TSNE
gv = meta.genre.values[val_rows]
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
for ax, sp in zip(axs, ("id", "st")):
    P = TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(fc["vv"][sp])
    for g in sorted(set(gv)):
        m = gv == g
        ax.scatter(P[m, 0], P[m, 1], s=8, alpha=0.7, label=g)
    ax.set_title(f"{FINAL_CONDITION}: {'identity' if sp == 'id' else 'style'} subspace (validation)")
    ax.set_xticks([]); ax.set_yticks([])
axs[1].legend(markerscale=3, fontsize=8)
plt.tight_layout(); plt.savefig(f"{R}/tsne_identity_vs_style.png", dpi=130); plt.show()

# Phase 4 official numbers: probe on the STYLE subspace, 10% draw seed 0, TEST split once
pr = ten_percent_rows(meta, 0.10, seed=0)
tw = embed_windows(enc, [store.track(r) for r in test_rows], dev, spaces=["id", "st"], hop_sec=1.0)
res = linear_probe(np.stack([fc["pv"]["st"][x] for x in pr]), meta.genre.values[pr],
                   track_vectors(tw["st"]), meta.genre.values[test_rows])
print(f"TEST macro-F1 (style subspace) = {res['f1_macro']:.3f}, accuracy = {res['acc']:.3f}")
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
print(classification_report(meta.genre.values[test_rows], res["pred"], digits=3))
ConfusionMatrixDisplay.from_predictions(meta.genre.values[test_rows], res["pred"], xticks_rotation=45,
                                        normalize="true", values_format=".2f")
plt.title(f"Genre probe on style subspace — test split"); plt.tight_layout()
plt.savefig(f"{R}/final_confusion_test.png", dpi=130); plt.show()
import joblib; joblib.dump(res["clf"], f"{W}/genre_probe.joblib")

# Phase 3 database: clean TEST split, identity windows every 1 s
ids = meta.track_id.values[test_rows].astype(np.int64)
lens = np.array([len(w) for w in tw["id"]])
np.savez(f"{W}/database.npz", ids=ids, offsets=np.concatenate([[0], np.cumsum(lens)]),
         emb=np.concatenate(tw["id"]).astype(np.float16))
shutil.copy(path, f"{W}/encoder.pth")
config = {**{k: cfg[k] for k in ("d_id", "d_st")}, "trunk_dim": cfg.get("trunk_dim", 512),
          "win_sec": 5.0, "hop_sec": 1.0, "retrieval_space": "id",
          "condition": FINAL_CONDITION, "seed": FINAL_SEED, "ood_k": 10, "ood_threshold": None}
json.dump(config, open(f"{W}/config.json", "w"), indent=2)
for f in sorted(os.listdir(W)):
    print(f"{f:22s} {os.path.getsize(os.path.join(W, f)) / 1e6:7.1f} MB")

# %% End-to-end check with the real submission.py on the 50 dev .wav queries
import importlib
os.makedirs("/tmp/echofind/weights", exist_ok=True)
for f in os.listdir(W):
    shutil.copy(os.path.join(W, f), "/tmp/echofind/weights/")
import submission; importlib.reload(submission)
val_db = {int(t): w for t, w in zip(val_ids, fc["vw"]["id"])}
wav_dir = os.path.dirname(find_file("truth.csv"))
truth = pd.read_csv(f"{wav_dir}/truth.csv")
t = time.time()
pred = [submission.predict_track(f"{wav_dir}/{f}", val_db) for f in truth.file]
acc = float(np.mean(np.array(pred) == truth.track_id.values))
lo, hi = wilson_interval(acc, len(truth))
print(f"submission.predict_track on {len(truth)} dev wavs: {100 * acc:.1f}% [{100 * lo:.0f}-{100 * hi:.0f}] "
      f"({(time.time() - t) / len(truth):.2f} s/query)")
e = submission.get_embedding(f"{wav_dir}/{truth.file[0]}")
print("get_embedding:", e.shape, "norm", float(np.linalg.norm(e)))
