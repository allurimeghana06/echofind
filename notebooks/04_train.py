# %% [markdown]
# # EchoFind · 04 — Phase 2: Barlow Twins pretraining (with resume + live evaluation)
#
# **Kaggle settings:** Accelerator = **GPU T4 x1** (not P100: T4 has tensor cores for fp16)
# · Internet = On · Persistence: Files only · run as a commit ("Save & Run All").
# **Inputs to add:** outputs of notebooks 01 (FMA), 02 (noise/RIR), 03 (dev sets).
# To RESUME after a session ends: also add THIS notebook's previous output as an input.
# The newest checkpoint found anywhere in /kaggle/input or /kaggle/working is used.
#
# Every epoch: checkpoint + log line. Every EVAL_EVERY epochs: dev retrieval top-1,
# probe F1, OOD AUROC, alignment/uniformity -> so Extension B/C curves come for free.

# %% Config  (change RUN_NAME + NOISE_AUG=False for the Extension B ablation run)
CFG = dict(
    RUN_NAME="bt_main",
    EPOCHS=100,
    BATCH=128,
    LR=3e-4, WEIGHT_DECAY=1e-5, WARMUP_EPOCHS=5,
    PAIRS_PER_TRACK=2,            # each epoch sees every track twice with fresh crops
    CROP_SEC=5.0,
    PROJ_DIM=2048, LAMBDA=5e-3,
    TRAIN_SPLITS=["training", "validation", "test"],   # audio only, no labels (see guide)
    NOISE_AUG=True,               # False = no environmental noise/babble (ablation)
    EVAL_EVERY=5,
    MAX_HOURS=11.0,               # stop cleanly before Kaggle's 12 h session limit
    NUM_WORKERS=4,
    SMOKE_TEST=False,             # True: 500 tracks, 5 epochs, eval every epoch
)
import os, json
CFG.update(json.loads(os.environ.get("ECHOFIND_CFG", "{}")))   # optional overrides (testing)
if CFG["SMOKE_TEST"]:
    CFG.update(EPOCHS=5, EVAL_EVERY=1, RUN_NAME=CFG["RUN_NAME"] + "_smoke")

# %% Setup
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

import glob, json, math, time, shutil
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from features import SR
from augment import AugmentationPipeline
from data import FMAStore, SSLDataset
from model import AudioEncoder, Projector, barlow_twins_loss, alignment_uniformity, collapse_check
from evaluate import (embed_waves, retrieval_topk, ten_percent_rows, linear_probe,
                      knn_ood_scores, ood_metrics)
from kaggle_utils import find_file, load_ragged

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "")
torch.backends.cudnn.benchmark = True
T_START = time.time()

CK = f"/kaggle/working/ckpt/{CFG['RUN_NAME']}"
os.makedirs(CK, exist_ok=True)

# %% Data
store = FMAStore(find_file("audio_int16.npy"), find_file("meta.csv"))
rows = store.rows(CFG["TRAIN_SPLITS"])
if CFG["SMOKE_TEST"]:
    rows = np.random.default_rng(0).choice(rows, size=min(500, len(rows)), replace=False)
ds = SSLDataset(store, rows, CFG["CROP_SEC"], gpu_augment=True, pairs_per_track=CFG["PAIRS_PER_TRACK"])
dl = DataLoader(ds, batch_size=CFG["BATCH"], shuffle=True, drop_last=True,
                num_workers=CFG["NUM_WORKERS"], pin_memory=True, persistent_workers=True)
print(f"{len(rows)} tracks, {len(dl)} steps/epoch")

NR = os.path.dirname(find_file("rir_train.npz"))
aug = AugmentationPipeline(noise_bank=f"{NR}/noise_env_train.npz",
                           babble_bank=f"{NR}/babble_train.npz",
                           rir_bank=f"{NR}/rir_train.npz",
                           crop_seconds=CFG["CROP_SEC"],
                           p_noise=0.9 if CFG["NOISE_AUG"] else 0.0).banks_to(dev)

# %% Model, optimiser, schedule
enc = AudioEncoder().to(dev)
proj = Projector(512, CFG["PROJ_DIM"], CFG["PROJ_DIM"]).to(dev)
params = list(enc.parameters()) + list(proj.parameters())
opt = torch.optim.AdamW(params, lr=CFG["LR"], weight_decay=CFG["WEIGHT_DECAY"])
scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
TOTAL = CFG["EPOCHS"] * len(dl)
WARM = CFG["WARMUP_EPOCHS"] * len(dl)

def lr_at(step):
    if step < WARM:
        return CFG["LR"] * (step + 1) / WARM
    p = (step - WARM) / max(1, TOTAL - WARM)
    return CFG["LR"] * 0.5 * (1 + math.cos(math.pi * p))

# %% Resume from the newest checkpoint for this run, if any
start_epoch, log = 0, []
cands = glob.glob(f"/kaggle/input/**/ckpt/{CFG['RUN_NAME']}/last.pt", recursive=True) + \
        glob.glob(f"{CK}/last.pt")
best = None
for c in cands:
    try:
        e = torch.load(c, map_location="cpu", weights_only=False)["epoch"]
        if best is None or e > best[0]:
            best = (e, c)
    except Exception as ex:
        print("unreadable checkpoint", c, ex)
if best:
    st = torch.load(best[1], map_location="cpu", weights_only=False)
    enc.load_state_dict(st["encoder"]); proj.load_state_dict(st["projector"])
    opt.load_state_dict(st["opt"]); scaler.load_state_dict(st["scaler"])
    start_epoch, log = st["epoch"], st["log"]
    for f in ("log.csv", "encoder.pth", "best_encoder.pth"):        # carry files forward
        src = os.path.join(os.path.dirname(best[1]), f)
        if os.path.exists(src) and os.path.abspath(src) != os.path.abspath(f"{CK}/{f}"):
            shutil.copy(src, f"{CK}/{f}")
    print(f"resumed from {best[1]} at epoch {start_epoch}")
else:
    print("starting from scratch")

# %% Evaluation hook data (loaded once)
meta = store.meta
dq = np.load(find_file("dev_queries.npz"))
q_waves = [w.astype(np.float32) for w in dq["wave"]]
q_true = dq["track_id"]
val_rows = store.rows(["validation"])
val_waves = [store.track(r) for r in val_rows]
val_ids = meta.track_id.values[val_rows]
probe_rows = ten_percent_rows(meta, 0.10, seed=0)
probe_waves = [store.track(r) for r in probe_rows]
ood_clips, ood_lab = load_ragged(find_file("dev_ood.npz"))
if CFG["SMOKE_TEST"]:
    q_waves, q_true = q_waves[:100], q_true[:100]
print(f"hook: {len(q_waves)} queries, {len(val_rows)} val tracks, {len(probe_rows)} probe tracks, {len(ood_clips)} OOD")

def eval_hook():
    t = time.time()
    db = embed_waves(enc, val_waves, dev, mode="windows")
    q = embed_waves(enc, q_waves, dev, mode="full")
    r = retrieval_topk(q, q_true, db, val_ids)
    pe = embed_waves(enc, probe_waves, dev, mode="windows")
    pr = linear_probe(pe, meta.genre.values[probe_rows], db, meta.genre.values[val_rows])
    oe = embed_waves(enc, ood_clips, dev, mode="full")
    s_in, s_out = knn_ood_scores(pe, q, k=10), knn_ood_scores(pe, oe, k=10)
    om = ood_metrics(s_in, s_out)
    enc.train()
    return {"top1": r["top1"], "top5": r["top5"], "probe_f1": pr["f1_macro"],
            "ood_auroc": om["auroc"], "ood_fpr95": om["fpr_at_95tpr"], "eval_sec": time.time() - t}

# %% Training loop
def save(epoch, extra_best=None):
    st = {"epoch": epoch, "encoder": enc.state_dict(), "projector": proj.state_dict(),
          "opt": opt.state_dict(), "scaler": scaler.state_dict(), "log": log, "cfg": CFG}
    torch.save(st, f"{CK}/last.tmp"); os.replace(f"{CK}/last.tmp", f"{CK}/last.pt")
    torch.save(enc.state_dict(), f"{CK}/encoder.pth")
    pd.DataFrame(log).to_csv(f"{CK}/log.csv", index=False)

best_top1 = max([l.get("top1", -1) for l in log if l.get("top1") == l.get("top1")] + [-1])
step = start_epoch * len(dl)
for epoch in range(start_epoch, CFG["EPOCHS"]):
    enc.train(); proj.train()
    t0, tot, n = time.time(), 0.0, 0
    for c1, c2, _ in dl:
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        c1, c2 = c1.to(dev, non_blocking=True), c2.to(dev, non_blocking=True)
        v1, v2 = aug.views_from_crops(c1, c2)
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            h1, h2 = enc.embed(v1, normalize=False), enc.embed(v2, normalize=False)
            z1, z2 = proj(h1), proj(h2)
        loss, on, off = barlow_twins_loss(z1, z2, CFG["LAMBDA"])
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        scaler.step(opt); scaler.update()
        if not torch.isfinite(loss):
            raise RuntimeError("loss is NaN/inf — lower LR or check the inputs")
        tot += loss.item(); n += 1; step += 1
    a, u = alignment_uniformity(h1.detach(), h2.detach())
    row = {"epoch": epoch + 1, "loss": tot / n, "on_diag": on.item(), "off_diag": off.item(),
           "lr": lr_at(step), "align": a, "uniform": u, "emb_std": collapse_check(h1.detach()),
           "epoch_sec": time.time() - t0}
    if (epoch + 1) % CFG["EVAL_EVERY"] == 0 or epoch + 1 == CFG["EPOCHS"]:
        row.update(eval_hook())
        if row["top1"] > best_top1:
            best_top1 = row["top1"]
            torch.save(enc.state_dict(), f"{CK}/best_encoder.pth")
    log.append(row)
    save(epoch + 1)
    print(" | ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
          flush=True)
    if (time.time() - T_START) / 3600 > CFG["MAX_HOURS"]:
        print(f"Time budget reached after epoch {epoch + 1}. Re-run this notebook with its own "
              "output added as an input to continue.")
        break

# %% Plot the training curves
import matplotlib.pyplot as plt
L = pd.DataFrame(log)
cols = [c for c in ["loss", "emb_std", "align", "uniform", "top1", "probe_f1", "ood_auroc"] if c in L]
fig, axs = plt.subplots(1, len(cols), figsize=(3.2 * len(cols), 3))
for ax, c in zip(axs, cols):
    d = L[["epoch", c]].dropna()
    ax.plot(d.epoch, d[c], "o-" if c in ("top1", "probe_f1", "ood_auroc") else "-")
    ax.set_title(c); ax.set_xlabel("epoch")
plt.tight_layout(); plt.savefig(f"{CK}/curves.png", dpi=120); plt.show()
print("files in", CK, os.listdir(CK))
