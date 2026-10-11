# %% [markdown]
# # EchoFind · 04 — Train one experimental condition (two-head SSL)
#
# **Kaggle settings:** Accelerator = **GPU T4 x2** (code uses one GPU) · Internet = On ·
# Persistence: Files only · run as a commit ("Save & Run All").
# **Inputs to add:** outputs of notebooks 01 (FMA), 02 (noise/RIR), 03 (dev sets).
# To RESUME after a session ends: also add THIS notebook's previous output as an input.
#
# ONE notebook trains ALL conditions: pick CONDITION below and commit once per condition.
#
#   condition   heads (d_id/d_st)   losses                          role
#   M-id        512 / 0             identity                        baseline + retrieval specialist
#   M-st        0 / 512             style (+NN positives)           baseline + genre specialist
#   S-50        256 / 256           identity + style                main experimental condition
#   S-70        358 / 154           identity + style                split sweep
#   S-90        461 / 51            identity + style                split sweep
#   S-50+orth   256 / 256           identity + style + orthogonality  HEADLINE comparison vs S-50
#
# Every condition shares trunk, data (training split only), augmentations, batch size,
# epochs and evaluation, so differences come from the heads and losses alone.

# %% Choose the condition
CONDITION = "S-50"          # "M-id" | "M-st" | "S-50" | "S-70" | "S-90" | "S-50+orth"
SEED = 0                    # second seeds (week 6): SEED = 1

PRESETS = {
    "M-id":      dict(D_ID=512, D_ST=0,   ORTH_LAMBDA=0.0),
    "M-st":      dict(D_ID=0,   D_ST=512, ORTH_LAMBDA=0.0),
    "S-50":      dict(D_ID=256, D_ST=256, ORTH_LAMBDA=0.0),
    "S-70":      dict(D_ID=358, D_ST=154, ORTH_LAMBDA=0.0),
    "S-90":      dict(D_ID=461, D_ST=51,  ORTH_LAMBDA=0.0),
    "S-50+orth": dict(D_ID=256, D_ST=256, ORTH_LAMBDA=5.0),
}
CFG = dict(
    CONDITION=CONDITION, SEED=SEED, **PRESETS[CONDITION],
    EPOCHS=100,
    BATCH=96,                    # same for every condition (fair number of negatives)
    LR=1e-3, WEIGHT_DECAY=1e-5, WARMUP_EPOCHS=5,
    PAIRS_PER_TRACK=1,
    CROP_SEC=5.0, MAX_SHIFT_SEC=0.5,
    TAU_ID=0.05, TAU_ST=0.2, W_STYLE=1.0,
    NN_START_EPOCH=10, NN_RAMP_EPOCHS=10, NN_MAX_P=0.5, NN_QUEUE=4096,
    TRAIN_SPLITS=["training"],   # training split only (decision of 11 Oct)
    EVAL_EVERY=5,
    MAX_HOURS=11.0,
    NUM_WORKERS=4,
    SMOKE_TEST=False,            # True: 500 tracks, 12 epochs (NN starts at 10), eval every 3
    LIMIT_TRACKS=None,           # e.g. 200 for a quick debugging run
)
import os, json
_over = json.loads(os.environ.get("ECHOFIND_CFG", "{}"))       # optional overrides (testing)
CFG.update(PRESETS[_over.get("CONDITION", CFG["CONDITION"])])  # the condition always sets its heads
CFG.update(_over)
CFG["RUN_NAME"] = CFG.get("RUN_NAME") or f"{CFG['CONDITION'].replace('+', '_')}_s{CFG['SEED']}"
if CFG["SMOKE_TEST"]:
    CFG.update(EPOCHS=12, EVAL_EVERY=3, WARMUP_EPOCHS=1, RUN_NAME=CFG["RUN_NAME"] + "_smoke")
USE_ID, USE_ST = CFG["D_ID"] > 0, CFG["D_ST"] > 0
print(json.dumps(CFG, indent=1))

# %% Setup
REPO = "https://github.com/allurimeghana06/echofind.git"
import sys, subprocess
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

import glob, math, time, shutil, random
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from augment import AugmentationPipeline
from data import FMAStore, TripleViewDataset
from model import (AudioEncoder, Projector, nt_xent, style_loss, orthogonality_penalty,
                   NNQueue, nn_probability, alignment_uniformity, collapse_check, linear_cka)
from evaluate import (embed_windows, track_vectors, MultiVectorIndex, retrieval_eval,
                      ten_percent_rows, linear_probe)
from kaggle_utils import find_file, load_ragged

torch.manual_seed(CFG["SEED"]); np.random.seed(CFG["SEED"]); random.seed(CFG["SEED"])
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "")
torch.backends.cudnn.benchmark = True
T_START = time.time()
CK = f"/kaggle/working/ckpt/{CFG['RUN_NAME']}"
os.makedirs(CK, exist_ok=True)

# %% Data (training split only; labels are never read here)
store = FMAStore(find_file("audio_int16.npy"), find_file("meta.csv"))
rows = store.rows(CFG["TRAIN_SPLITS"])
if CFG["SMOKE_TEST"] or CFG["LIMIT_TRACKS"]:
    k = CFG["LIMIT_TRACKS"] or 500
    rows = np.random.default_rng(0).choice(rows, size=min(k, len(rows)), replace=False)
ds = TripleViewDataset(store, rows, CFG["CROP_SEC"], CFG["MAX_SHIFT_SEC"],
                       pairs_per_track=CFG["PAIRS_PER_TRACK"])
dl = DataLoader(ds, batch_size=CFG["BATCH"], shuffle=True, drop_last=True,
                num_workers=CFG["NUM_WORKERS"], pin_memory=True,
                persistent_workers=CFG["NUM_WORKERS"] > 0)
print(f"{len(rows)} training tracks, {len(dl)} steps/epoch")

NR = os.path.dirname(find_file("rir_train.npz"))
aug = AugmentationPipeline(noise_bank=f"{NR}/noise_env_train.npz", babble_bank=f"{NR}/babble_train.npz",
                           rir_bank=f"{NR}/rir_train.npz", crop_seconds=CFG["CROP_SEC"],
                           max_shift_seconds=CFG["MAX_SHIFT_SEC"], seed=CFG["SEED"]).banks_to(dev)

# %% Model, projectors, optimiser, schedule
enc = AudioEncoder(CFG["D_ID"], CFG["D_ST"]).to(dev)
proj_id = Projector(CFG["D_ID"]).to(dev) if USE_ID else None
proj_st = Projector(CFG["D_ST"]).to(dev) if USE_ST else None
params = list(enc.parameters()) + [p for m in (proj_id, proj_st) if m is not None for p in m.parameters()]
opt = torch.optim.AdamW(params, lr=CFG["LR"], weight_decay=CFG["WEIGHT_DECAY"])
scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
queue = NNQueue(CFG["NN_QUEUE"], 128, dev) if USE_ST else None
TOTAL, WARM = CFG["EPOCHS"] * len(dl), CFG["WARMUP_EPOCHS"] * len(dl)

def lr_at(step):
    if step < WARM:
        return CFG["LR"] * (step + 1) / WARM
    return CFG["LR"] * 0.5 * (1 + math.cos(math.pi * (step - WARM) / max(1, TOTAL - WARM)))

# %% Resume from the newest checkpoint of this run, if any
start_epoch, log = 0, []
cands = glob.glob(f"/kaggle/input/**/ckpt/{CFG['RUN_NAME']}/last.pt", recursive=True) + glob.glob(f"{CK}/last.pt")
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
    enc.load_state_dict(st["encoder"])
    if proj_id is not None: proj_id.load_state_dict(st["proj_id"])
    if proj_st is not None: proj_st.load_state_dict(st["proj_st"])
    opt.load_state_dict(st["opt"]); scaler.load_state_dict(st["scaler"])
    start_epoch, log = st["epoch"], st["log"]
    if queue is not None and st.get("queue") is not None:
        queue.z, queue.ids = st["queue"]["z"].to(dev), st["queue"]["ids"].to(dev)
        queue.ptr, queue.filled = st["queue"]["ptr"], st["queue"]["filled"]
    for f in ("log.csv", "encoder.pth", "encoder_config.json"):
        src = os.path.join(os.path.dirname(best[1]), f)
        if os.path.exists(src) and os.path.abspath(src) != os.path.abspath(f"{CK}/{f}"):
            shutil.copy(src, f"{CK}/{f}")
    print(f"resumed from {best[1]} at epoch {start_epoch}")
else:
    print("starting from scratch")
json.dump({**enc.config(), "condition": CFG["CONDITION"], "seed": CFG["SEED"]},
          open(f"{CK}/encoder_config.json", "w"), indent=1)

# %% Evaluation hook (validation split only; loaded once)
meta = store.meta
dq = np.load(find_file("dev_queries.npz"))
q_waves, q_true = [w.astype(np.float32) for w in dq["wave"]], dq["track_id"]
val_rows = store.rows(["validation"])
val_ids = meta.track_id.values[val_rows]
probe_rows = ten_percent_rows(meta, 0.10, seed=0)
if CFG["SMOKE_TEST"]:
    q_waves, q_true = q_waves[:100], q_true[:100]

def eval_hook():
    t = time.time()
    spaces = [s for s in ("id", "st") if (s == "id" and USE_ID) or (s == "st" and USE_ST)]
    vw = embed_windows(enc, [store.track(r) for r in val_rows], dev, spaces=spaces, hop_sec=1.0)
    qw = embed_windows(enc, q_waves, dev, spaces=spaces, hop_sec=1.0)
    pw = embed_windows(enc, [store.track(r) for r in probe_rows], dev, spaces=spaces, hop_sec=2.5)
    out = {}
    for sp in spaces:
        r = retrieval_eval(MultiVectorIndex(vw[sp], val_ids), qw[sp], q_true)
        pr = linear_probe(track_vectors(pw[sp]), meta.genre.values[probe_rows],
                          track_vectors(vw[sp]), meta.genre.values[val_rows])
        out[f"top1_{sp}"], out[f"f1_{sp}"] = r["top1"], pr["f1_macro"]
    if len(spaces) == 2:
        A, B = np.concatenate(vw["id"]), np.concatenate(vw["st"])
        sel = np.random.default_rng(0).choice(len(A), size=min(4000, len(A)), replace=False)
        out["cka_id_st"] = linear_cka(A[sel], B[sel])
    out["eval_sec"] = time.time() - t
    enc.train()
    return out

# %% Training loop
def save(epoch):
    st = {"epoch": epoch, "encoder": enc.state_dict(), "opt": opt.state_dict(),
          "scaler": scaler.state_dict(), "log": log, "cfg": CFG,
          "proj_id": proj_id.state_dict() if proj_id is not None else None,
          "proj_st": proj_st.state_dict() if proj_st is not None else None,
          "queue": None if queue is None else {"z": queue.z.cpu(), "ids": queue.ids.cpu(),
                                               "ptr": queue.ptr, "filled": queue.filled}}
    torch.save(st, f"{CK}/last.tmp"); os.replace(f"{CK}/last.tmp", f"{CK}/last.pt")
    torch.save(enc.state_dict(), f"{CK}/encoder.pth")
    pd.DataFrame(log).to_csv(f"{CK}/log.csv", index=False)

B = CFG["BATCH"]
step = start_epoch * len(dl)
for epoch in range(start_epoch, CFG["EPOCHS"]):
    enc.train()
    for m in (proj_id, proj_st):
        if m is not None: m.train()
    p_nn = nn_probability(epoch, CFG["NN_START_EPOCH"], CFG["NN_RAMP_EPOCHS"], CFG["NN_MAX_P"]) if USE_ST else 0.0
    t0, sums, n = time.time(), {"loss": 0.0, "l_id": 0.0, "l_st": 0.0, "l_orth": 0.0}, 0
    for cA, cA2, cB, rws in dl:
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        cA = cA.to(dev, non_blocking=True)
        views = [aug.view(cA)]
        if USE_ID:
            views.append(aug.view(cA2.to(dev, non_blocking=True)))
        if USE_ST:
            views.append(aug.style_view(cB.to(dev, non_blocking=True)))
        rws = rws.to(dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            sub = enc.subspaces(torch.cat(views))
            loss = torch.zeros((), device=dev)
            l_id = l_st = l_orth = torch.zeros((), device=dev)
            if USE_ID:
                e = sub["id"].split(B)
                l_id = nt_xent(proj_id(e[0]), proj_id(e[1]), CFG["TAU_ID"])
                loss = loss + l_id
            if USE_ST:
                e = sub["st"].split(B)
                s1, s3 = proj_st(e[0]), proj_st(e[-1])
                l_st = style_loss(s1, s3, CFG["TAU_ST"], queue, rws, p_nn)
                loss = loss + CFG["W_STYLE"] * l_st
            if USE_ID and USE_ST:
                l_orth = orthogonality_penalty(sub["id"], sub["st"])   # tracked in every split run
                loss = loss + CFG["ORTH_LAMBDA"] * l_orth
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        scaler.step(opt); scaler.update()
        if not torch.isfinite(loss):
            raise RuntimeError("loss is NaN/inf — lower LR or check the inputs")
        if USE_ST:
            queue.add(s1.detach(), rws)
        for k, v in (("loss", loss), ("l_id", l_id), ("l_st", l_st), ("l_orth", l_orth)):
            sums[k] += float(v)
        n += 1; step += 1
    row = {"epoch": epoch + 1, **{k: v / n for k, v in sums.items()}, "p_nn": p_nn,
           "lr": lr_at(step), "epoch_sec": time.time() - t0}
    for sp in ("id", "st"):
        if sub[sp] is not None:
            h = sub[sp].detach().split(B)
            a, u = alignment_uniformity(h[0], h[1] if sp == "id" else h[-1])
            row.update({f"align_{sp}": a, f"unif_{sp}": u, f"std_{sp}": collapse_check(h[0])})
    if (epoch + 1) % CFG["EVAL_EVERY"] == 0 or epoch + 1 == CFG["EPOCHS"]:
        row.update(eval_hook())
    log.append(row)
    save(epoch + 1)
    print(" | ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
    if (time.time() - T_START) / 3600 > CFG["MAX_HOURS"]:
        print(f"Time budget reached after epoch {epoch + 1}. Re-run with this notebook's output as an input.")
        break

# %% Curves
import matplotlib.pyplot as plt
L = pd.DataFrame(log)
cols = [c for c in ["l_id", "l_st", "l_orth", "std_id", "std_st", "top1_id", "top1_st",
                    "f1_id", "f1_st", "cka_id_st"] if c in L and L[c].notna().any()]
fig, axs = plt.subplots(2, math.ceil(len(cols) / 2), figsize=(3.4 * math.ceil(len(cols) / 2), 6))
for ax, c in zip(axs.flat, cols):
    d = L[["epoch", c]].dropna()
    ax.plot(d.epoch, d[c], "o-" if c.startswith(("top1", "f1", "cka")) else "-")
    if USE_ST: ax.axvline(CFG["NN_START_EPOCH"], color="grey", ls=":", lw=1)
    ax.set_title(c); ax.set_xlabel("epoch")
for ax in list(axs.flat)[len(cols):]:
    ax.axis("off")
plt.suptitle(f"{CFG['RUN_NAME']}  (dotted line: NN positives start)")
plt.tight_layout(); plt.savefig(f"{CK}/curves.png", dpi=120); plt.show()
print("files in", CK, os.listdir(CK))
