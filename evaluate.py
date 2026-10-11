"""Evaluation helpers shared by the training hook, the evaluation notebook and Phase 5.

  embed_waves        - waveforms -> embeddings (full clip / mean of windows / multi-vector)
  retrieval_topk     - Phase 3 accuracy (top-1, top-5)
  linear_probe       - Phase 4 (10% labels, macro-F1)
  knn_ood_scores     - Extension B score; ood_metrics -> AUROC, FPR@95%TPR
  raw_mel_embed      - the "no learning at all" baseline
"""
import numpy as np
import torch
import torch.nn.functional as F

from features import SR, LogMel


# ----------------------------------------------------------------------------- embedding
def _windows(wave, win, hop):
    T = len(wave)
    if T <= win:
        return [wave]
    starts = list(range(0, T - win + 1, hop))
    if starts[-1] != T - win:
        starts.append(T - win)                 # always cover the end of the clip
    return [wave[s:s + win] for s in starts]


@torch.no_grad()
def embed_waves(encoder, waves, device="cuda", mode="full", win_sec=5.0, hop_sec=2.5,
                batch=32, layer="layer4", amp=True):
    """waves: list of 1-D float32 numpy arrays (any lengths).

    mode="full"    : one forward pass over the whole clip (global average pooling)
    mode="windows" : average of embeddings of overlapping win_sec windows
    mode="multi"   : keep every window embedding -> list of (K, D) arrays
    Returns (N, D) float32 L2-normalised array (or a list for "multi").
    """
    encoder.eval()
    win, hop = int(win_sec * SR), int(hop_sec * SR)
    use_amp = amp and str(device).startswith("cuda")

    def run(chunk):
        x = torch.from_numpy(np.stack(chunk)).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            h = encoder.embed(x, normalize=False, layer=layer)
        return h.float().cpu()

    if mode == "full":
        out = [None] * len(waves)
        by_len = {}
        for i, w in enumerate(waves):
            by_len.setdefault(len(w), []).append(i)
        for L, idx in by_len.items():
            b = max(1, batch * win // max(L, 1))
            for s in range(0, len(idx), b):
                part = idx[s:s + b]
                h = run([waves[i] for i in part])
                for j, i in enumerate(part):
                    out[i] = h[j]
        return F.normalize(torch.stack(out), dim=-1).numpy()

    pieces, owner = [], []
    for i, w in enumerate(waves):
        ws = _windows(w, win, hop)
        if len(ws[0]) < win:                  # short clip: embed as-is, alone
            pieces.append(ws[0]); owner.append(i)
        else:
            pieces += ws; owner += [i] * len(ws)
    owner = np.array(owner)
    same = [k for k, p in enumerate(pieces) if len(p) == win]
    other = [k for k in range(len(pieces)) if len(pieces[k]) != win]
    H = None                                   # feature size depends on the layer
    def put(idx, h):
        nonlocal H
        if H is None:
            H = torch.zeros(len(pieces), h.shape[-1])
        H[idx] = h
    for s in range(0, len(same), batch):
        part = same[s:s + batch]
        put(part, run([pieces[k] for k in part]))
    for k in other:
        put(k, run([pieces[k]])[0])
    H = F.normalize(H, dim=-1)
    if mode == "multi":
        return [H[owner == i].numpy() for i in range(len(waves))]
    agg = torch.stack([H[owner == i].mean(0) for i in range(len(waves))])
    return F.normalize(agg, dim=-1).numpy()


def raw_mel_embed(waves):
    """Baseline with no learning: mean and std of the log-mel over time (256-d)."""
    lm = LogMel()
    out = []
    for w in waves:
        s = lm(torch.from_numpy(w))[0, 0]
        out.append(torch.cat([s.mean(1), s.std(1)]))
    return F.normalize(torch.stack(out), dim=-1).numpy()


# ----------------------------------------------------------------------------- Phase 3
def similarity(q, db):
    """q: (Nq, D); db: (Nd, D) array, or list of (K, D) arrays (multi-vector -> max over K)."""
    if isinstance(db, (list, tuple)):
        return np.stack([(q @ m.T).max(1) for m in db], axis=1)
    return q @ db.T


def retrieval_topk(q_emb, q_true, db_emb, db_ids, ks=(1, 5)):
    """Fraction of queries whose true track id is in the top-k most similar database items."""
    sim = similarity(q_emb, db_emb)
    order = np.argsort(-sim, axis=1)
    ids = np.asarray(db_ids)[order]
    q_true = np.asarray(q_true)[:, None]
    res = {f"top{k}": float((ids[:, :k] == q_true).any(1).mean()) for k in ks}
    res["pred"] = ids[:, 0]
    return res


def wilson_interval(p, n, z=1.96):
    """95% confidence interval for an accuracy p measured on n queries."""
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# ----------------------------------------------------------------------------- Phase 4
def ten_percent_rows(meta, frac=0.10, seed=0, split="training"):
    """Stratified 10% of the labelled training tracks (the only labels the probe may use)."""
    from sklearn.model_selection import train_test_split
    m = meta[(meta.split == split) & meta.ok]
    keep, _ = train_test_split(m.row.values, train_size=frac, stratify=m.genre.values,
                               random_state=seed)
    return np.sort(keep)


def linear_probe(Xtr, ytr, Xte, yte, seed=0):
    """Standardise -> logistic regression (C picked by 5-fold CV on the 10% set only)."""
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import f1_score, accuracy_score
    clf = make_pipeline(StandardScaler(),
                        LogisticRegressionCV(Cs=np.logspace(-4, 1, 11), cv=5, max_iter=3000,
                                             scoring="f1_macro", random_state=seed))
    clf.fit(Xtr, ytr)
    p = clf.predict(Xte)
    return {"f1_macro": float(f1_score(yte, p, average="macro")),
            "acc": float(accuracy_score(yte, p)), "clf": clf, "pred": p}


# ----------------------------------------------------------------------------- Extension B
def knn_ood_scores(bank, X, k=10, exclude_self=False):
    """Anomaly score = cosine distance to the k-th nearest in-distribution embedding.
    bank: (N, D) normalised embeddings of training music. Higher score = more anomalous."""
    sims = []
    bank_t = torch.from_numpy(bank)
    for s in range(0, len(X), 1024):
        S = torch.from_numpy(X[s:s + 1024]) @ bank_t.T
        kk = k + 1 if exclude_self else k
        top = S.topk(kk, dim=1).values[:, -1]
        sims.append(top)
    return (1 - torch.cat(sims)).numpy()


def mahalanobis_scores(bank, X, shrink=0.1):
    """Alternative density score: Mahalanobis distance to one Gaussian fit on the bank."""
    mu = bank.mean(0)
    C = np.cov(bank - mu, rowvar=False)
    C = (1 - shrink) * C + shrink * np.trace(C) / C.shape[0] * np.eye(C.shape[0])
    P = np.linalg.inv(C)
    d = X - mu
    return np.einsum("nd,de,ne->n", d, P, d)


def ood_metrics(scores_in, scores_out):
    """AUROC (OOD = positive) and FPR@95%TPR (fraction of OOD accepted as music when 95% of
    music is accepted). Also the threshold that accepts 99% of in-distribution music."""
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.zeros(len(scores_in)), np.ones(len(scores_out))]
    s = np.r_[scores_in, scores_out]
    thr95 = np.quantile(scores_in, 0.95)
    return {"auroc": float(roc_auc_score(y, s)),
            "fpr_at_95tpr": float((scores_out <= thr95).mean()),
            "thr_accept99": float(np.quantile(scores_in, 0.99)),
            "ood_caught_at_thr99": float((scores_out > np.quantile(scores_in, 0.99)).mean())}
