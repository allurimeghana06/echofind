"""Evaluation helpers shared by the training hook, the comparison notebook and Phase 5.

Embedding
  embed_windows   - 5 s windows (hop 1 s) of each clip -> per-space window embeddings
  embed_clips     - one embedding per whole clip (global average pooling, any length)
  track_vectors   - mean of a track's window embeddings (used by the genre probe)
Phase 3
  MultiVectorIndex - FAISS inner-product index over ALL window vectors of all tracks;
                     a track's score = best-matching window, averaged over query windows
  retrieval_eval   - top-1 / top-5 + per-query correctness (for McNemar)
Phase 4
  ten_percent_rows, linear_probe, probe_repeated (several 10% draws)
Statistics
  wilson_interval, mcnemar
Extension B
  knn_ood_scores, mahalanobis_scores, ood_metrics
"""
import numpy as np
import torch
import torch.nn.functional as F

from features import SR, LogMel

try:
    import faiss                                  # pip install faiss-cpu
    HAVE_FAISS = True
except Exception:                                 # exact numpy/torch search gives identical results
    HAVE_FAISS = False


# ----------------------------------------------------------------------------- embedding
def window_starts(T, win, hop):
    if T <= win:
        return [0]
    s = list(range(0, T - win + 1, hop))
    if s[-1] != T - win:
        s.append(T - win)                         # always cover the end of the clip
    return s


@torch.no_grad()
def _run(encoder, chunk, device, spaces, amp=True):
    x = torch.from_numpy(np.stack(chunk)).to(device)
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp and str(device).startswith("cuda")):
        sub = encoder.subspaces(x)
    sub = {k: (v.float() if v is not None else None) for k, v in sub.items()}
    return {sp: encoder.combine(sub, sp).cpu() for sp in spaces}


@torch.no_grad()
def embed_windows(encoder, waves, device="cuda", spaces=None, win_sec=5.0, hop_sec=1.0, batch=64):
    """waves: list of 1-D float32 arrays. Returns {space: [ (K_i, d) float32 array per wave ]}.
    Clips shorter than a window are embedded whole (one 'window')."""
    encoder.eval()
    spaces = spaces or encoder.spaces
    win, hop = int(win_sec * SR), int(hop_sec * SR)
    pieces, owner = [], []
    for i, w in enumerate(waves):
        for s in window_starts(len(w), win, hop):
            pieces.append(w[s:s + win]); owner.append(i)
    owner = np.array(owner)
    out = {sp: [None] * len(pieces) for sp in spaces}
    by_len = {}
    for k, p in enumerate(pieces):
        by_len.setdefault(len(p), []).append(k)
    for L, idx in by_len.items():
        b = max(1, batch * win // max(L, 1))
        for s in range(0, len(idx), b):
            part = idx[s:s + b]
            res = _run(encoder, [pieces[k] for k in part], device, spaces)
            for sp in spaces:
                for j, k in enumerate(part):
                    out[sp][k] = res[sp][j]
    result = {}
    for sp in spaces:
        H = torch.stack(out[sp]).numpy().astype(np.float32)
        result[sp] = [H[owner == i] for i in range(len(waves))]
    return result


@torch.no_grad()
def embed_clips(encoder, waves, device="cuda", spaces=None, batch=32):
    """One embedding per whole clip (any length). Returns {space: (N, d) array}."""
    encoder.eval()
    spaces = spaces or encoder.spaces
    out = {sp: [None] * len(waves) for sp in spaces}
    by_len = {}
    for i, w in enumerate(waves):
        by_len.setdefault(len(w), []).append(i)
    ref = 5 * SR
    for L, idx in by_len.items():
        b = max(1, batch * ref // max(L, 1))
        for s in range(0, len(idx), b):
            part = idx[s:s + b]
            res = _run(encoder, [waves[i] for i in part], device, spaces)
            for sp in spaces:
                for j, i in enumerate(part):
                    out[sp][i] = res[sp][j]
    return {sp: torch.stack(v).numpy().astype(np.float32) for sp, v in out.items()}


def track_vectors(window_list):
    """Mean of each track's window embeddings, re-normalised -> (N, d)."""
    v = np.stack([w.mean(0) for w in window_list])
    return (v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)).astype(np.float32)


def raw_mel_windows(waves, win_sec=5.0, hop_sec=1.0):
    """Baseline with no learning: mean+std of the log-mel per 5 s window (256-d)."""
    lm = LogMel()
    win, hop = int(win_sec * SR), int(hop_sec * SR)
    res = []
    for w in waves:
        vs = []
        for s in window_starts(len(w), win, hop):
            m = lm(torch.from_numpy(w[s:s + win]))[0, 0]
            vs.append(torch.cat([m.mean(1), m.std(1)]))
        V = F.normalize(torch.stack(vs), dim=-1).numpy().astype(np.float32)
        res.append(V)
    return res


# ----------------------------------------------------------------------------- Phase 3
class MultiVectorIndex:
    """FAISS IndexFlatIP over every window vector of every database track.

    search(): each query window retrieves its k nearest windows; a track's score for that
    query window is its best window's similarity; scores are averaged over the query's
    windows (a 5 s query has one window, a 60 s query many -> Extension C for free).
    """

    def __init__(self, window_list, track_ids, k=256):
        self.ids = np.asarray(track_ids)
        self.M = np.concatenate(window_list).astype(np.float32)
        self.owner = np.concatenate([np.full(len(w), i) for i, w in enumerate(window_list)])
        self.k = min(k, len(self.M))
        if HAVE_FAISS:
            self.index = faiss.IndexFlatIP(self.M.shape[1])
            self.index.add(self.M)

    def _knn(self, Q):
        if HAVE_FAISS:
            return self.index.search(np.ascontiguousarray(Q, dtype=np.float32), self.k)
        S = torch.from_numpy(Q) @ torch.from_numpy(self.M).T
        v, i = S.topk(self.k, dim=1)
        return v.numpy(), i.numpy()

    def scores(self, query_windows):
        """query_windows: (K_q, d) -> (n_tracks,) score per database track."""
        D, I = self._knn(query_windows)
        n = len(self.ids)
        per = np.full((len(query_windows), n), -1.0, dtype=np.float32)   # unseen tracks: -1
        for r in range(len(query_windows)):
            tr = self.owner[I[r]]
            np.maximum.at(per[r], tr, D[r])
        return per.mean(0)

    def rank(self, query_windows, topk=5):
        s = self.scores(query_windows)
        order = np.argsort(-s)[:topk]
        return self.ids[order], s[order]


def retrieval_eval(index, query_window_list, true_ids, ks=(1, 5)):
    preds = [index.rank(q, max(ks))[0] for q in query_window_list]
    true_ids = np.asarray(true_ids)
    res = {f"top{k}": float(np.mean([t in p[:k] for p, t in zip(preds, true_ids)])) for k in ks}
    res["correct"] = np.array([p[0] == t for p, t in zip(preds, true_ids)])
    res["pred"] = np.array([p[0] for p in preds])
    return res


def wilson_interval(p, n, z=1.96):
    """95% confidence interval for an accuracy p measured on n queries."""
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar(correct_a, correct_b):
    """Exact McNemar test on paired per-query correctness. Returns (b, c, p_value):
    b = only A correct, c = only B correct. Small p -> the two systems really differ."""
    from scipy.stats import binomtest
    a, b_ = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    b = int((a & ~b_).sum()); c = int((~a & b_).sum())
    p = 1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue
    return b, c, float(p)


# ----------------------------------------------------------------------------- Phase 4
def ten_percent_rows(meta, frac=0.10, seed=0, split="training"):
    """Stratified 10% of the labelled training tracks (the only labels the probe may use)."""
    from sklearn.model_selection import train_test_split
    m = meta[(meta.split == split) & meta.ok]
    keep, _ = train_test_split(m.row.values, train_size=frac, stratify=m.genre.values,
                               random_state=seed)
    return np.sort(keep)


def linear_probe(Xtr, ytr, Xte, yte, seed=0):
    """Standardise -> logistic regression (C picked by 5-fold CV inside the 10% set only)."""
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


def probe_repeated(get_vectors, meta, eval_rows, draws=5):
    """Mean/std macro-F1 over several stratified 10% draws.
    get_vectors(rows) -> (len(rows), d) track embeddings."""
    Xe = get_vectors(eval_rows)
    ye = meta.genre.values[eval_rows]
    f1 = []
    for seed in range(draws):
        pr = ten_percent_rows(meta, 0.10, seed=seed)
        f1.append(linear_probe(get_vectors(pr), meta.genre.values[pr], Xe, ye, seed)["f1_macro"])
    return float(np.mean(f1)), float(np.std(f1)), f1


# ----------------------------------------------------------------------------- Extension B
def knn_ood_scores(bank, X, k=10):
    """Anomaly score = cosine distance to the k-th nearest in-distribution embedding."""
    bank_t = torch.from_numpy(np.ascontiguousarray(bank, dtype=np.float32))
    out = []
    for s in range(0, len(X), 1024):
        S = torch.from_numpy(np.ascontiguousarray(X[s:s + 1024], dtype=np.float32)) @ bank_t.T
        out.append(S.topk(min(k, S.shape[1]), dim=1).values[:, -1])
    return (1 - torch.cat(out)).numpy()


def mahalanobis_scores(bank, X, shrink=0.1):
    mu = bank.mean(0)
    C = np.cov(bank - mu, rowvar=False)
    C = (1 - shrink) * C + shrink * np.trace(C) / C.shape[0] * np.eye(C.shape[0])
    P = np.linalg.inv(C)
    d = X - mu
    return np.einsum("nd,de,ne->n", d, P, d)


def ood_metrics(scores_in, scores_out):
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.zeros(len(scores_in)), np.ones(len(scores_out))]
    s = np.r_[scores_in, scores_out]
    thr95 = np.quantile(scores_in, 0.95)
    return {"auroc": float(roc_auc_score(y, s)),
            "fpr_at_95tpr": float((scores_out <= thr95).mean()),
            "thr_accept99": float(np.quantile(scores_in, 0.99)),
            "ood_caught_at_thr99": float((scores_out > np.quantile(scores_in, 0.99)).mean())}
