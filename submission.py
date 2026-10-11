"""EchoFind submission interface (IEEE Impulse 2026).

Required by the problem statement:
    class AudioEncoder
    get_embedding(audio_path)                    -> L2-normalised numpy vector (512,)
    predict_track(noisy_audio_path, database)    -> track id of the best match

Model: one shared ResNet-18 trunk with two heads.
    get_embedding returns the full embedding e = [identity ; style] (unit length).
    predict_track uses only the IDENTITY subspace, searched with a FAISS index that holds
    several 5 s window vectors per track ({track_id: windows}).
    The genre probe (Phase 4) uses the STYLE subspace.

Extras:
    get_embedding(path, space="id" | "st" | "full")
    build_database(audio_paths_or_folder)        -> {track_id: (K, d_id) window vectors}
    save_database / load_database                -> weights/database.npz
    is_anomaly(audio_path)                        -> (bool, score), Extension B (if trained)

Files in weights/:
    encoder.pth     trained encoder weights (state_dict)
    config.json     subspace sizes, window settings, run name
    database.npz    pre-built index of the clean FMA-small test set
"""
import os
import json
import glob
import numpy as np
import torch

from model import AudioEncoder                       # re-exported for the organisers
from features import load_audio, SR
from evaluate import embed_windows, embed_clips, MultiVectorIndex, knn_ood_scores

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "weights")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_CONFIG = {"d_id": 256, "d_st": 256, "trunk_dim": 512,
                  "win_sec": 5.0, "hop_sec": 1.0, "retrieval_space": "id",
                  "ood_k": 10, "ood_threshold": None}

_ENCODER, _CONFIG, _INDEX_CACHE = None, None, {}


def _config():
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = dict(DEFAULT_CONFIG)
        p = os.path.join(WEIGHTS, "config.json")
        if os.path.exists(p):
            _CONFIG.update(json.load(open(p)))
    return _CONFIG


def load_encoder(path=os.path.join(WEIGHTS, "encoder.pth")):
    global _ENCODER
    if _ENCODER is None:
        c = _config()
        enc = AudioEncoder(c["d_id"], c["d_st"], c["trunk_dim"])
        if os.path.exists(path):
            state = torch.load(path, map_location="cpu")
            enc.load_state_dict(state["encoder"] if "encoder" in state else state)
        else:
            print(f"[EchoFind] WARNING: {path} not found, using random weights")
        _ENCODER = enc.to(DEVICE).eval()
    return _ENCODER


def get_embedding(audio_path, space="full"):
    """Embedding of an audio file of any length (2 s ... 60 s+), L2-normalised.
    space='full' (default, 512-d) | 'id' (identity subspace) | 'st' (style subspace)."""
    wave = load_audio(audio_path, SR)
    return embed_clips(load_encoder(), [wave], DEVICE, spaces=[space])[space][0]


def _windows(waves):
    c = _config()
    return embed_windows(load_encoder(), waves, DEVICE, spaces=[c["retrieval_space"]],
                         win_sec=c["win_sec"], hop_sec=c["hop_sec"])[c["retrieval_space"]]


def _track_id(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return int(stem) if stem.isdigit() else stem


def build_database(audio_paths):
    """Index clean tracks: list of paths or a folder (searched recursively).
    Returns {track_id: (K, d_id) array of window vectors}. Track id = file name stem."""
    if isinstance(audio_paths, str) and os.path.isdir(audio_paths):
        exts = ("*.mp3", "*.wav", "*.flac", "*.ogg")
        audio_paths = sorted(p for e in exts for p in glob.glob(os.path.join(audio_paths, "**", e),
                                                                recursive=True))
    db = {}
    for i in range(0, len(audio_paths), 16):
        chunk = audio_paths[i:i + 16]
        for p, w in zip(chunk, _windows([load_audio(x, SR) for x in chunk])):
            db[_track_id(p)] = w
    return db


def save_database(db, path=os.path.join(WEIGHTS, "database.npz")):
    ids = list(db.keys())
    lens = np.array([len(db[k]) for k in ids])
    np.savez(path, ids=np.array(ids), offsets=np.concatenate([[0], np.cumsum(lens)]),
             emb=np.concatenate([db[k] for k in ids]).astype(np.float16))


def load_database(path=os.path.join(WEIGHTS, "database.npz")):
    z = np.load(path, allow_pickle=False)
    ids, o, E = z["ids"], z["offsets"], z["emb"].astype(np.float32)
    conv = (lambda v: int(v)) if np.issubdtype(ids.dtype, np.integer) else str
    return {conv(t): E[o[i]:o[i + 1]] for i, t in enumerate(ids)}


def _index(database):
    if database is None:
        database = os.path.join(WEIGHTS, "database.npz")
    key = database if isinstance(database, str) else id(database)
    if key not in _INDEX_CACHE:
        if isinstance(database, str):
            database = build_database(database) if os.path.isdir(database) else load_database(database)
        ids = list(database.keys())
        wins = [np.atleast_2d(np.asarray(database[k], np.float32)) for k in ids]
        _INDEX_CACHE[key] = MultiVectorIndex(wins, ids)
    return _INDEX_CACHE[key]


def is_anomaly(audio_path=None, embedding=None):
    """Extension B: True if the input is far from all training music (k-NN distance).
    Needs weights/ood_bank.npy and an ood_threshold in config.json."""
    c = _config()
    bank_path = os.path.join(WEIGHTS, "ood_bank.npy")
    if not os.path.exists(bank_path) or c["ood_threshold"] is None:
        raise RuntimeError("Extension B is not calibrated in this submission")
    bank = np.load(bank_path).astype(np.float32)
    q = embedding if embedding is not None else get_embedding(audio_path)
    score = float(knn_ood_scores(bank, q[None].astype(np.float32), k=c["ood_k"])[0])
    return score > c["ood_threshold"], score


def predict_track(noisy_audio_path, database=None, check_ood=False):
    """Return the id of the database track that best matches the query.

    database: {track_id: vector or (K, d) window vectors} dict, a path to a .npz index,
              a folder of clean audio (indexed on the fly), or None (the bundled index of
              the FMA-small test set).
    """
    if check_ood and is_anomaly(noisy_audio_path)[0]:
        return "Unknown/Anomaly"
    q = _windows([load_audio(noisy_audio_path, SR)])[0]
    best = _index(database).rank(q, 1)[0][0]
    return best.item() if hasattr(best, "item") else best


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(predict_track(sys.argv[1]))
