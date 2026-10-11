"""EchoFind submission interface (IEEE Impulse 2026).

Required by the problem statement:
    class AudioEncoder
    get_embedding(audio_path)                    -> L2-normalised numpy vector (512,)
    predict_track(noisy_audio_path, database)    -> track id of the best match

Extras:
    build_database(audio_paths_or_folder)        -> {track_id: vector}
    is_anomaly(audio_path)                       -> (bool, score)   (Extension B)

Files used (all in weights/):
    encoder.pth     trained encoder weights (state_dict)
    config.json     how tracks/queries are embedded + OOD threshold
    database.npz    pre-built index of the clean FMA-small test set (ids, emb)
    ood_bank.npy    embeddings of training music, for the anomaly detector
"""
import os
import json
import glob
import numpy as np
import torch

from model import AudioEncoder                      # re-exported for the organisers
from features import load_audio, SR
from evaluate import embed_waves, knn_ood_scores

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "weights")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_CONFIG = {
    "db_mode": "windows",       # how clean database tracks are embedded
    "query_mode": "full",       # how queries are embedded (any length, Extension C)
    "win_sec": 5.0, "hop_sec": 2.5,
    "ood_k": 10, "ood_threshold": None,
}

_ENCODER, _CONFIG, _DB, _OOD = None, None, None, None


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
        enc = AudioEncoder()
        if os.path.exists(path):
            state = torch.load(path, map_location="cpu")
            enc.load_state_dict(state.get("encoder", state) if isinstance(state, dict) else state)
        else:
            print(f"[EchoFind] WARNING: {path} not found, using random weights")
        _ENCODER = enc.to(DEVICE).eval()
    return _ENCODER


def _embed(waves, mode):
    c = _config()
    return embed_waves(load_encoder(), waves, device=DEVICE, mode=mode,
                       win_sec=c["win_sec"], hop_sec=c["hop_sec"])


def get_embedding(audio_path):
    """Embedding of an audio file of any length (2 s ... 60 s+), L2-normalised, shape (512,)."""
    wave = load_audio(audio_path, SR)
    return _embed([wave], _config()["query_mode"])[0]


def _track_id(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return int(stem) if stem.isdigit() else stem


def build_database(audio_paths, mode=None):
    """Index clean tracks: list of paths or a folder (searched recursively for audio files).
    Returns {track_id: vector}. Track id = file name without extension (FMA style 000123.mp3)."""
    if isinstance(audio_paths, str) and os.path.isdir(audio_paths):
        exts = ("*.mp3", "*.wav", "*.flac", "*.ogg")
        audio_paths = sorted(p for e in exts for p in glob.glob(os.path.join(audio_paths, "**", e),
                                                                recursive=True))
    mode = mode or _config()["db_mode"]
    db = {}
    for i in range(0, len(audio_paths), 32):
        chunk = audio_paths[i:i + 32]
        waves = [load_audio(p, SR) for p in chunk]
        for p, v in zip(chunk, _embed(waves, mode)):
            db[_track_id(p)] = v
    return db


def save_database(db, path=os.path.join(WEIGHTS, "database.npz")):
    ids = np.array(list(db.keys()))
    np.savez(path, ids=ids, emb=np.stack([db[k] for k in ids]).astype(np.float32))


def load_database(path=os.path.join(WEIGHTS, "database.npz")):
    z = np.load(path, allow_pickle=False)
    return {(int(i) if np.issubdtype(z["ids"].dtype, np.integer) else str(i)): v
            for i, v in zip(z["ids"], z["emb"])}


def _as_matrix(database):
    global _DB
    if database is None:
        if _DB is None:
            _DB = load_database()
        database = _DB
    elif isinstance(database, str):
        database = build_database(database) if os.path.isdir(database) else load_database(database)
    ids = list(database.keys())
    return ids, np.stack([np.asarray(database[k], np.float32) for k in ids])


def is_anomaly(audio_path=None, embedding=None):
    """Extension B: True if the input is far from all training music (k-NN distance)."""
    global _OOD
    c = _config()
    if _OOD is None:
        _OOD = np.load(os.path.join(WEIGHTS, "ood_bank.npy")).astype(np.float32)
    q = embedding if embedding is not None else get_embedding(audio_path)
    score = float(knn_ood_scores(_OOD, q[None].astype(np.float32), k=c["ood_k"])[0])
    thr = c["ood_threshold"]
    return (thr is not None and score > thr), score


def predict_track(noisy_audio_path, database=None, check_ood=False):
    """Return the track id whose clean embedding is most cosine-similar to the query.

    database: {track_id: vector} dict, a path to a .npz index, a folder of clean audio
              (indexed on the fly), or None (the bundled index of the FMA-small test set).
    check_ood: if True, returns "Unknown/Anomaly" for non-music (Extension B). Off by
               default so a very noisy music query is never rejected.
    """
    q = get_embedding(noisy_audio_path)
    if check_ood and is_anomaly(embedding=q)[0]:
        return "Unknown/Anomaly"
    ids, M = _as_matrix(database)
    return ids[int(np.argmax(M @ q))]


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(predict_track(sys.argv[1], check_ood="--ood" in sys.argv))
