"""Small helpers shared by the Kaggle notebooks: finding inputs, loading the dev sets."""
import os
import glob
import numpy as np


def find_file(name, roots=("/kaggle/input", "/kaggle/working")):
    """First file called `name` anywhere under the Kaggle input/working folders."""
    for root in roots:
        hits = sorted(glob.glob(os.path.join(root, "**", name), recursive=True))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"{name} not found — did you 'Add Input' the notebook that makes it?")


def find_dir_of(name):
    return os.path.dirname(find_file(name))


def load_ragged(path, prefix=""):
    """Ragged bank (data + offsets) -> list of float32 numpy arrays, plus any label arrays."""
    z = np.load(path, allow_pickle=False)
    d, o = z[prefix + "data"], z[prefix + "offsets"]
    clips = [d[o[i]:o[i + 1]].astype(np.float32) for i in range(len(o) - 1)]
    labels = {k: z[k] for k in z.files if k not in (prefix + "data", prefix + "offsets")}
    return clips, labels


def save_ragged(path, clips, dtype=np.float16, **labels):
    lens = np.array([len(c) for c in clips], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lens)])
    np.savez(path, data=np.concatenate(clips).astype(dtype), offsets=offsets, **labels)


def int16_to_float(a):
    return a.astype(np.float32) / 32768.0
