"""Phase 1: the SSL Dataset (three views per track for the two-head model).

Audio comes from the single memory-mapped array made by notebook 01
(audio_int16.npy, shape (8000, 661500)) plus meta.csv. Reading a random 5 s crop
from it takes microseconds, so the data loader never starves the GPU.

Each item = three raw crops of ONE track:
  cA   5 s at a random moment                      -> view v1 (both heads)
  cA2  5 s at the same moment, shifted <= 0.5 s     -> view v2 (identity positive)
  cB   5.5 s from a NON-overlapping part of the track -> view v3 (style positive;
       the extra 0.5 s leaves room for speed perturbation)

The training loop moves the batch to the GPU and builds the augmented views there.
With gpu_augment=False and a pipeline, the dataset returns augmented views directly
(the CPU path the problem statement describes).
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from features import SR


class FMAStore:
    """Thin wrapper around the decoded FMA array + metadata."""

    def __init__(self, audio_npy, meta_csv):
        self.audio = np.load(audio_npy, mmap_mode="r")
        self.meta = pd.read_csv(meta_csv)
        self.n_samples = np.minimum(self.meta.n_samples.values, self.audio.shape[1])

    def rows(self, splits=("training",), only_ok=True):
        m = self.meta
        keep = m.split.isin(splits)
        if only_ok:
            keep &= m.ok
        return m.row[keep].values

    def track(self, row):
        """Full clean track as float32 numpy (trailing zero padding removed)."""
        return self.audio[row, :self.n_samples[row]].astype(np.float32) / 32768.0

    def crop(self, row, start, length):
        x = self.audio[row, start:start + length].astype(np.float32) / 32768.0
        if len(x) < length:                                   # only for very short tracks
            x = np.pad(x, (0, length - len(x)))
        return x


class TripleViewDataset(Dataset):
    """Three crops per track: identity pair (cA, cA2) and style partner cB.

    pairs_per_track > 1 makes one "epoch" visit every track several times with fresh crops.
    """

    def __init__(self, store, rows, crop_seconds=5.0, max_shift_seconds=0.5,
                 style_extra=1.1, pipeline=None, gpu_augment=True, pairs_per_track=1):
        self.store, self.rows = store, np.asarray(rows)
        self.crop = int(crop_seconds * SR)
        self.long = int(crop_seconds * style_extra * SR)
        self.shift = int(max_shift_seconds * SR)
        self.pipeline, self.gpu_augment = pipeline, gpu_augment
        self.ppt = pairs_per_track

    def __len__(self):
        return len(self.rows) * self.ppt

    def starts(self, n, rng):
        """Start samples for cA, cA2, cB. cB does not overlap cA when the track allows it."""
        sA = int(rng.integers(0, max(1, n - self.crop)))
        sA2 = int(np.clip(sA + rng.integers(-self.shift, self.shift + 1), 0, max(0, n - self.crop)))
        hiB = max(1, n - self.long)
        sB = int(rng.integers(0, hiB))
        for _ in range(20):                       # rejection-sample a non-overlapping segment
            if sB + self.long <= sA or sB >= sA + self.crop:
                break
            sB = int(rng.integers(0, hiB))
        return sA, sA2, sB

    def __getitem__(self, i):
        row = self.rows[i % len(self.rows)]
        rng = np.random.default_rng()             # fresh randomness in every worker / epoch
        sA, sA2, sB = self.starts(int(self.store.n_samples[row]), rng)
        cA = torch.from_numpy(self.store.crop(row, sA, self.crop))
        cA2 = torch.from_numpy(self.store.crop(row, sA2, self.crop))
        cB = torch.from_numpy(self.store.crop(row, sB, self.long))
        if self.gpu_augment or self.pipeline is None:
            return cA, cA2, cB, int(row)
        p = self.pipeline
        with torch.no_grad():
            v1, v2 = p.view(cA[None])[0], p.view(cA2[None])[0]
            v3 = p.style_view(cB[None])[0]
        return v1, v2, v3, int(row)
