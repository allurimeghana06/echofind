"""Phase 1: the SSL Dataset.

Audio comes from the single memory-mapped array made by notebook 01
(audio_int16.npy, shape (8000, 661500)) plus meta.csv. Reading a random 5 s crop
from it takes microseconds, so the data loader never starves the GPU.

Each item = two crops of the SAME track taken at DIFFERENT random moments.
  * gpu_augment=True  (training): returns the two raw crops; the training loop moves
    the whole batch to the GPU and augments it there (fast).
  * gpu_augment=False (what the spec describes): returns two fully augmented
    log-mel views (x_i, x_j) produced on the CPU by AugmentationPipeline.
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

    def rows(self, splits=("training", "validation", "test"), only_ok=True):
        m = self.meta
        keep = m.split.isin(splits)
        if only_ok:
            keep &= m.ok
        return m.row[keep].values

    def track(self, row):
        """Full clean track as float32 numpy (trailing zero padding removed)."""
        return self.audio[row, :self.n_samples[row]].astype(np.float32) / 32768.0

    def crop(self, row, start, length):
        return self.audio[row, start:start + length].astype(np.float32) / 32768.0


class SSLDataset(Dataset):
    """Returns a positive pair for one track.

    pairs_per_track > 1 makes one "epoch" visit every track several times with fresh
    crops, which is useful because 8000 tracks / batch 128 = only 62 steps per epoch.
    """

    def __init__(self, store, rows, crop_seconds=5.0, pipeline=None, gpu_augment=True,
                 pairs_per_track=1, seed=0):
        self.store, self.rows = store, np.asarray(rows)
        self.crop = int(crop_seconds * SR)
        self.pipeline, self.gpu_augment = pipeline, gpu_augment
        self.ppt = pairs_per_track
        self.seed = seed

    def __len__(self):
        return len(self.rows) * self.ppt

    def __getitem__(self, i):
        row = self.rows[i % len(self.rows)]
        rng = np.random.default_rng()          # fresh randomness in every worker / epoch
        hi = max(1, self.store.n_samples[row] - self.crop)
        s1, s2 = rng.integers(0, hi, size=2)
        c1 = torch.from_numpy(self.store.crop(row, s1, self.crop))
        c2 = torch.from_numpy(self.store.crop(row, s2, self.crop))
        if self.gpu_augment or self.pipeline is None:
            return c1, c2, int(row)
        v1, v2 = self.pipeline.views_from_crops(c1.unsqueeze(0), c2.unsqueeze(0))
        return v1[0], v2[0], int(row)
