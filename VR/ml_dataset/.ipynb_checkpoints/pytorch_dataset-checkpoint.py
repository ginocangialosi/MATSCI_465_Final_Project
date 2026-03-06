# ── pytorch_dataset.py ────────────────────────────────────────────────────────
# Drop this file into your ML project.  No kikuchipy dependency needed at
# training time — only h5py, numpy, and torch.
# ─────────────────────────────────────────────────────────────────────────────

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class EBSDStrainDataset(Dataset):
    """
    PyTorch Dataset for the FCC Fe EBSD ML training set.

    Each item is (pattern_tensor, strain_pct_tensor).
    The pattern tensor has shape (1, H, W) — single-channel "image".

    Parameters
    ----------
    h5_path  : str — path to ebsd_fcc_fe.h5
    split    : "train" | "val" | "test"
    transform: optional torchvision transform (e.g. random crop, flip)

    Example
    -------
    from torch.utils.data import DataLoader
    ds     = EBSDStrainDataset("ml_dataset/ebsd_fcc_fe.h5", split="train")
    loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=4)
    for patterns, strains in loader:
        # patterns: (64, 1, 80, 80) float32
        # strains:  (64,)           float32  in %
        ...
    """

    def __init__(self, h5_path, split="train", transform=None):
        self.h5_path   = h5_path
        self.split     = split
        self.transform = transform
        self._h5       = None          # opened lazily (required for num_workers>0)

        with h5py.File(h5_path, "r") as f:
            self.indices = f[f"idx_{split}"][:].tolist()

    def _open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        self._open()
        idx     = self.indices[item]
        pat     = self._h5["patterns"][idx]          # (H, W) float32
        strain  = self._h5["strain_pct"][idx]        # scalar float32
        euler   = self._h5["euler_angles_deg"][idx]  # (3,) float32

        pat_tensor    = torch.tensor(pat[None])       # (1, H, W)
        strain_tensor = torch.tensor(strain)
        euler_tensor  = torch.tensor(euler)

        if self.transform:
            pat_tensor = self.transform(pat_tensor)

        return pat_tensor, strain_tensor, euler_tensor

    def __del__(self):
        if self._h5 is not None:
            self._h5.close()