import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class EBSDStrainDataset(Dataset):
    """
    PyTorch Dataset for the FCC Fe EBSD strain classification dataset.

    Parameters
    ----------
    h5_path   : str   — path to ebsd_fcc_fe.h5
    split     : str   — "train" | "val" | "test"
    transform : optional torchvision transform

    Returns (per item)
    ------------------
    pat_t     : float32 tensor  (1, H, W)
    strain_t  : float32 scalar  strain in %
    euler_t   : float32 tensor  (3,) Bunge Euler angles in degrees
    """

    def __init__(self, h5_path, split="train", transform=None):
        self.h5_path   = h5_path
        self.split     = split
        self.transform = transform
        self._h5       = None

        with h5py.File(h5_path, "r") as f:
            self.indices = f[f"idx_{split}"][:].tolist()

    def _open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        self._open()
        idx          = self.indices[item]
        pat          = self._h5["patterns"][idx]
        strain       = self._h5["strain_pct"][idx]
        euler        = self._h5["euler_angles_deg"][idx]
        pat_t        = torch.tensor(pat[None], dtype=torch.float32)
        if self.transform:
            pat_t = self.transform(pat_t)
        return pat_t, torch.tensor(strain), torch.tensor(euler)

    def __del__(self):
        if self._h5 is not None:
            self._h5.close()
