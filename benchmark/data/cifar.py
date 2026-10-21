"""CIFAR-100 benchmark: CIFAR-100(-C) as ID, SVHN/Places365/LSUN/iSUN/Textures(-C) as OOD.

Expected layout under ``data_dir``::

    CIFAR-100-C/{original,<corruption>}.npy, labels.npy
    SVHN-C/{original,<corruption>}.npy, labels.npy
    places365/images/                     LSUN_resize/images/   ...
    places365/images-C_32/<corruption>/<severity>/   (made by tools/make_ood_c.py)
"""

from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from robustbench.data import load_cifar100c

from .folder import load_flat_folder

SVHN_SIZE = 26032
# OOD name -> folder under data_dir
FOLDER_OOD = {
    "Places365": "places365",
    "LSUN": "LSUN_resize",
    "iSUN": "iSUN",
    "textures": "textures",
}
OOD_DATASETS = ["svhn"] + list(FOLDER_OOD)


def load_id(n_examples, corruption, severity, data_dir):
    """CIFAR-100 test set; ``corruption='original'`` reads the clean images."""
    if corruption == "original":
        severity = 1
    return load_cifar100c(n_examples, severity, data_dir, True, [corruption])


def _load_svhn(n_examples, corruption, severity, data_dir):
    root = Path(data_dir) / "SVHN-C"
    images = np.load(root / f"{corruption}.npy")
    if corruption == "original":
        images = np.transpose(images, (3, 2, 0, 1))[:n_examples]
    else:
        images = images[(severity - 1) * SVHN_SIZE:severity * SVHN_SIZE][:n_examples]
        images = np.transpose(images, (0, 3, 1, 2))
    x = torch.tensor(images.astype(np.float32) / 255)
    x = x[torch.from_numpy(np.random.permutation(len(x)))]
    return x, torch.full((x.size(0),), -1, dtype=torch.long)


def load_ood(name, n_examples, corruption, severity, data_dir):
    if name == "svhn":
        return _load_svhn(n_examples, corruption, severity, data_dir)

    root = Path(data_dir) / FOLDER_OOD[name]
    if corruption == "original":
        return load_flat_folder(root / "images", n_examples,
                                T.Compose([T.Resize((32, 32)), T.ToTensor()]))
    transform = T.ToTensor() if name != "textures" else T.Compose([T.Resize((32, 32)), T.ToTensor()])
    return load_flat_folder(root / "images-C_32" / corruption / str(severity), n_examples,
                            transform, permute_after_repeat=True)
