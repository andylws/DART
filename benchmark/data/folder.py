"""Generic loaders shared by the CIFAR and ImageNet benchmarks."""

import logging
import random
from pathlib import Path

import torch
import torch.utils.data as data
from PIL import Image

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")


def _list_images(directory):
    files = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(directory.glob(ext))
        files.extend(directory.glob(ext.upper()))
    return sorted(files)


def load_flat_folder(directory, n_examples, transform, shuffle=True, permute_after_repeat=False):
    """Load ``n_examples`` images from a flat folder as an OOD set.

    Files are shuffled, the first ``n_examples`` are read, and the set is tiled
    if the folder holds fewer images. ``permute_after_repeat`` additionally
    permutes the tiled tensor, as done for the corrupted OOD sets.

    Returns:
        (x, y) with ``y = -1`` for every sample.
    """
    directory = Path(directory)
    files = _list_images(directory)
    if not files:
        raise FileNotFoundError(f"No images found in {directory}")
    if shuffle:
        random.shuffle(files)

    images = []
    for path in files[:n_examples]:
        try:
            images.append(transform(Image.open(path).convert("RGB")))
        except OSError as e:
            logger.warning(f"Skipping {path}: {e}")

    x = torch.stack(images * (n_examples // len(images) + 1))
    if shuffle and permute_after_repeat:
        x = x[torch.randperm(x.size(0))]
    x = x[:n_examples]
    return x, torch.full((x.size(0),), -1, dtype=torch.long)


def load_image_folder(dataset, n_examples, shuffle=True, num_workers=8):
    """Draw ``n_examples`` samples from a torchvision dataset (tiled if smaller)."""
    dataset = data.ConcatDataset([dataset] * (n_examples // len(dataset) + 1))
    indices = torch.randperm(len(dataset))[:n_examples] if shuffle else torch.arange(n_examples)
    loader = data.DataLoader(data.Subset(dataset, indices.tolist()), batch_size=500,
                             shuffle=False, num_workers=num_workers)
    xs, ys = zip(*loader)
    return torch.cat(xs), torch.cat(ys)
