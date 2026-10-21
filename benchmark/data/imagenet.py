"""ImageNet benchmark: ImageNet(-C) as ID, ImageNet-O/Places/SUN/iNaturalist/Textures(-C) as OOD.

Expected layout under ``data_dir``::

    ILSVRC12/val/<wnid>/*.JPEG
    ImageNet-C/<corruption>/<severity>/<wnid>/*.JPEG
    imagenet-o/<wnid>/*.jpg
    imagenet-o/images-C_224/<corruption>/<severity>/<wnid>/*.jpg   (tools/make_ood_c.py)
    Places/images/*.jpg
    Places/images-C_64/<corruption>/<severity>/*.jpg               (tools/make_ood_c.py)
    (same for SUN, iNaturalist, textures)
"""

from pathlib import Path

import torchvision.transforms as T
from torchvision.datasets import ImageFolder

from .folder import load_flat_folder, load_image_folder

FOLDER_OOD = ["Places", "SUN", "iNaturalist", "textures"]
OOD_DATASETS = ["imagenet_o"] + FOLDER_OOD

RESIZE_CROP = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])


def load_id(n_examples, corruption, severity, data_dir):
    if corruption == "original":
        transform = T.Compose([T.Resize(256, antialias=True), T.CenterCrop(224), T.ToTensor()])
        dataset = ImageFolder(Path(data_dir) / "ILSVRC12" / "val", transform)
    else:
        dataset = ImageFolder(Path(data_dir) / "ImageNet-C" / corruption / str(severity), T.ToTensor())
    return load_image_folder(dataset, n_examples)


def load_ood(name, n_examples, corruption, severity, data_dir):
    data_dir = Path(data_dir)
    if name == "imagenet_o":
        if corruption == "original":
            dataset = ImageFolder(data_dir / "imagenet-o", RESIZE_CROP)
        else:
            dataset = ImageFolder(data_dir / "imagenet-o" / "images-C_224" / corruption / str(severity),
                                  T.ToTensor())
        x, _ = load_image_folder(dataset, n_examples)
        return x, -1 * x.new_ones(x.size(0)).long()

    if corruption == "original":
        return load_flat_folder(data_dir / name / "images", n_examples, RESIZE_CROP)
    return load_flat_folder(data_dir / name / "images-C_64" / corruption / str(severity), n_examples,
                            RESIZE_CROP, permute_after_repeat=True)
