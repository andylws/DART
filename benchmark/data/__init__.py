from . import cifar, imagenet

CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
]

BENCHMARKS = {"cifar100": cifar, "imagenet": imagenet}


def load_id(dataset, n_examples, corruption, severity, data_dir):
    """ID test samples. ``corruption='original'`` means the clean set."""
    return BENCHMARKS[dataset].load_id(n_examples, corruption, severity, data_dir)


def load_ood(dataset, name, n_examples, corruption, severity, data_dir):
    """OOD test samples (label -1) under the same corruption as the ID data."""
    module = BENCHMARKS[dataset]
    if name not in module.OOD_DATASETS:
        raise ValueError(f"Unknown OOD dataset {name} for {dataset}; choose from {module.OOD_DATASETS}")
    return module.load_ood(name, n_examples, corruption, severity, data_dir)
