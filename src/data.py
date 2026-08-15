"""CIFAR-10/100 (clean) and CIFAR-10-C/100-C (corruption) data loading.

Clean sets download automatically via torchvision. The corruption benchmarks are
Hendrycks & Dietterich (2019): 15 corruption types, each a (50000,32,32,3) uint8
array stacked as 5 severities x 10000 test images, plus a shared labels.npy.
Download once from Zenodo:

    CIFAR-10-C : https://zenodo.org/records/2535967/files/CIFAR-10-C.tar
    CIFAR-100-C: https://zenodo.org/records/3555552/files/CIFAR-100-C.tar
    tar xf <file> -C data/    # -> data/CIFAR-10-C/<corruption>.npy etc.

`dataset` selects "cifar10" (default) or "cifar100" throughout.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as T

# per-dataset channel stats, corruption dir, torchvision class, #classes
DATASETS = {
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465), "std": (0.2470, 0.2435, 0.2616),
        "cdir": "CIFAR-10-C", "cls": torchvision.datasets.CIFAR10, "num_classes": 10,
    },
    "cifar100": {
        "mean": (0.5071, 0.4865, 0.4409), "std": (0.2673, 0.2564, 0.2762),
        "cdir": "CIFAR-100-C", "cls": torchvision.datasets.CIFAR100, "num_classes": 100,
    },
}

CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]


def num_classes(dataset="cifar10"):
    return DATASETS[dataset]["num_classes"]


def _normalize(dataset):
    d = DATASETS[dataset]
    return T.Normalize(d["mean"], d["std"])


def clean_loaders(root="data", batch_size=128, num_workers=2, dataset="cifar10"):
    """Train (augmented) and test (clean) loaders for the chosen dataset."""
    norm = _normalize(dataset)
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), norm,
    ])
    test_tf = T.Compose([T.ToTensor(), norm])
    cls = DATASETS[dataset]["cls"]
    train = cls(root, train=True, download=True, transform=train_tf)
    test = cls(root, train=False, download=True, transform=test_tf)
    return (
        DataLoader(train, batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(test, batch_size, shuffle=False, num_workers=num_workers),
    )


class CorruptionSet(Dataset):
    """One corruption type at one severity (1-5) from a *-C benchmark's .npy files."""

    def __init__(self, root, corruption, severity=5, dataset="cifar10"):
        cdir = os.path.join(root, DATASETS[dataset]["cdir"])
        path = os.path.join(cdir, f"{corruption}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Download {DATASETS[dataset]['cdir']} into {cdir}/.")
        assert 1 <= severity <= 5
        lo, hi = (severity - 1) * 10000, severity * 10000
        self.images = np.load(path)[lo:hi]                       # uint8 HWC
        self.labels = np.load(os.path.join(cdir, "labels.npy"))[lo:hi].astype(np.int64)
        self.tf = T.Compose([T.ToTensor(), _normalize(dataset)])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.tf(self.images[i]), int(self.labels[i])


# back-compat alias
CIFAR10C = CorruptionSet


def corruption_loader(root, corruption, severity=5, batch_size=128, num_workers=2,
                      max_samples=None, dataset="cifar10", seed=None):
    """Stream of one corruption. seed=None keeps the deterministic in-order stream (the
    original behaviour, so prior results reproduce exactly). Passing an int seed shuffles
    the stream with a seeded generator, which is how we get multi-seed error bars: online
    test-time adaptation is order-dependent, so a different arrival order is a valid re-run."""
    ds = CorruptionSet(root, corruption, severity, dataset)
    if seed is None:
        if max_samples is not None and max_samples < len(ds):
            ds = Subset(ds, range(max_samples))      # deterministic first-N (stream prefix)
        return DataLoader(ds, batch_size, shuffle=False, num_workers=num_workers)
    g = torch.Generator().manual_seed(seed)
    if max_samples is not None and max_samples < len(ds):
        idx = torch.randperm(len(ds), generator=g)[:max_samples].tolist()
        return DataLoader(Subset(ds, idx), batch_size, shuffle=False, num_workers=num_workers)
    return DataLoader(ds, batch_size, shuffle=True, num_workers=num_workers, generator=g)
