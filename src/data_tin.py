"""Tiny-ImageNet (clean) and Tiny-ImageNet-C (corruption) loaders.

Tiny-ImageNet: 200 classes, 64x64, 500 train / 50 val per class. Folder-structured
(ImageFolder), unlike the CIFAR-C .npy benchmarks. Tiny-ImageNet-C is organized as
Tiny-ImageNet-C/<corruption>/<severity>/<wnid>/*.JPEG. ImageFolder sorts class (wnid)
folders alphabetically everywhere, so train / val / corruption share one class->index map.

Download once:
  http://cs231n.stanford.edu/tiny-imagenet-200.zip          -> data/tiny-imagenet-200/
  https://zenodo.org/records/2536630/files/Tiny-ImageNet-C.tar -> data/Tiny-ImageNet-C/
"""
import os
import shutil
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

# Compat shim: newer scikit-image renamed gaussian(multichannel=) -> gaussian(channel_axis=).
# imagecorruptions still passes multichannel; patch the name it calls. Applied at module
# load so every DataLoader worker (which re-imports this module) gets it.
try:
    import numpy as _np
    if not hasattr(_np, "float_"):          # NumPy 2.0 removed np.float_; imagecorruptions uses it
        _np.float_ = _np.float64
    import imagecorruptions.corruptions as _icc
    _orig_gaussian = _icc.gaussian

    def _gaussian_compat(image, *a, multichannel=None, **k):
        if multichannel is not None and "channel_axis" not in k:
            k["channel_axis"] = -1 if multichannel else None
        return _orig_gaussian(image, *a, **k)

    _icc.gaussian = _gaussian_compat
except Exception:
    pass

# ImageNet channel statistics (Tiny-ImageNet is an ImageNet subset).
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
NUM_CLASSES = 200

CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]


def _norm():
    return T.Normalize(MEAN, STD)


def reorganize_val(root):
    """Move val/images/*.JPEG into val/<wnid>/ using val_annotations.txt (idempotent)."""
    vdir = os.path.join(root, "tiny-imagenet-200", "val")
    imgdir = os.path.join(vdir, "images")
    if not os.path.isdir(imgdir):
        return                                   # already reorganized
    mapping = {}
    with open(os.path.join(vdir, "val_annotations.txt")) as f:
        for line in f:
            p = line.split("\t")
            mapping[p[0]] = p[1]
    for fname, wnid in mapping.items():
        dst = os.path.join(vdir, wnid)
        os.makedirs(dst, exist_ok=True)
        src = os.path.join(imgdir, fname)
        if os.path.exists(src):
            shutil.move(src, os.path.join(dst, fname))
    try:
        os.rmdir(imgdir)
    except OSError:
        pass


def clean_loaders(root="data", batch_size=128, num_workers=8):
    base = os.path.join(root, "tiny-imagenet-200")
    reorganize_val(root)
    train_tf = T.Compose([T.RandomCrop(64, padding=8), T.RandomHorizontalFlip(),
                          T.ToTensor(), _norm()])
    test_tf = T.Compose([T.ToTensor(), _norm()])
    train = torchvision.datasets.ImageFolder(os.path.join(base, "train"), train_tf)
    test = torchvision.datasets.ImageFolder(os.path.join(base, "val"), test_tf)
    return (
        DataLoader(train, batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(test, batch_size, shuffle=False, num_workers=num_workers),
    )


class GeneratedTINC(torch.utils.data.Dataset):
    """Tiny-ImageNet-C generated locally from the clean val set with the official
    `imagecorruptions` library (the same Hendrycks & Dietterich implementation the
    released Tiny-ImageNet-C was built with) -- avoids the 1.8GB download."""

    def __init__(self, root, corruption, severity):
        # store only picklable state (so multiprocessing workers can spawn): the corrupt
        # function and numpy are imported lazily in __getitem__, not stored as attributes.
        self.corruption, self.severity = corruption, severity
        base = os.path.join(root, "tiny-imagenet-200")
        reorganize_val(root)
        self.val = torchvision.datasets.ImageFolder(os.path.join(base, "val"))  # (PIL_RGB, label)
        self.tf = T.Compose([T.ToTensor(), _norm()])

    def __len__(self):
        return len(self.val)

    def __getitem__(self, i):
        import numpy as np
        from imagecorruptions import corrupt
        img, lbl = self.val[i]                    # pil_loader converts to RGB
        arr = np.asarray(img.convert("RGB"))
        c = corrupt(arr, corruption_name=self.corruption, severity=self.severity)
        return self.tf(c.astype("uint8")), lbl


def corruption_loader(root, corruption, severity=5, batch_size=128, num_workers=4,
                      max_samples=None):
    """One corruption/severity, generated locally. Shuffled (seeded) so the test-time
    stream is class-mixed, not sorted by class (which would be pathological for TTA)."""
    ds = GeneratedTINC(root, corruption, severity)
    g = torch.Generator().manual_seed(0)
    order = torch.randperm(len(ds), generator=g).tolist()
    if max_samples is not None and max_samples < len(ds):
        order = order[:max_samples]
    ds = Subset(ds, order)                        # fixed shuffled (mixed-class) stream
    return DataLoader(ds, batch_size, shuffle=False, num_workers=num_workers)
