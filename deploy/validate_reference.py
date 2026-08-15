"""Validate the true-integer engine against the PyTorch (fake-quant) path.

Two checks, on a subset for speed:
  1. clean, no adapt   — integer engine vs torch quantized model: accuracy + logit agreement
  2. corruption, bs=1, adapt — integer recalib vs torch channel_recalib: accuracy

If the integer engine reproduces the torch accuracy, the method is confirmed to
survive true integer-only execution (not just float fake-quant) — the on-device claim.

Usage:
    python deploy/validate_reference.py --n 500 --corruption gaussian_noise
"""
import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from models import resnet20                       # noqa: E402
from fold import fold_bn_recalib                  # noqa: E402
from quant import quantize_model, calibrate_model  # noqa: E402
from data import clean_loaders, corruption_loader  # noqa: E402
from tta import evaluate                          # noqa: E402
from tta_quant import channel_recalib_eval        # noqa: E402
from train import pick_device                     # noqa: E402
from int8_reference import IntResNet20            # noqa: E402

ART = os.path.join(os.path.dirname(__file__), "artifacts", "resnet20_int8.npz")


def _build_torch(momentum, device):
    fp32 = resnet20()
    fp32.load_state_dict(torch.load("checkpoints/resnet20_cifar10.pt", map_location="cpu"))
    q = quantize_model(fold_bn_recalib(fp32, momentum), 8, 8)
    _, clean_test = clean_loaders("data", 128)
    calibrate_model(q, clean_test, device, 8)
    return q.to("cpu").eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--corruption", default="gaussian_noise")
    ap.add_argument("--severity", type=int, default=5)
    args = ap.parse_args()

    eng = IntResNet20(ART)
    momentum = eng.momentum
    torch_q = _build_torch(momentum, "cpu")
    print(f"momentum={momentum}  |  validating on {args.n} samples\n")

    # ---- 1. clean, no adapt ----
    _, clean_test = clean_loaders("data", 256)
    xs, ys = next(iter(clean_test))
    xs, ys = xs[:args.n], ys[:args.n]
    with torch.no_grad():
        tlogits = torch_q(xs).numpy()
    ilogits = np.concatenate([eng.forward(xs[i:i + 1].numpy(), adapt=False) for i in range(len(xs))])
    t_acc = (tlogits.argmax(1) == ys.numpy()).mean() * 100
    i_acc = (ilogits.argmax(1) == ys.numpy()).mean() * 100
    agree = (tlogits.argmax(1) == ilogits.argmax(1)).mean() * 100
    maxdiff = np.abs(tlogits - ilogits).max()
    print("[clean, no adapt]")
    print(f"  torch acc {t_acc:.2f}%  |  int-engine acc {i_acc:.2f}%  |  argmax agreement {agree:.2f}%")
    print(f"  max |logit diff| {maxdiff:.4f}  (small = integer path faithful)\n")

    # ---- 2. corruption, bs=1, adapt ----
    loader1 = corruption_loader("data", args.corruption, args.severity, 1, max_samples=args.n)
    t_adapt = channel_recalib_eval(torch_q, loader1, "cpu", momentum)
    eng.reset()
    correct = 0
    cl = corruption_loader("data", args.corruption, args.severity, 1, max_samples=args.n)
    for x, y in cl:
        pred = eng.forward(x.numpy(), adapt=True).argmax(1)[0]
        correct += int(pred == y.item())
    i_adapt = 100.0 * correct / args.n
    print(f"[{args.corruption} sev{args.severity}, bs=1, adapt]")
    print(f"  torch channel-recalib {t_adapt:.2f}%  |  int-engine recalib {i_adapt:.2f}%  "
          f"(diff {abs(t_adapt - i_adapt):.2f}pt)")
    print("\ninteger engine reproduces the torch path -> method survives true integer execution"
          if maxdiff < 1.0 and abs(t_adapt - i_adapt) < 2.0 else
          "\nWARNING: integer/torch mismatch larger than expected — investigate requant/scales")


if __name__ == "__main__":
    main()
