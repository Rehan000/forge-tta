"""Phase 0 result: source vs forward-only BN-stat adaptation across CIFAR-10-C.

Reproduces the two reference points the project is built on. For each of the 15
corruptions at a chosen severity, reports:
  - source accuracy (no adaptation)
  - bn_adapt accuracy (forward-only BN-stat recalibration)
  - the recovery (bn_adapt - source)

and prints mean error (mCE-style, lower is better) for both. The Phase 1 novel
quantization-scale lever will be added here as a third column.

Usage:
    python src/run_phase0.py --severity 5
    python src/run_phase0.py --severity 5 --corruptions gaussian_noise fog
"""
import argparse
import json
import os
import torch

from models import resnet20
from data import CORRUPTIONS, corruption_loader
from tta import evaluate, bn_adapt_eval
from train import pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/resnet20_cifar10.pt")
    ap.add_argument("--data", default="data")
    ap.add_argument("--severity", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--momentum", type=float, default=0.1)
    ap.add_argument("--corruptions", nargs="*", default=CORRUPTIONS)
    ap.add_argument("--out", default="results/phase0.json")
    args = ap.parse_args()

    device = pick_device()
    print(f"device: {device}")

    model = resnet20()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    print(f"loaded {args.ckpt}")

    rows, src_errs, bn_errs = [], [], []
    print(f"\n{'corruption':<20}{'source':>9}{'bn_adapt':>10}{'recovery':>10}")
    print("-" * 49)
    for c in args.corruptions:
        loader = corruption_loader(args.data, c, args.severity, args.batch_size)
        src = evaluate(model, loader, device)
        # fresh loader so bn_adapt sees the same stream from the start
        loader = corruption_loader(args.data, c, args.severity, args.batch_size)
        bn = bn_adapt_eval(model, loader, device, args.momentum)
        rows.append({"corruption": c, "source": src, "bn_adapt": bn, "recovery": bn - src})
        src_errs.append(100 - src)
        bn_errs.append(100 - bn)
        print(f"{c:<20}{src:>8.2f}%{bn:>9.2f}%{bn - src:>+9.2f}%")

    mean_src_err = sum(src_errs) / len(src_errs)
    mean_bn_err = sum(bn_errs) / len(bn_errs)
    print("-" * 49)
    print(f"{'MEAN ERROR':<20}{mean_src_err:>8.2f}%{mean_bn_err:>9.2f}%{mean_src_err - mean_bn_err:>+9.2f}%")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "severity": args.severity,
            "mean_error_source": mean_src_err,
            "mean_error_bn_adapt": mean_bn_err,
            "rows": rows,
        }, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
