"""Phase 1, step 2: the gap — fold BN, and the forward-only recovery vanishes.

Folds BN into the convs (true integer-only deployment form), quantizes to int8,
and re-runs the corruption table. Because the folded model has NO BatchNorm layers,
the forward-only BN-stat baseline has nothing to recalibrate, so its +20pt recovery
disappears. This is the empirical motivation for the quantization-scale lever:
in the deployed model, normalization-statistic adaptation is simply unavailable.

Reports, per corruption: folded-int8 source acc, the (no-op) BN-adapt acc, and the
recovery (~0). Compares against the BN-preserved int8 recovery from phase1_quant.json.

Usage:
    python src/run_phase1_fold.py --severity 5
"""
import argparse
import json
import os
import torch

from models import resnet20
from data import CORRUPTIONS, corruption_loader, clean_loaders
from tta import evaluate, bn_adapt_eval
from quant import quantize_model, calibrate_model
from fold import fold_bn, count_bn
from train import pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/resnet20_cifar10.pt")
    ap.add_argument("--data", default="data")
    ap.add_argument("--severity", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--momentum", type=float, default=0.1)
    ap.add_argument("--w-bits", type=int, default=8)
    ap.add_argument("--a-bits", type=int, default=8)
    ap.add_argument("--calib-batches", type=int, default=8)
    ap.add_argument("--corruptions", nargs="*", default=CORRUPTIONS)
    ap.add_argument("--preserved-results", default="results/phase1_quant.json")
    ap.add_argument("--out", default="results/phase1_fold.json")
    args = ap.parse_args()

    device = pick_device()
    print(f"device: {device}  |  folded BN + int8 W{args.w_bits}A{args.a_bits}")

    fp32 = resnet20()
    fp32.load_state_dict(torch.load(args.ckpt, map_location="cpu"))

    folded = fold_bn(fp32)
    print(f"BN layers after folding: {count_bn(folded)} (was {count_bn(fp32)})")

    _, clean_test = clean_loaders(args.data, args.batch_size)
    qmodel = quantize_model(folded, args.w_bits, args.a_bits)
    calibrate_model(qmodel, clean_test, device, args.calib_batches)
    print(f"folded int8 clean acc: {evaluate(qmodel, clean_test, device):.2f}%")

    rows, src_errs, bn_errs = [], [], []
    print(f"\n{'corruption':<20}{'folded src':>11}{'bn_adapt':>10}{'recovery':>10}")
    print("-" * 51)
    for c in args.corruptions:
        loader = corruption_loader(args.data, c, args.severity, args.batch_size)
        src = evaluate(qmodel, loader, device)
        loader = corruption_loader(args.data, c, args.severity, args.batch_size)
        bn = bn_adapt_eval(qmodel, loader, device, args.momentum)   # no BN -> no-op
        rows.append({"corruption": c, "folded_source": src, "folded_bn_adapt": bn, "recovery": bn - src})
        src_errs.append(100 - src)
        bn_errs.append(100 - bn)
        print(f"{c:<20}{src:>10.2f}%{bn:>9.2f}%{bn - src:>+9.2f}%")

    mean_src_err = sum(src_errs) / len(src_errs)
    mean_bn_err = sum(bn_errs) / len(bn_errs)
    print("-" * 51)
    print(f"{'MEAN ERROR':<20}{mean_src_err:>10.2f}%{mean_bn_err:>9.2f}%{mean_src_err - mean_bn_err:>+9.2f}%")

    print("\n=== the gap: forward-only recovery, BN-preserved vs folded (int8) ===")
    print(f"{'':<22}{'source':>10}{'bn_adapt':>10}{'recovery':>10}")
    if os.path.exists(args.preserved_results):
        with open(args.preserved_results) as f:
            pr = json.load(f)
        print(f"{'int8, BN preserved':<22}{pr['mean_error_int8_source']:>9.2f}%"
              f"{pr['mean_error_int8_bn_adapt']:>9.2f}%"
              f"{pr['mean_error_int8_source'] - pr['mean_error_int8_bn_adapt']:>+9.2f}%")
    print(f"{'int8, BN FOLDED (deploy)':<22}{mean_src_err:>9.2f}%{mean_bn_err:>9.2f}%"
          f"{mean_src_err - mean_bn_err:>+9.2f}%")
    print("\n-> In the deployed (folded) model the +20pt forward-only recovery is gone:")
    print("   there are no BN statistics left to recalibrate. This is the gap the")
    print("   quantization-scale/clip adaptation lever (Phase 1 step 3) must fill.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "w_bits": args.w_bits, "a_bits": args.a_bits, "severity": args.severity,
            "bn_layers_after_fold": count_bn(folded),
            "mean_error_folded_source": mean_src_err, "mean_error_folded_bn_adapt": mean_bn_err,
            "rows": rows,
        }, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
