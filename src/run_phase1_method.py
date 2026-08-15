"""Phase 1, step 3: forward-only quantization-scale adaptation on the folded int8 model.

Tests whether re-estimating activation quant scales online (forward-only, no BN)
recovers accuracy the folded model loses under corruption. Compares against:
  - folded int8 source           (+0.00 baseline — nothing adapts)
  - int8 BN-preserved BN-adapt    (+20.09 ceiling — what stats-recalibration gets
                                    when BN still exists)

Usage:
    python src/run_phase1_method.py --severity 5
    python src/run_phase1_method.py --severity 5 --momentum 0.05 --percentile 0.999
"""
import argparse
import json
import os
import torch

from models import resnet20
from data import CORRUPTIONS, corruption_loader, clean_loaders
from tta import evaluate
from tta_quant import scale_adapt_eval
from quant import quantize_model, calibrate_model
from fold import fold_bn
from train import pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/resnet20_cifar10.pt")
    ap.add_argument("--data", default="data")
    ap.add_argument("--severity", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--momentum", type=float, default=0.1)
    ap.add_argument("--percentile", type=float, default=None)
    ap.add_argument("--w-bits", type=int, default=8)
    ap.add_argument("--a-bits", type=int, default=8)
    ap.add_argument("--calib-batches", type=int, default=8)
    ap.add_argument("--corruptions", nargs="*", default=CORRUPTIONS)
    ap.add_argument("--fold-results", default="results/phase1_fold.json")
    ap.add_argument("--preserved-results", default="results/phase1_quant.json")
    ap.add_argument("--out", default="results/phase1_method.json")
    args = ap.parse_args()

    device = pick_device()
    print(f"device: {device}  |  scale-adapt: momentum={args.momentum} percentile={args.percentile}")

    fp32 = resnet20()
    fp32.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    folded = fold_bn(fp32)
    _, clean_test = clean_loaders(args.data, args.batch_size)
    qmodel = quantize_model(folded, args.w_bits, args.a_bits)
    calibrate_model(qmodel, clean_test, device, args.calib_batches)

    rows, src_errs, ad_errs = [], [], []
    print(f"\n{'corruption':<20}{'folded src':>11}{'scale-adapt':>13}{'recovery':>10}")
    print("-" * 54)
    for c in args.corruptions:
        loader = corruption_loader(args.data, c, args.severity, args.batch_size)
        src = evaluate(qmodel, loader, device)
        loader = corruption_loader(args.data, c, args.severity, args.batch_size)
        ad = scale_adapt_eval(qmodel, loader, device, args.momentum, args.percentile)
        rows.append({"corruption": c, "folded_source": src, "scale_adapt": ad, "recovery": ad - src})
        src_errs.append(100 - src)
        ad_errs.append(100 - ad)
        print(f"{c:<20}{src:>10.2f}%{ad:>12.2f}%{ad - src:>+9.2f}%")

    mean_src_err = sum(src_errs) / len(src_errs)
    mean_ad_err = sum(ad_errs) / len(ad_errs)
    print("-" * 54)
    print(f"{'MEAN ERROR':<20}{mean_src_err:>10.2f}%{mean_ad_err:>12.2f}%{mean_src_err - mean_ad_err:>+9.2f}%")

    print("\n=== recovery on the folded (deployable) int8 model ===")
    print(f"{'method':<34}{'mean err':>10}{'recovery':>10}")
    print(f"{'folded source (no adapt)':<34}{mean_src_err:>9.2f}%{0.0:>+9.2f}%")
    print(f"{'scale-adapt (ours, forward-only)':<34}{mean_ad_err:>9.2f}%{mean_src_err - mean_ad_err:>+9.2f}%")
    if os.path.exists(args.preserved_results):
        with open(args.preserved_results) as f:
            pr = json.load(f)
        ceil = pr["mean_error_int8_source"] - pr["mean_error_int8_bn_adapt"]
        print(f"{'(ceiling: BN-adapt, BN-preserved)':<34}{pr['mean_error_int8_bn_adapt']:>9.2f}%{ceil:>+9.2f}%")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "w_bits": args.w_bits, "a_bits": args.a_bits, "severity": args.severity,
            "momentum": args.momentum, "percentile": args.percentile,
            "mean_error_folded_source": mean_src_err, "mean_error_scale_adapt": mean_ad_err,
            "rows": rows,
        }, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
