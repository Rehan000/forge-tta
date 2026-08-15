"""Phase 1, step 1: does forward-only adaptation recovery survive int8?

Quantizes the source ResNet-20 to simulated int8 (BN preserved), calibrates on
clean data, then re-runs the source-vs-BN-adapt corruption table on the quantized
model. Prints int8 clean accuracy (the quantization drop) and the int8 recovery,
and compares the int8 BN-adapt mean error to the FP32 baseline in results/phase0.json.

This is the experiment that motivates the whole method: it measures how much of
the +20.55pt FP32 recovery still holds once weights and activations are int8.

Usage:
    python src/run_phase1_quant.py --severity 5
    python src/run_phase1_quant.py --severity 5 --w-bits 8 --a-bits 8
"""
import argparse
import json
import os
import torch

from models import resnet20
from data import CORRUPTIONS, corruption_loader, clean_loaders
from tta import evaluate, bn_adapt_eval
from quant import quantize_model, calibrate_model
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
    ap.add_argument("--percentile", type=float, default=1.0)
    ap.add_argument("--calib-batches", type=int, default=8)
    ap.add_argument("--corruptions", nargs="*", default=CORRUPTIONS)
    ap.add_argument("--fp32-results", default="results/phase0.json")
    ap.add_argument("--out", default="results/phase1_quant.json")
    args = ap.parse_args()

    device = pick_device()
    print(f"device: {device}  |  quantization: W{args.w_bits}A{args.a_bits} (per-channel weights, per-tensor acts)")

    fp32 = resnet20()
    fp32.load_state_dict(torch.load(args.ckpt, map_location="cpu"))

    # quantize + calibrate on clean training data
    _, clean_test = clean_loaders(args.data, args.batch_size)
    qmodel = quantize_model(fp32, args.w_bits, args.a_bits, args.percentile)
    calibrate_model(qmodel, clean_test, device, args.calib_batches)

    # clean accuracy: FP32 vs int8 (the quantization drop)
    fp32_clean = evaluate(fp32, clean_test, device)
    int8_clean = evaluate(qmodel, clean_test, device)
    print(f"\nclean acc: FP32 {fp32_clean:.2f}%  ->  int8 {int8_clean:.2f}%  (drop {fp32_clean - int8_clean:+.2f}%)")

    rows, src_errs, bn_errs = [], [], []
    print(f"\n{'corruption':<20}{'int8 src':>9}{'int8 bn':>9}{'recovery':>10}")
    print("-" * 48)
    for c in args.corruptions:
        loader = corruption_loader(args.data, c, args.severity, args.batch_size)
        src = evaluate(qmodel, loader, device)
        loader = corruption_loader(args.data, c, args.severity, args.batch_size)
        bn = bn_adapt_eval(qmodel, loader, device, args.momentum)
        rows.append({"corruption": c, "int8_source": src, "int8_bn_adapt": bn, "recovery": bn - src})
        src_errs.append(100 - src)
        bn_errs.append(100 - bn)
        print(f"{c:<20}{src:>8.2f}%{bn:>8.2f}%{bn - src:>+9.2f}%")

    mean_src_err = sum(src_errs) / len(src_errs)
    mean_bn_err = sum(bn_errs) / len(bn_errs)
    print("-" * 48)
    print(f"{'MEAN ERROR':<20}{mean_src_err:>8.2f}%{mean_bn_err:>8.2f}%{mean_src_err - mean_bn_err:>+9.2f}%")

    # compare to FP32 baseline if available
    if os.path.exists(args.fp32_results):
        with open(args.fp32_results) as f:
            fp = json.load(f)
        print("\n=== int8 vs FP32 (mean corruption error, lower better) ===")
        print(f"{'':<14}{'source':>10}{'bn_adapt':>10}{'recovery':>10}")
        print(f"{'FP32':<14}{fp['mean_error_source']:>9.2f}%{fp['mean_error_bn_adapt']:>9.2f}%"
              f"{fp['mean_error_source'] - fp['mean_error_bn_adapt']:>+9.2f}%")
        print(f"{'int8':<14}{mean_src_err:>9.2f}%{mean_bn_err:>9.2f}%{mean_src_err - mean_bn_err:>+9.2f}%")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "w_bits": args.w_bits, "a_bits": args.a_bits, "severity": args.severity,
            "fp32_clean": fp32_clean, "int8_clean": int8_clean,
            "mean_error_int8_source": mean_src_err, "mean_error_int8_bn_adapt": mean_bn_err,
            "rows": rows,
        }, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
