"""Phase 1 validation gate: channel-recalib at batch-size-1 (the real MCU regime).

Single-sample streaming is the deployment regime: the device sees one frame at a
time, not a batch. Per-channel statistics from one image are noisy, so the EMA
momentum must be small (effective averaging window ~ batch_size / momentum).
This runs the full 15-corruption sweep at bs=1 with a tuned momentum and compares
the validated bs=1 recovery against the bs=64 result.

Usage:
    python src/run_phase1_bs1.py --severity 5 --momentum 0.01
    python src/run_phase1_bs1.py --momentum 0.01 --max-samples 5000   # faster read
"""
import argparse
import json
import os
import torch

from models import resnet20
from data import CORRUPTIONS, corruption_loader, clean_loaders
from tta import evaluate
from tta_quant import channel_recalib_eval
from quant import quantize_model, calibrate_model
from fold import fold_bn_recalib
from train import pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/resnet20_cifar10.pt")
    ap.add_argument("--data", default="data")
    ap.add_argument("--severity", type=int, default=5)
    ap.add_argument("--momentum", type=float, default=0.01)
    ap.add_argument("--source-batch-size", type=int, default=128)
    ap.add_argument("--w-bits", type=int, default=8)
    ap.add_argument("--a-bits", type=int, default=8)
    ap.add_argument("--calib-batches", type=int, default=8)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--corruptions", nargs="*", default=CORRUPTIONS)
    ap.add_argument("--bs64-results", default="results/phase1_recalib.json")
    ap.add_argument("--out", default="results/phase1_bs1.json")
    args = ap.parse_args()

    device = pick_device()
    print(f"device: {device}  |  channel-recalib @ bs=1, momentum={args.momentum}"
          + (f", first {args.max_samples}/corruption" if args.max_samples else ", full 10k/corruption"))

    fp32 = resnet20()
    fp32.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    qmodel = quantize_model(fold_bn_recalib(fp32, args.momentum), args.w_bits, args.a_bits)
    _, clean_test = clean_loaders(args.data, args.source_batch_size)
    calibrate_model(qmodel, clean_test, device, args.calib_batches)

    rows, src_errs, rec_errs = [], [], []
    print(f"\n{'corruption':<20}{'source':>9}{'bs=1 recalib':>14}{'recovery':>10}")
    print("-" * 53)
    for c in args.corruptions:
        src = evaluate(qmodel, corruption_loader(args.data, c, args.severity,
                       args.source_batch_size, max_samples=args.max_samples), device)
        rec = channel_recalib_eval(qmodel, corruption_loader(args.data, c, args.severity,
                       1, max_samples=args.max_samples), device, args.momentum)
        rows.append({"corruption": c, "source": src, "bs1_recalib": rec, "recovery": rec - src})
        src_errs.append(100 - src)
        rec_errs.append(100 - rec)
        print(f"{c:<20}{src:>8.2f}%{rec:>13.2f}%{rec - src:>+9.2f}%")

    mean_src_err = sum(src_errs) / len(src_errs)
    mean_rec_err = sum(rec_errs) / len(rec_errs)
    print("-" * 53)
    print(f"{'MEAN ERROR':<20}{mean_src_err:>8.2f}%{mean_rec_err:>13.2f}%{mean_src_err - mean_rec_err:>+9.2f}%")

    print("\n=== batch-size-1 (deployment) vs batch-64 ===")
    print(f"{'regime':<32}{'mean err':>10}{'recovery':>10}")
    print(f"{'bs=1 channel-recalib (m=' + str(args.momentum) + ')':<32}{mean_rec_err:>9.2f}%{mean_src_err - mean_rec_err:>+9.2f}%")
    if os.path.exists(args.bs64_results):
        with open(args.bs64_results) as f:
            b = json.load(f)
        r64 = b["mean_error_folded_source"] - b["mean_error_channel_recalib"]
        print(f"{'bs=64 channel-recalib (m=0.1)':<32}{b['mean_error_channel_recalib']:>9.2f}%{r64:>+9.2f}%")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "severity": args.severity, "momentum": args.momentum, "batch_size": 1,
            "max_samples": args.max_samples,
            "mean_error_source": mean_src_err, "mean_error_bs1_recalib": mean_rec_err,
            "rows": rows,
        }, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
