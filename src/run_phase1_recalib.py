"""Phase 1, step 3b: forward-only per-channel recalibration on the folded int8 model.

The folded-model analog of BN-stat adaptation: recover accuracy on a deployed
integer-only model (no BN layers) by re-normalizing each channel to its clean
(beta, |gamma|) target from test-stream statistics. Forward-only, no gradients.

Reports per corruption: folded source vs channel-recalib, and the full Phase-1
recovery picture vs the dead scale-adapt lever and the BN-preserved ceiling.

Usage:
    python src/run_phase1_recalib.py --severity 5
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
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--momentum", type=float, default=0.1)
    ap.add_argument("--w-bits", type=int, default=8)
    ap.add_argument("--a-bits", type=int, default=8)
    ap.add_argument("--calib-batches", type=int, default=8)
    ap.add_argument("--corruptions", nargs="*", default=CORRUPTIONS)
    ap.add_argument("--scale-results", default="results/phase1_method.json")
    ap.add_argument("--preserved-results", default="results/phase1_quant.json")
    ap.add_argument("--out", default="results/phase1_recalib.json")
    args = ap.parse_args()

    device = pick_device()
    print(f"device: {device}  |  channel-recalib: momentum={args.momentum}")

    fp32 = resnet20()
    fp32.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    fr = fold_bn_recalib(fp32, args.momentum)
    _, clean_test = clean_loaders(args.data, args.batch_size)
    qmodel = quantize_model(fr, args.w_bits, args.a_bits)
    calibrate_model(qmodel, clean_test, device, args.calib_batches)
    print(f"folded+recalib int8 clean acc: {evaluate(qmodel, clean_test, device):.2f}%")

    rows, src_errs, rec_errs = [], [], []
    print(f"\n{'corruption':<20}{'folded src':>11}{'recalib':>10}{'recovery':>10}")
    print("-" * 51)
    for c in args.corruptions:
        loader = corruption_loader(args.data, c, args.severity, args.batch_size)
        src = evaluate(qmodel, loader, device)
        loader = corruption_loader(args.data, c, args.severity, args.batch_size)
        rec = channel_recalib_eval(qmodel, loader, device, args.momentum)
        rows.append({"corruption": c, "folded_source": src, "channel_recalib": rec, "recovery": rec - src})
        src_errs.append(100 - src)
        rec_errs.append(100 - rec)
        print(f"{c:<20}{src:>10.2f}%{rec:>9.2f}%{rec - src:>+9.2f}%")

    mean_src_err = sum(src_errs) / len(src_errs)
    mean_rec_err = sum(rec_errs) / len(rec_errs)
    print("-" * 51)
    print(f"{'MEAN ERROR':<20}{mean_src_err:>10.2f}%{mean_rec_err:>9.2f}%{mean_src_err - mean_rec_err:>+9.2f}%")

    print("\n=== Phase 1 recovery picture: forward-only TTA on the deployable folded int8 model ===")
    print(f"{'method':<40}{'mean err':>10}{'recovery':>10}")
    print(f"{'folded source (no adapt)':<40}{mean_src_err:>9.2f}%{0.0:>+9.2f}%")
    if os.path.exists(args.scale_results):
        with open(args.scale_results) as f:
            sm = json.load(f)
        sr = sm["mean_error_folded_source"] - sm["mean_error_scale_adapt"]
        print(f"{'scale-adapt (quant scales only)':<40}{sm['mean_error_scale_adapt']:>9.2f}%{sr:>+9.2f}%")
    print(f"{'channel-recalib (ours, forward-only)':<40}{mean_rec_err:>9.2f}%{mean_src_err - mean_rec_err:>+9.2f}%")
    if os.path.exists(args.preserved_results):
        with open(args.preserved_results) as f:
            pr = json.load(f)
        ceil = pr["mean_error_int8_source"] - pr["mean_error_int8_bn_adapt"]
        print(f"{'(ceiling: BN-adapt, BN-preserved)':<40}{pr['mean_error_int8_bn_adapt']:>9.2f}%{ceil:>+9.2f}%")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "w_bits": args.w_bits, "a_bits": args.a_bits, "severity": args.severity,
            "momentum": args.momentum,
            "mean_error_folded_source": mean_src_err, "mean_error_channel_recalib": mean_rec_err,
            "rows": rows,
        }, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
