"""Ablations for the FORGE recalibration on ResNet-20 / CIFAR-10-C (folded int8).

Two sweeps, single-seed full test set, recovery measured against the folded int8 source
(the deployed model), consistent with the main tables:

  momentum : mean recovery vs the EMA momentum at bs=64 -- robustness to the one
             hyperparameter the method has.
  bitwidth : clean acc, int8 source acc, and recovery at several (weight,act) bit widths
             -- the method should hold as precision drops.

Writes results/ablations.json.

Usage:
  python src/run_ablations.py
  python src/run_ablations.py --max-samples 400   # smoke
"""
import argparse
import json
import os
from statistics import mean as _mean

import torch

from models import resnet20
from data import CORRUPTIONS, corruption_loader, clean_loaders, num_classes
from tta import evaluate
from tta_quant import channel_recalib_eval
from quant import quantize_model, calibrate_model
from fold import fold_bn_recalib
from train import pick_device


def build_q(fp, momentum, clean_test, device, w_bits, a_bits):
    q = quantize_model(fold_bn_recalib(fp, momentum), w_bits, a_bits)
    calibrate_model(q, clean_test, device, a_bits)
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--severity", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--momenta", type=float, nargs="*", default=[0.01, 0.03, 0.1, 0.3, 1.0])
    ap.add_argument("--bits", nargs="*", default=["8,8", "6,6", "4,8", "4,4"])
    ap.add_argument("--out", default="results/ablations.json")
    args = ap.parse_args()

    device = pick_device()
    ds = "cifar10"
    fp = resnet20(num_classes=num_classes(ds))
    fp.load_state_dict(torch.load(f"checkpoints/resnet20_{ds}.pt", map_location="cpu"))
    _, clean_test = clean_loaders(args.data, args.batch_size, dataset=ds)

    def ld(c):
        return corruption_loader(args.data, c, args.severity, args.batch_size, num_workers=0,
                                 dataset=ds, max_samples=args.max_samples)

    out = {"dataset": ds, "batch_size": args.batch_size, "momentum": {}, "bitwidth": {}}
    os.makedirs("results", exist_ok=True)

    def save():
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)

    # ---- momentum sweep (fixed 8/8) ----
    print("== momentum sweep (int8, bs=%d) ==" % args.batch_size)
    q8 = build_q(fp, 0.1, clean_test, device, 8, 8)
    src8 = _mean(evaluate(q8, ld(c), device) for c in CORRUPTIONS)   # adapt off by default
    for m in args.momenta:
        rec = _mean(channel_recalib_eval(q8, ld(c), device, m) for c in CORRUPTIONS) - src8
        out["momentum"][str(m)] = round(rec, 2)
        save()
        print(f"  m={m:<5}  recovery {rec:+.2f}")
    out["momentum_src"] = round(src8, 2)
    save()

    # ---- bit-width sweep (momentum 0.1) ----
    print("== bit-width sweep (w,a) ==")
    for spec in args.bits:
        w, a = (int(x) for x in spec.split(","))
        q = build_q(fp, 0.1, clean_test, device, w, a)
        clean = evaluate(q, clean_test, device)
        src = _mean(evaluate(q, ld(c), device) for c in CORRUPTIONS)
        adp = _mean(channel_recalib_eval(q, ld(c), device, 0.1) for c in CORRUPTIONS)
        out["bitwidth"][spec] = {"clean": round(clean, 2), "source": round(src, 2),
                                 "adapted": round(adp, 2), "recovery": round(adp - src, 2)}
        save()
        print(f"  w{w}a{a}: clean {clean:.1f}  src {src:.1f}  adapt {adp:.1f}  rec {adp-src:+.2f}")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
