"""Head-to-head baseline comparison on CIFAR-*-C.

Mean accuracy (higher better) across the 15 corruptions for:
  source       : no adaptation (lower bound)
  BN-adapt     : forward-only BN-stat recalibration (needs BN layers) [Nado'20/LeanTTA-class]
  TENT         : backprop entropy minimization (needs gradients)       [Wang'21]
  ours         : forward-only channel-recalib on the FOLDED int8 model

The capability matrix below the table is the point: only `ours` is both forward-only
AND runnable on the deployed folded integer-only model.

Usage:
    python src/run_baselines.py --dataset cifar10 --max-samples 2000
"""
import argparse
import json
import os
import torch

from models import resnet20
from data import CORRUPTIONS, corruption_loader, clean_loaders, num_classes
from tta import evaluate, bn_adapt_eval
from tta_quant import channel_recalib_eval
from baselines import tent_eval, cotta_eval
from quant import quantize_model, calibrate_model
from fold import fold_bn_recalib
from train import pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--data", default="data")
    ap.add_argument("--severity", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--momentum", type=float, default=0.1)
    ap.add_argument("--tent-lr", type=float, default=1e-3)
    ap.add_argument("--cotta", action="store_true", help="also run CoTTA (slow: aug-averaged backprop)")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--corruptions", nargs="*", default=CORRUPTIONS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ckpt = args.ckpt or f"checkpoints/resnet20_{args.dataset}.pt"
    out = args.out or f"results/baselines_{args.dataset}.json"

    device = pick_device()
    fp = resnet20(num_classes=num_classes(args.dataset))
    fp.load_state_dict(torch.load(ckpt, map_location="cpu"))

    # folded int8 model for "ours"
    _, clean_test = clean_loaders(args.data, args.batch_size, dataset=args.dataset)
    qmodel = quantize_model(fold_bn_recalib(fp, args.momentum), 8, 8)
    calibrate_model(qmodel, clean_test, device, 8)

    def ld(c):
        return corruption_loader(args.data, c, args.severity, args.batch_size,
                                 num_workers=0, max_samples=args.max_samples, dataset=args.dataset)

    methods = ["source", "BN-adapt", "TENT"] + (["CoTTA"] if args.cotta else []) + ["ours"]
    accs = {m: [] for m in methods}
    print(f"{args.dataset} | bs={args.batch_size} | "
          + (f"first {args.max_samples}/corr" if args.max_samples else "full 10k/corr") + "\n")
    print(f"{'corruption':<18}" + "".join(f"{m:>11}" for m in methods))
    print("-" * (18 + 11 * len(methods)))
    rows = []
    for c in args.corruptions:
        r = {
            "source": evaluate(fp, ld(c), device),
            "BN-adapt": bn_adapt_eval(fp, ld(c), device, args.momentum),
            "TENT": tent_eval(fp, ld(c), device, args.tent_lr),
            "ours": channel_recalib_eval(qmodel, ld(c), device, args.momentum),
        }
        if args.cotta:
            r["CoTTA"] = cotta_eval(fp, ld(c), device)
        for m in methods:
            accs[m].append(r[m])
        rows.append({"corruption": c, **r})
        print(f"{c:<18}" + "".join(f"{r[m]:>10.2f}%" for m in methods))

    print("-" * (18 + 11 * len(methods)))
    means = {m: sum(accs[m]) / len(accs[m]) for m in methods}
    print(f"{'MEAN ACC':<18}" + "".join(f"{means[m]:>10.2f}%" for m in methods))
    print(f"{'recovery vs src':<18}" + "".join(f"{means[m]-means['source']:>+10.2f} " for m in methods))

    print("\ncapability matrix:")
    print(f"  {'method':<10}{'forward-only':>14}{'no BN needed':>14}{'folded-int8 MCU':>17}")
    caps = {"source": (True, True, True), "BN-adapt": (True, False, False),
            "TENT": (False, False, False), "CoTTA": (False, False, False),
            "ours": (True, True, True)}
    for m in methods:
        fo, nobn, mcu = caps[m]
        print(f"  {m:<10}{'yes' if fo else 'NO':>14}{'yes' if nobn else 'NO':>14}{'yes' if mcu else 'NO':>17}")
    print("  -> only 'ours' adapts forward-only on the deployed folded integer-only model")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"dataset": args.dataset, "batch_size": args.batch_size,
                   "means": means, "rows": rows}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
