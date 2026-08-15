"""Multi-seed error bars for the headline recovery numbers.

Online test-time adaptation is order-dependent: BN-adapt's batch statistics, TENT's
gradient trajectory, and our channel-recalibration EMA all depend on the order in which
samples arrive. We therefore hold each trained checkpoint fixed and re-run adaptation
under N seeded stream orderings (a valid re-run for online TTA), and report the mean
recovery across the 15 corruptions as mean +/- std over seeds.

Source (no adaptation) is order-independent, so it is computed once (deterministic) and
has no error bar. Each config reproduces the exact eval path of its results table:
  - baselines_cifar10 : the run_baselines.py path (Table: baselines).
  - generalize        : the run_generalize.py path (Table: generalization).

Results are written incrementally to results/multiseed_<config>.json and the run is
resumable: re-invoking skips seeds already recorded.

Usage:
  python src/run_multiseed.py --config baselines_cifar10 --seeds 5
  python src/run_multiseed.py --config generalize --seeds 5
  python src/run_multiseed.py --config baselines_cifar10 --seeds 2 --max-samples 400   # smoke
"""
import argparse
import json
import os
from statistics import mean as _mean, stdev

import torch

from models import resnet20, build_model
from data import CORRUPTIONS, corruption_loader, clean_loaders, num_classes
from tta import evaluate, bn_adapt_eval
from tta_quant import channel_recalib_eval, channel_recalib_eval_fast, set_active_recalib
from baselines import tent_eval
from quant import quantize_model, calibrate_model
from fold import fold_bn_recalib, fold_for_arch
from train import pick_device


def agg(vals):
    return {"vals": vals, "mean": _mean(vals), "std": (stdev(vals) if len(vals) > 1 else 0.0)}


def load_existing(out):
    if os.path.exists(out):
        with open(out) as f:
            return json.load(f)
    return None


def save(out, obj):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(obj, f, indent=2)


def summarize(per_seed, methods):
    """per_seed: {seed_str: {method: recovery}} -> {method: {mean,std,vals}}."""
    return {m: agg([per_seed[s][m] for s in sorted(per_seed, key=int)]) for m in methods}


def run_baselines_cifar10(seeds, args, device, out):
    ds, bs, mom = "cifar10", args.batch_size, args.momentum
    methods = ["BN-adapt", "TENT", "ours"]
    fp = resnet20(num_classes=num_classes(ds))
    fp.load_state_dict(torch.load(f"checkpoints/resnet20_{ds}.pt", map_location="cpu"))
    _, clean_test = clean_loaders(args.data, bs, dataset=ds)
    q = quantize_model(fold_bn_recalib(fp, mom), 8, 8)
    calibrate_model(q, clean_test, device, 8)

    def ld(c, seed):
        return corruption_loader(args.data, c, args.severity, bs, num_workers=0,
                                 dataset=ds, max_samples=args.max_samples, seed=seed)

    state = load_existing(out) or {"config": "baselines_cifar10", "batch_size": bs,
                                   "momentum": mom, "per_seed": {}}
    if "src_mean" not in state:
        state["src_mean"] = _mean(evaluate(fp, ld(c, None), device) for c in args.corruptions)
        print(f"source mean acc = {state['src_mean']:.2f}%")
    src = state["src_mean"]

    for s in seeds:
        if str(s) in state["per_seed"]:
            print(f"seed {s}: already done, skipping")
            continue
        bn = _mean(bn_adapt_eval(fp, ld(c, s), device, mom) for c in args.corruptions)
        te = _mean(tent_eval(fp, ld(c, s), device, args.tent_lr) for c in args.corruptions)
        ou = _mean(channel_recalib_eval(q, ld(c, s), device, mom) for c in args.corruptions)
        state["per_seed"][str(s)] = {"BN-adapt": bn - src, "TENT": te - src, "ours": ou - src}
        state["summary"] = summarize(state["per_seed"], methods)
        save(out, state)
        print(f"seed {s}: BN-adapt {bn-src:+.2f}  TENT {te-src:+.2f}  ours {ou-src:+.2f}  -> saved")
    return state


GEN_CONFIGS = [  # (arch, dataset, width, stem_stride) -- Table-4 cells that use the .npy C loaders
    ("resnet20", "cifar10", 0.5, 1),
    ("resnet20", "cifar100", 0.5, 1),
    ("mobilenetv2", "cifar10", 0.5, 1),
]


def run_generalize(seeds, args, device, out):
    state = load_existing(out) or {"config": "generalize", "batch_size": args.batch_size,
                                   "momentum": args.momentum, "cells": {}}
    for arch, ds, width, stem in GEN_CONFIGS:
        tag = f"{arch}_{ds}"
        nc = num_classes(ds)
        fp32 = build_model(arch, num_classes=nc, width=width, stem_stride=stem)
        fp32.load_state_dict(torch.load(f"checkpoints/{arch}_{ds}.pt", map_location="cpu"))
        q = fold_for_arch(fp32, arch, args.momentum)
        _, clean_test = clean_loaders(args.data, args.batch_size, dataset=ds)
        q = quantize_model(q, 8, 8)
        calibrate_model(q, clean_test, device, 8)
        q = q.to(device).eval()
        for p in q.parameters():
            p.requires_grad_(False)

        def ld(c, seed):
            return corruption_loader(args.data, c, args.severity, args.batch_size, num_workers=0,
                                     dataset=ds, max_samples=args.max_samples, seed=seed)

        cell = state["cells"].get(tag, {"per_seed": {}})
        if "src_mean" not in cell:
            set_active_recalib(q, sites=set())                          # all adapt OFF + reset
            cell["src_mean"] = _mean(evaluate(q, ld(c, None), device) for c in args.corruptions)
        src = cell["src_mean"]
        for s in seeds:
            if str(s) in cell["per_seed"]:
                continue
            rec = _mean(channel_recalib_eval_fast(q, ld(c, s), device, sites=None,
                                                  momentum=args.momentum) for c in args.corruptions)
            cell["per_seed"][str(s)] = {"ours": rec - src}
            cell["summary"] = summarize(cell["per_seed"], ["ours"])
            state["cells"][tag] = cell
            save(out, state)
            print(f"{tag} seed {s}: ours {rec-src:+.2f}  -> saved")
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=["baselines_cifar10", "generalize"])
    ap.add_argument("--seeds", type=int, default=5, help="number of stream-order seeds (0..N-1)")
    ap.add_argument("--data", default="data")
    ap.add_argument("--severity", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--momentum", type=float, default=0.1)
    ap.add_argument("--tent-lr", type=float, default=1e-3)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--corruptions", nargs="*", default=CORRUPTIONS)
    args = ap.parse_args()

    device = pick_device()
    seeds = list(range(args.seeds))
    out = f"results/multiseed_{args.config}.json"
    print(f"device={device} | config={args.config} | seeds={seeds} | "
          + (f"{args.max_samples}/corr" if args.max_samples else "full") + f" -> {out}\n")

    if args.config == "baselines_cifar10":
        state = run_baselines_cifar10(seeds, args, device, out)
        print("\nSUMMARY (mean recovery +/- std over seeds):")
        for m, a in state["summary"].items():
            print(f"  {m:<10} {a['mean']:+.2f} +/- {a['std']:.2f}")
    else:
        state = run_generalize(seeds, args, device, out)
        print("\nSUMMARY (mean recovery +/- std over seeds):")
        for tag, cell in state["cells"].items():
            a = cell["summary"]["ours"]
            print(f"  {tag:<22} {a['mean']:+.2f} +/- {a['std']:.2f}")


if __name__ == "__main__":
    main()
