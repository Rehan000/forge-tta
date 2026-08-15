"""Phase 2 (rigorous): held-out selective recalibration — no selection bias.

The earlier selective study ranked layers on the same corruptions it evaluated on.
This fixes that: rank layer importance on a SELECTION set of corruptions, then
evaluate the selected layers on a DISJOINT TEST set. If a few held-out-selected
layers still recover most of the benefit on unseen corruptions, the finding is real.

Reports:
  - test-set cumulative recovery using the SELECTION-derived ranking (held-out)
  - vs an ORACLE ranking derived on the test set itself (upper bound)
  - knee (layers to reach 90% of full) for each
  - consistency: overlap of the top-6 important layers between selection and test

Usage:
    python src/run_phase2_heldout.py --severity 5 --max-samples 2000
"""
import argparse
import json
import os
import torch

from models import resnet20
from data import CORRUPTIONS, corruption_loader, clean_loaders
from tta import evaluate
from tta_quant import channel_recalib_eval_fast, num_recalib_sites
from fold import fold_bn_recalib
from train import pick_device

# disjoint split: one corruption from each family for selection, the rest for test
SEL = ["gaussian_noise", "motion_blur", "snow", "contrast"]
TEST = [c for c in CORRUPTIONS if c not in SEL]


def mean_rec(qmodel, device, args, sites, corrs, src, loaders):
    rs = [channel_recalib_eval_fast(qmodel, loaders[c], device, sites=sites,
          momentum=args.momentum) - src[c] for c in corrs]
    return sum(rs) / len(rs)


def per_layer_rank(qmodel, device, args, corrs, src, loaders, N):
    alone = [mean_rec(qmodel, device, args, {i}, corrs, src, loaders) for i in range(N)]
    order = sorted(range(N), key=lambda i: -alone[i])
    return order, alone


def cumulative(qmodel, device, args, corrs, src, loaders, ranking, N):
    return [mean_rec(qmodel, device, args, set(ranking[:k]), corrs, src, loaders)
            for k in range(1, N + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/resnet20_cifar10.pt")
    ap.add_argument("--data", default="data")
    ap.add_argument("--severity", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--momentum", type=float, default=0.1)
    ap.add_argument("--max-samples", type=int, default=2000)
    ap.add_argument("--out", default="results/phase2_heldout.json")
    args = ap.parse_args()

    device = pick_device()
    fp32 = resnet20()
    fp32.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    qmodel = fold_bn_recalib(fp32, args.momentum).to(device).eval()
    for p in qmodel.parameters():
        p.requires_grad_(False)
    N = num_recalib_sites(qmodel)
    print(f"device: {device} | {N} sites | SEL={SEL} | TEST={len(TEST)} corruptions | {args.max_samples} imgs\n")

    loaders = {c: corruption_loader(args.data, c, args.severity, args.batch_size,
               num_workers=0, max_samples=args.max_samples) for c in CORRUPTIONS}
    src = {c: evaluate(qmodel, loaders[c], device) for c in CORRUPTIONS}

    # rankings
    sel_order, sel_alone = per_layer_rank(qmodel, device, args, SEL, src, loaders, N)
    test_order, test_alone = per_layer_rank(qmodel, device, args, TEST, src, loaders, N)
    overlap = len(set(sel_order[:6]) & set(test_order[:6]))
    print(f"top-6 important layers: selection {sorted(sel_order[:6])} | test {sorted(test_order[:6])}")
    print(f"  overlap: {overlap}/6  (high = importance is consistent across corruptions)\n")

    # full recovery on test, and cumulative with each ranking
    full_test = mean_rec(qmodel, device, args, None, TEST, src, loaders)
    cum_heldout = cumulative(qmodel, device, args, TEST, src, loaders, sel_order, N)
    cum_oracle = cumulative(qmodel, device, args, TEST, src, loaders, test_order, N)

    print(f"full recovery on TEST (all {N} layers): +{full_test:.2f}pt\n")
    print(f"{'k':>3} {'held-out sel':>14} {'oracle':>10}")
    for k in range(1, N + 1):
        print(f"{k:>3} {cum_heldout[k-1]:>+13.2f} {cum_oracle[k-1]:>+10.2f}")

    knee_h = next((k for k in range(1, N + 1) if cum_heldout[k - 1] >= 0.9 * full_test), N)
    knee_o = next((k for k in range(1, N + 1) if cum_oracle[k - 1] >= 0.9 * full_test), N)
    print(f"\nKNEE (90% of full +{full_test:.2f}pt on UNSEEN test corruptions):")
    print(f"  held-out selection: {knee_h}/{N} layers (+{cum_heldout[knee_h-1]:.2f})")
    print(f"  oracle (upper bnd): {knee_o}/{N} layers (+{cum_oracle[knee_o-1]:.2f})")
    print(f"  -> held-out costs {knee_h - knee_o:+d} extra layers vs oracle "
          f"({'generalizes well' if knee_h - knee_o <= 2 else 'some selection gap'})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"severity": args.severity, "sel_corruptions": SEL, "test_corruptions": TEST,
                   "sel_order": sel_order, "test_order": test_order, "top6_overlap": overlap,
                   "full_recovery_test": full_test, "cum_heldout": cum_heldout,
                   "cum_oracle": cum_oracle, "knee_heldout": knee_h, "knee_oracle": knee_o}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
