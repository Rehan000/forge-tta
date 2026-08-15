"""Phase 2: batch-size degradation curve — does recalib survive single-sample streaming?

The deployment regime is batch size 1 (one frame at a time). Per-channel statistics
from a single image are noisy, so a fixed EMA momentum collapses at small batches.
The fix is window-matched momentum: the effective averaging window ~ batch_size /
momentum, so scaling momentum with batch size keeps the window constant.

This produces two curves over batch sizes [64,16,4,1]:
  - fixed momentum (m=0.1)        -> naive; degrades as batch shrinks
  - window-matched (m = bs/W)     -> holds the window constant -> robust at bs=1

Usage:
    python src/run_phase2_bscurve.py --severity 5 --max-samples 2000
"""
import argparse
import json
import os
import torch
from torch.utils.data import DataLoader, Subset

from models import resnet20
from data import CIFAR10C
from tta import evaluate
from tta_quant import channel_recalib_eval_fast
from fold import fold_bn_recalib
from train import pick_device

SUBSET = ["gaussian_noise", "contrast", "defocus_blur", "fog", "pixelate"]
BATCH_SIZES = [64, 16, 4, 1]
WINDOW = 640.0   # bs/m for the bs=64, m=0.1 reference -> matched m = bs / WINDOW


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/resnet20_cifar10.pt")
    ap.add_argument("--data", default="data")
    ap.add_argument("--severity", type=int, default=5)
    ap.add_argument("--fixed-momentum", type=float, default=0.1)
    ap.add_argument("--max-samples", type=int, default=2000)
    ap.add_argument("--corruptions", nargs="*", default=SUBSET)
    ap.add_argument("--out", default="results/phase2_bscurve.json")
    args = ap.parse_args()
    corrs = args.corruptions

    device = pick_device()
    fp32 = resnet20()
    fp32.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    qmodel = fold_bn_recalib(fp32, args.fixed_momentum).to(device).eval()
    for p in qmodel.parameters():
        p.requires_grad_(False)

    # load each corruption once; make loaders at any batch size from the cached dataset
    datasets = {}
    for c in corrs:
        d = CIFAR10C(args.data, c, args.severity)
        datasets[c] = Subset(d, range(args.max_samples)) if args.max_samples else d

    def loader(c, bs):
        return DataLoader(datasets[c], batch_size=bs, shuffle=False, num_workers=0)

    src = {c: evaluate(qmodel, loader(c, 128), device) for c in corrs}

    def mean_recovery(bs, momentum):
        rs = [channel_recalib_eval_fast(qmodel, loader(c, bs), device, sites=None,
              momentum=momentum) - src[c] for c in corrs]
        return sum(rs) / len(rs)

    print(f"device: {device} | FP-folded | {len(corrs)} corruptions | {args.max_samples} imgs/corruption")
    print(f"window-matched momentum = bs/{WINDOW:.0f}\n")
    print(f"{'batch':>6} {'fixed m=0.1':>13} {'matched m=bs/640':>18} {'matched m':>11}")

    rows = []
    for bs in BATCH_SIZES:
        m_fixed = args.fixed_momentum
        m_match = min(bs / WINDOW, 1.0)
        r_fixed = mean_recovery(bs, m_fixed)
        r_match = mean_recovery(bs, m_match)
        rows.append({"batch_size": bs, "fixed": r_fixed, "matched": r_match, "matched_momentum": m_match})
        print(f"{bs:>6} {r_fixed:>+12.2f} {r_match:>+17.2f} {m_match:>11.5f}")

    bs1 = rows[-1]
    bs64 = rows[0]
    print(f"\nbs=1 vs bs=64 recovery drop:")
    print(f"  fixed momentum:   +{bs64['fixed']:.2f} -> +{bs1['fixed']:.2f} "
          f"(lost {bs64['fixed']-bs1['fixed']:.2f}pt)")
    print(f"  window-matched:   +{bs64['matched']:.2f} -> +{bs1['matched']:.2f} "
          f"(lost {bs64['matched']-bs1['matched']:.2f}pt)")
    print("  -> window-matching rescues single-sample streaming"
          if bs1['matched'] > bs1['fixed'] + 1 else "  -> matching gives little benefit here")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"severity": args.severity, "corruptions": corrs, "window": WINDOW,
                   "fixed_momentum": args.fixed_momentum, "rows": rows}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
