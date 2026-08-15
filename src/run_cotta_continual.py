"""CoTTA in its native CONTINUAL protocol (CIFAR-10-C).

CoTTA is designed for a long, never-reset stream: the 15 corruptions are concatenated
in sequence and the model adapts continually (its weight-EMA teacher and stochastic
restore only pay off over a long stream). Evaluating it per-corruption (episodic), like
the Table-2 baselines, starves its slow warmup and understates it. We therefore report
CoTTA continually and contrast it with the source model on the same stream.

Usage:
    python src/run_cotta_continual.py --max-samples 2000
"""
import argparse
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from models import resnet20
from data import CORRUPTIONS, CIFAR10C
from tta import evaluate
from baselines import cotta_eval
from train import pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/resnet20_cifar10.pt")
    ap.add_argument("--data", default="data")
    ap.add_argument("--severity", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-samples", type=int, default=2000)
    args = ap.parse_args()
    device = pick_device()

    fp = resnet20()
    fp.load_state_dict(torch.load(args.ckpt, map_location="cpu"))

    # concatenate corruptions into one continual stream (corruption-by-corruption order)
    dsets = []
    for c in CORRUPTIONS:
        ds = CIFAR10C(args.data, c, args.severity)
        if args.max_samples:
            ds = Subset(ds, range(args.max_samples))
        dsets.append(ds)
    loader = DataLoader(ConcatDataset(dsets), args.batch_size, shuffle=False, num_workers=0)
    n = len(CORRUPTIONS) * (args.max_samples or 10000)
    print(f"device: {device} | continual stream of {n} images ({len(CORRUPTIONS)} corruptions)\n")

    src = evaluate(fp, loader, device)
    cotta = cotta_eval(fp, loader, device)
    print(f"{'method':<28}{'acc (%)':>9}{'recovery':>10}")
    print("-" * 47)
    print(f"{'source (no adapt)':<28}{src:>8.2f}{0.0:>+10.2f}")
    print(f"{'CoTTA (continual, backprop)':<28}{cotta:>8.2f}{cotta - src:>+10.2f}")
    print(f"\nCoTTA recovers +{cotta - src:.1f} pts continually, but (like TENT) needs "
          "backpropagation and cannot run on the folded integer-only model.")


if __name__ == "__main__":
    main()
