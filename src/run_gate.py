"""When-to-adapt gate: skip recalibration on near-clean streams.

FORGE recovers most on severe corruptions but can slightly hurt on already-mild ones, where
the deployed model is already confident (Table per-corruption). A cheap forward-only gate
makes adaptation *safe*: adapt only when the deployed model's mean confidence on the stream
falls a fixed fraction below its clean-calibration confidence. The confidence is the same
forward pass the model already runs (mean top-1 softmax probability), so the gate adds no
new computation and no gradients.

Stream-level demonstration on CIFAR-10-C / ResNet-20 (folded int8). The threshold is
calibrated only on clean data (tau = rho * clean confidence); we report the gate at the
default rho and a small sensitivity sweep. Writes results/gate.json.

Usage:
  python src/run_gate.py
  python src/run_gate.py --max-samples 400   # smoke
"""
import argparse
import json
import os
from statistics import mean as _mean

import torch

from models import resnet20
from data import CORRUPTIONS, corruption_loader, clean_loaders, num_classes
from tta_quant import channel_recalib_eval, configure_channel_recalib
from quant import quantize_model, calibrate_model
from fold import fold_bn_recalib
from train import pick_device


@torch.no_grad()
def acc_and_conf(model, loader, device):
    """Top-1 accuracy and mean top-1 softmax confidence over the stream (adapt off)."""
    model = model.to(device).eval()
    correct = total = 0
    conf_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        p = model(x).softmax(1)
        mx, pred = p.max(1)
        correct += (pred == y).sum().item()
        total += y.numel()
        conf_sum += mx.sum().item()
    return 100.0 * correct / total, conf_sum / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--severity", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--momentum", type=float, default=0.1)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--rho", type=float, default=0.9, help="adapt if stream conf < rho*clean conf")
    ap.add_argument("--out", default="results/gate.json")
    args = ap.parse_args()

    device = pick_device()
    ds = "cifar10"
    fp = resnet20(num_classes=num_classes(ds))
    fp.load_state_dict(torch.load(f"checkpoints/resnet20_{ds}.pt", map_location="cpu"))
    _, clean_test = clean_loaders(args.data, args.batch_size, dataset=ds)
    q = quantize_model(fold_bn_recalib(fp, args.momentum), 8, 8)
    calibrate_model(q, clean_test, device, 8)

    def ld(c):
        return corruption_loader(args.data, c, args.severity, args.batch_size, num_workers=0,
                                 dataset=ds, max_samples=args.max_samples)

    _, conf_clean = acc_and_conf(q, clean_test, device)          # deployed model, adapt off
    print(f"clean confidence (deployed int8): {conf_clean:.3f}\n")

    rows = []
    print(f"{'corruption':<16}{'src':>7}{'conf':>7}{'adapt':>8}{'aconf':>7}{'dConf':>7}")
    for c in CORRUPTIONS:
        src, conf = acc_and_conf(q, ld(c), device)               # adapt off: source + confidence
        m_adp = configure_channel_recalib(q, args.momentum, None)
        adp, aconf = acc_and_conf(m_adp, ld(c), device)          # adapt on: acc + confidence
        rows.append({"corruption": c, "source": src, "conf": conf,
                     "adapted": adp, "adapt_conf": aconf})
        print(f"{c:<16}{src:>6.1f}%{conf:>7.3f}{adp:>7.1f}%{aconf:>7.3f}{aconf-conf:>+7.3f}")

    msrc = _mean(r["source"] for r in rows)
    mung = _mean(r["adapted"] for r in rows)
    hurt_ung = sum(r["adapted"] < r["source"] - 1e-6 for r in rows)

    # Gate A: absolute confidence (adapt if stream conf < rho * clean conf)
    def gate_abs(rho):
        tau = rho * conf_clean
        gated = [(r["adapted"] if r["conf"] < tau else r["source"]) for r in rows]
        return {"rho": rho, "tau": round(tau, 3),
                "n_adapt": sum(r["conf"] < tau for r in rows),
                "rec": round(_mean(gated) - msrc, 2),
                "hurt": sum(g < r["source"] - 1e-6 for g, r in zip(gated, rows))}

    # Gate B: confidence-improvement (keep adaptation only if it raises mean confidence)
    def gate_imp(margin):
        gated = [(r["adapted"] if r["adapt_conf"] - r["conf"] > margin else r["source"]) for r in rows]
        return {"margin": margin,
                "n_adapt": sum(r["adapt_conf"] - r["conf"] > margin for r in rows),
                "rec": round(_mean(gated) - msrc, 2),
                "hurt": sum(g < r["source"] - 1e-6 for g, r in zip(gated, rows))}

    out = {"dataset": ds, "conf_clean": round(conf_clean, 3), "rows": rows,
           "ungated_rec": round(mung - msrc, 2), "hurt_ungated": hurt_ung,
           "gate_abs": [gate_abs(r) for r in [0.85, 0.90, 0.95]],
           "gate_improvement": [gate_imp(m) for m in [-0.005, 0.0, 0.005]]}
    os.makedirs("results", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nungated recovery {mung-msrc:+.2f}  (hurts {hurt_ung}/15 corruptions)")
    print("Gate A (absolute conf):   rho/#adapt/rec/hurt:",
          [(g["rho"], g["n_adapt"], g["rec"], g["hurt"]) for g in out["gate_abs"]])
    print("Gate B (conf-improvement): margin/#adapt/rec/hurt:",
          [(g["margin"], g["n_adapt"], g["rec"], g["hurt"]) for g in out["gate_improvement"]])
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
