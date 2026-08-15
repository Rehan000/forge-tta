"""Cross-dataset when-to-adapt gate (Reviewer ma3M point 3, generalization).

Generalizes run_gate.py beyond CIFAR-10/ResNet-20 to CIFAR-100 (ResNet-20) and
Tiny-ImageNet (MobileNetV2), so the confidence-improvement gate's claim ("keep
adaptation only when it raises the deployed model's own confidence, and no corruption
is degraded") can be checked across datasets/architectures rather than a single run.

The gate is the confidence-improvement rule (Gate B): on each stream, compare the
deployed model's mean top-1 softmax confidence with adaptation OFF (conf) vs ON (aconf),
and keep adaptation only if aconf - conf > margin (default 0). Both confidences are the
same forward passes the model already runs; in deployment this is a short calibration
window (see paper Sec. 4.8). We report, per dataset: ungated recovery and #corruptions
hurt, vs gated recovery and #hurt.

Usage:
  python src/run_gate_general.py --dataset cifar100 --arch resnet20
  python src/run_gate_general.py --dataset tinyimagenet --arch mobilenetv2
  python src/run_gate_general.py --dataset cifar10 --arch resnet20 --max-samples 400  # smoke
"""
import argparse
import json
import os
from statistics import mean as _mean

import torch

from models import build_model
from data import CORRUPTIONS, corruption_loader, clean_loaders, num_classes
from tta_quant import configure_channel_recalib
from quant import quantize_model, calibrate_model
from fold import fold_for_arch
from train import pick_device


@torch.no_grad()
def acc_and_conf(model, loader, device):
    """Top-1 accuracy and mean top-1 softmax confidence over the stream."""
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
    ap.add_argument("--dataset", default="cifar100",
                    choices=["cifar10", "cifar100", "tinyimagenet"])
    ap.add_argument("--arch", default="resnet20", choices=["resnet20", "mobilenetv2"])
    ap.add_argument("--width", type=float, default=0.5)
    ap.add_argument("--stem-stride", type=int, default=None,
                    help="default 2 for tinyimagenet (as the released checkpoint was trained), else 1")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--data", default="data")
    ap.add_argument("--severity", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--momentum", type=float, default=0.1)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--margin", type=float, default=0.0, help="keep adapt if aconf-conf>margin")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    # stem_stride does not change any weight shape, so a mismatch loads silently and
    # destroys accuracy (10.9% vs 53.9% on Tiny-ImageNet). Derive it from the dataset.
    if args.stem_stride is None:
        args.stem_stride = 2 if args.dataset == "tinyimagenet" else 1
    ckpt = args.ckpt or f"checkpoints/{args.arch}_{args.dataset}.pt"
    out = args.out or f"results/gate_{args.arch}_{args.dataset}.json"

    # dataset dispatch (Tiny-ImageNet is folder-structured -> data_tin loaders)
    if args.dataset == "tinyimagenet":
        import data_tin as D
        nc = D.NUM_CLASSES
        corruptions = D.CORRUPTIONS
        def clean_fn():
            return D.clean_loaders(args.data, args.batch_size)
        def corr_fn(c):
            return D.corruption_loader(args.data, c, args.severity, args.batch_size,
                                       num_workers=4, max_samples=args.max_samples)
    else:
        nc = num_classes(args.dataset)
        corruptions = CORRUPTIONS
        def clean_fn():
            return clean_loaders(args.data, args.batch_size, dataset=args.dataset)
        def corr_fn(c):
            return corruption_loader(args.data, c, args.severity, args.batch_size,
                                     num_workers=0, max_samples=args.max_samples,
                                     dataset=args.dataset)

    device = pick_device()
    fp32 = build_model(args.arch, num_classes=nc, width=args.width, stem_stride=args.stem_stride)
    fp32.load_state_dict(torch.load(ckpt, map_location="cpu"))
    q = fold_for_arch(fp32, args.arch, args.momentum)
    _, clean_test = clean_fn()
    q = quantize_model(q, 8, 8)
    calibrate_model(q, clean_test, device, 8)
    q = q.to(device).eval()
    for p in q.parameters():
        p.requires_grad_(False)

    _, conf_clean = acc_and_conf(q, clean_test, device)
    print(f"{args.arch}/{args.dataset} | clean confidence (deployed int8): {conf_clean:.3f}\n")

    rows = []
    print(f"{'corruption':<18}{'src':>7}{'conf':>7}{'adapt':>8}{'aconf':>7}{'dConf':>7}")
    for c in corruptions:
        src, conf = acc_and_conf(q, corr_fn(c), device)                 # adapt OFF
        m_adp = configure_channel_recalib(q, args.momentum, None)       # fresh adapt-ON copy
        adp, aconf = acc_and_conf(m_adp, corr_fn(c), device)            # adapt ON
        rows.append({"corruption": c, "source": src, "conf": conf,
                     "adapted": adp, "adapt_conf": aconf})
        print(f"{c:<18}{src:>6.1f}%{conf:>7.3f}{adp:>7.1f}%{aconf:>7.3f}{aconf-conf:>+7.3f}")

    msrc = _mean(r["source"] for r in rows)
    mung = _mean(r["adapted"] for r in rows)
    hurt_ung = sum(r["adapted"] < r["source"] - 1e-6 for r in rows)

    def gate_imp(margin):
        gated = [(r["adapted"] if r["adapt_conf"] - r["conf"] > margin else r["source"])
                 for r in rows]
        return {"margin": margin,
                "n_adapt": sum(r["adapt_conf"] - r["conf"] > margin for r in rows),
                "rec": round(_mean(gated) - msrc, 2),
                "hurt": sum(g < r["source"] - 1e-6 for g, r in zip(gated, rows))}

    gates = [gate_imp(m) for m in [-0.005, 0.0, 0.005]]
    out_d = {"dataset": args.dataset, "arch": args.arch, "n_corruptions": len(rows),
             "conf_clean": round(conf_clean, 3), "rows": rows,
             "ungated_rec": round(mung - msrc, 2), "hurt_ungated": hurt_ung,
             "gate_improvement": gates}
    os.makedirs("results", exist_ok=True)
    with open(out, "w") as f:
        json.dump(out_d, f, indent=2)

    g0 = next(g for g in gates if g["margin"] == 0.0)
    print(f"\nungated recovery {mung-msrc:+.2f}  (hurts {hurt_ung}/{len(rows)})")
    print(f"gated  recovery {g0['rec']:+.2f}  (hurts {g0['hurt']}/{len(rows)}, "
          f"adapts {g0['n_adapt']}/{len(rows)})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
