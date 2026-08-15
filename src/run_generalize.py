"""Generalization runner: channel-recalib recovery on any CIFAR-*-C dataset.

Dataset-agnostic (--dataset cifar10|cifar100). Reports, across the 15 corruptions:
folded-int8 source acc vs forward-only channel-recalib acc, and the mean recovery.
Run at bs=64/m=0.1 (batched) and bs=1/matched-momentum (deployment) to show the
method and the single-sample finding hold beyond CIFAR-10.

Usage:
    python src/run_generalize.py --dataset cifar100 --batch-size 64 --momentum 0.1
    python src/run_generalize.py --dataset cifar100 --batch-size 1 --momentum 0.0015625 --max-samples 2000
"""
import argparse
import json
import os
import torch

from models import build_model
from data import CORRUPTIONS, corruption_loader, clean_loaders, num_classes
from tta import evaluate
from tta_quant import channel_recalib_eval_fast, set_active_recalib
from quant import quantize_model, calibrate_model
from fold import fold_for_arch
from train import pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100", choices=["cifar10", "cifar100", "tinyimagenet"])
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
    ap.add_argument("--corruptions", nargs="*", default=CORRUPTIONS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    # stem_stride does not change any weight shape, so a mismatch loads silently and
    # destroys accuracy (10.9% vs 53.9% on Tiny-ImageNet). Derive it from the dataset.
    if args.stem_stride is None:
        args.stem_stride = 2 if args.dataset == "tinyimagenet" else 1
    ckpt = args.ckpt or f"checkpoints/{args.arch}_{args.dataset}.pt"
    out = args.out or f"results/generalize_{args.arch}_{args.dataset}_bs{args.batch_size}.json"

    # dataset dispatch (Tiny-ImageNet is folder-structured -> data_tin loaders)
    if args.dataset == "tinyimagenet":
        import data_tin as D
        nc = D.NUM_CLASSES
        def clean_fn():
            return D.clean_loaders(args.data, args.batch_size)
        def corr_fn(c):
            return D.corruption_loader(args.data, c, args.severity, args.batch_size,
                                       num_workers=4, max_samples=args.max_samples)
    else:
        nc = num_classes(args.dataset)
        def clean_fn():
            return clean_loaders(args.data, args.batch_size, dataset=args.dataset)
        def corr_fn(c):
            return corruption_loader(args.data, c, args.severity, args.batch_size,
                                     num_workers=0, max_samples=args.max_samples, dataset=args.dataset)

    device = pick_device()
    fp32 = build_model(args.arch, num_classes=nc, width=args.width, stem_stride=args.stem_stride)
    fp32.load_state_dict(torch.load(ckpt, map_location="cpu"))
    qmodel = fold_for_arch(fp32, args.arch, args.momentum)
    _, clean_test = clean_fn()
    qmodel = quantize_model(qmodel, 8, 8)
    calibrate_model(qmodel, clean_test, device, 8)
    qmodel = qmodel.to(device).eval()
    for p in qmodel.parameters():
        p.requires_grad_(False)
    clean = evaluate(qmodel, clean_test, device)
    print(f"{args.arch}/{args.dataset} | bs={args.batch_size} m={args.momentum} | folded int8 clean acc {clean:.2f}%"
          + (f" | first {args.max_samples}/corruption" if args.max_samples else "") + "\n")

    # build loaders once; compute ALL sources first on the clean model (the fast
    # in-place recalib eval mutates adapt flags, so source must be measured before it).
    loaders = {c: corr_fn(c) for c in args.corruptions}
    set_active_recalib(qmodel, sites=set())          # all adapt OFF + reset
    src = {c: evaluate(qmodel, loaders[c], device) for c in args.corruptions}

    rows, src_errs, rec_errs = [], [], []
    print(f"{'corruption':<20}{'source':>9}{'recalib':>10}{'recovery':>10}")
    print("-" * 49)
    for c in args.corruptions:
        rec = channel_recalib_eval_fast(qmodel, loaders[c], device, sites=None, momentum=args.momentum)
        rows.append({"corruption": c, "source": src[c], "recalib": rec, "recovery": rec - src[c]})
        src_errs.append(100 - src[c])
        rec_errs.append(100 - rec)
        print(f"{c:<20}{src[c]:>8.2f}%{rec:>9.2f}%{rec - src[c]:>+9.2f}%")

    ms, mr = sum(src_errs) / len(src_errs), sum(rec_errs) / len(rec_errs)
    print("-" * 49)
    print(f"{'MEAN ERROR':<20}{ms:>8.2f}%{mr:>9.2f}%{ms - mr:>+9.2f}%")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"dataset": args.dataset, "batch_size": args.batch_size, "momentum": args.momentum,
                   "clean_acc": clean, "mean_error_source": ms, "mean_error_recalib": mr,
                   "mean_recovery": ms - mr, "rows": rows}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
