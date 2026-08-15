"""Phase 2: selective-layer recalibration — how few layers do we actually need?

Adapting all 21 recalib sites recovers ~+20pt but costs a recalibration at every
layer. This asks: which layers matter, and how much of the benefit survives if we
adapt only a few? A strong "knee" (most benefit from a handful of layers) is a real
edge-efficiency contribution — and it directly cuts the on-device adaptation energy.

Two analyses (on a representative corruption subset for speed):
  1. per-layer-alone : adapt ONLY site i -> mean recovery (which single layers help)
  2. cumulative depth : adapt sites {0..k-1} for k=1..N -> recovery vs #layers adapted

Usage:
    python src/run_phase2_selective.py --severity 5 --max-samples 2000
"""
import argparse
import json
import os
import torch

from models import resnet20
from data import corruption_loader, clean_loaders
from tta import evaluate
from tta_quant import channel_recalib_eval_fast, num_recalib_sites
from quant import quantize_model, calibrate_model
from fold import fold_bn_recalib
from train import pick_device

# diverse subset: heavy noise, contrast, blur, weather, digital
SUBSET = ["gaussian_noise", "contrast", "defocus_blur", "fog", "pixelate"]


def mean_recovery(qmodel, device, args, sites, src_cache, loaders):
    """Mean over the corruption subset of (recalib_acc - source_acc). Reuses loaders."""
    recs = []
    for c in SUBSET:
        acc = channel_recalib_eval_fast(qmodel, loaders[c], device, sites=sites, momentum=args.momentum)
        recs.append(acc - src_cache[c])
    return sum(recs) / len(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/resnet20_cifar10.pt")
    ap.add_argument("--data", default="data")
    ap.add_argument("--severity", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--momentum", type=float, default=0.1)
    ap.add_argument("--max-samples", type=int, default=2000)
    ap.add_argument("--quant", action="store_true", help="use int8 model (slower); default FP-folded")
    ap.add_argument("--out", default="results/phase2_selective.json")
    args = ap.parse_args()

    device = pick_device()
    fp32 = resnet20()
    fp32.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    folded = fold_bn_recalib(fp32, args.momentum)
    N = num_recalib_sites(folded)
    _, clean_test = clean_loaders(args.data, args.batch_size)
    # Selective-layer analysis runs on the FP folded model: the recalib-importance
    # pattern is independent of int8 (we showed int8 ~= FP for recalib), and FP avoids
    # the per-forward fake-quant overhead (~10x faster for this 42-config sweep).
    if args.quant:
        qmodel = quantize_model(folded, 8, 8)
        calibrate_model(qmodel, clean_test, device, 8)
    else:
        qmodel = folded
    qmodel = qmodel.to(device).eval()
    for p in qmodel.parameters():
        p.requires_grad_(False)
    print(f"device: {device} | {N} recalib sites | model={'int8' if args.quant else 'FP-folded'} "
          f"| subset {SUBSET} | {args.max_samples} imgs/corruption\n")

    # build each corruption loader ONCE (avoids reloading the 150MB npy per eval)
    # num_workers=0: data is already in RAM; workers would respawn each re-iteration
    # (macOS spawn ~1-2s) and dominate this 200+ eval sweep.
    loaders = {c: corruption_loader(args.data, c, args.severity, args.batch_size,
                                    num_workers=0, max_samples=args.max_samples) for c in SUBSET}
    src_cache = {c: evaluate(qmodel, loaders[c], device) for c in SUBSET}

    full = mean_recovery(qmodel, device, args, None, src_cache, loaders)
    print(f"ALL {N} layers adapted: mean recovery +{full:.2f}pt (reference ceiling)\n")

    # 1. per-layer-alone sensitivity
    print("per-layer-alone recovery (adapt only site i):")
    alone = []
    for i in range(N):
        r = mean_recovery(qmodel, device, args, {i}, src_cache, loaders)
        alone.append(r)
        print(f"  site {i:2d}: +{r:5.2f}pt")
    order = sorted(range(N), key=lambda i: -alone[i])
    print(f"\ntop sites by individual recovery: {order[:6]}")

    # 2. cumulative in depth order (first k layers)
    print("\ncumulative recovery, depth order (adapt first k sites):")
    cum_depth = []
    for k in range(1, N + 1):
        r = mean_recovery(qmodel, device, args, set(range(k)), src_cache, loaders)
        cum_depth.append(r)
        bar = "#" * int(r / max(full, 1e-6) * 30)
        print(f"  first {k:2d} layers: +{r:5.2f}pt ({r/full*100:4.0f}% of full) {bar}")

    # 3. cumulative in IMPORTANCE order (greedy: add layers best-individual-first)
    print("\ncumulative recovery, importance order (greedy, best layers first):")
    cum_imp = []
    for k in range(1, N + 1):
        r = mean_recovery(qmodel, device, args, set(order[:k]), src_cache, loaders)
        cum_imp.append(r)
        bar = "#" * int(r / max(full, 1e-6) * 30)
        print(f"  top {k:2d} layers: +{r:5.2f}pt ({r/full*100:4.0f}% of full) {bar}")

    knee_d = next((k for k in range(1, N + 1) if cum_depth[k - 1] >= 0.9 * full), N)
    knee_i = next((k for k in range(1, N + 1) if cum_imp[k - 1] >= 0.9 * full), N)
    print(f"\nKNEE (90% of full +{full:.2f}pt): depth-order {knee_d}/{N} layers | "
          f"importance-order {knee_i}/{N} layers")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"severity": args.severity, "n_sites": N, "subset": SUBSET,
                   "full_recovery": full, "per_layer_alone": alone,
                   "importance_order": order,
                   "cumulative_depth": cum_depth, "cumulative_importance": cum_imp,
                   "knee_depth_90pct": knee_d, "knee_importance_90pct": knee_i}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
