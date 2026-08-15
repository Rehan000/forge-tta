"""Export one CIFAR-10-C sample + its golden logits as a C header for the firmware.

Writes deploy/esp32s3/main/test_image.h:
  test_image[3*32*32]  normalized float CHW sample (what the model sees)
  golden_logits[10]    the true-integer-engine logits for that exact sample

On-device the firmware feeds test_image through its ported forward() and must match
golden_logits to <1e-2 before any latency/energy number is trusted.

Usage:
    python deploy/export_test_image.py --corruption gaussian_noise --index 0
"""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data import CIFAR10C                           # noqa: E402
sys.path.insert(0, os.path.dirname(__file__))
from int8_reference import IntResNet20              # noqa: E402

ART = os.path.join(os.path.dirname(__file__), "artifacts", "resnet20_int8.npz")
OUT = os.path.join(os.path.dirname(__file__), "esp32s3", "main", "test_image.h")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corruption", default="gaussian_noise")
    ap.add_argument("--severity", type=int, default=5)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--espnn", action="store_true",
                    help="esp_nn-scheme golden: quantize each conv output to int8 @ out_scale "
                         "(matches the ESP-NN firmware, which requantizes conv outputs)")
    args = ap.parse_args()

    ds = CIFAR10C("data", args.corruption, args.severity)
    img, label = ds[args.index]                      # img: normalized CHW float tensor
    x = img.unsqueeze(0).numpy().astype(np.float32)

    eng = IntResNet20(ART)
    if args.espnn:
        eng.quant_out = True                          # use the exported per-conv out_scales
        assert eng.out_scales, "no out_scales in npz; re-run export_model.py"
    eng.reset()
    logits = eng.forward(x, adapt=False)[0].astype(np.float32)

    flat = x.reshape(-1)
    lines = [f"// {args.corruption} sev{args.severity} idx{args.index}, true label {int(label)}",
             "// normalized CHW float sample + golden true-integer-engine logits (adapt off)",
             "#pragma once", ""]
    lines.append(f"static const float test_image[{flat.size}] = {{"
                 + ",".join(f"{v:.6f}f" for v in flat) + "};")
    lines.append(f"static const float golden_logits[10] = {{"
                 + ",".join(f"{v:.6f}f" for v in logits) + "};")
    lines.append(f"static const int test_label = {int(label)};")
    lines.append("")
    with open(OUT, "w") as f:
        f.write("\n".join(lines))
    print(f"pred {int(logits.argmax())} | true {int(label)} | logits {np.round(logits,2)}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
