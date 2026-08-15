"""De-risk ESP-NN integration on the host: does per-conv output quantization (which
is what esp_nn's integer requant does) preserve accuracy?

esp_nn_conv_s8 produces an int8 output requantized to a per-tensor out_scale. That is
numerically a fake-quant of the conv output. So if we add that output quantization to
the golden int8 reference and accuracy holds (clean + corruption, no-adapt + adapt),
the full ESP-NN port is accuracy-safe and the rest is mechanical (layout + requant
params). If accuracy drops a lot, we learn that BEFORE writing any firmware.

Usage:
    python deploy/espnn_validate.py --n 1000
"""
import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from data import clean_loaders, corruption_loader   # noqa: E402
from int8_reference import IntResNet20               # noqa: E402

ART = os.path.join(os.path.dirname(__file__), "artifacts", "resnet20_int8.npz")


def acc(eng, loader, n, adapt):
    eng.reset()
    correct = total = 0
    for x, y in loader:
        for i in range(x.shape[0]):
            pred = eng.forward(x[i:i + 1].numpy(), adapt=adapt).argmax(1)[0]
            correct += int(pred == y[i].item())
            total += 1
            if total >= n:
                return 100.0 * correct / total
    return 100.0 * correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--corruptions", nargs="*", default=["gaussian_noise", "contrast", "defocus_blur"])
    args = ap.parse_args()

    eng = IntResNet20(ART)
    _, clean = clean_loaders("data", 128)

    # calibrate conv output scales on a few hundred clean images
    cal_imgs = []
    for x, _ in clean:
        for i in range(x.shape[0]):
            cal_imgs.append(x[i:i + 1].numpy())
        if len(cal_imgs) >= 256:
            break
    eng.calibrate_out_scales(cal_imgs[:256])
    print(f"calibrated {len(eng.out_scales)} conv output scales\n")

    # clean accuracy: exact reference vs esp_nn-scheme (output-quantized)
    eng.quant_out = False
    c_exact = acc(eng, clean, args.n, adapt=False)
    eng.quant_out = True
    c_espnn = acc(eng, clean, args.n, adapt=False)
    print(f"[clean] exact int8 {c_exact:.2f}%  |  esp_nn-scheme {c_espnn:.2f}%  (drop {c_exact-c_espnn:+.2f})\n")

    print(f"{'corruption':<16}{'exact src':>10}{'espnn src':>11}{'exact rec':>11}{'espnn rec':>11}")
    for c in args.corruptions:
        eng.quant_out = False
        s_ex = acc(eng, corruption_loader("data", c, 5, 1, max_samples=args.n), args.n, adapt=False)
        r_ex = acc(eng, corruption_loader("data", c, 5, 1, max_samples=args.n), args.n, adapt=True)
        eng.quant_out = True
        s_en = acc(eng, corruption_loader("data", c, 5, 1, max_samples=args.n), args.n, adapt=False)
        r_en = acc(eng, corruption_loader("data", c, 5, 1, max_samples=args.n), args.n, adapt=True)
        print(f"{c:<16}{s_ex:>9.2f}%{s_en:>10.2f}%{r_ex:>10.2f}%{r_en:>10.2f}%")

    print("\n-> if esp_nn-scheme columns track the exact columns, ESP-NN integration is "
          "accuracy-safe; proceed to firmware port.")


if __name__ == "__main__":
    main()
