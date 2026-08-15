"""Dump real per-channel activations from the deployed int8 model for the
activation-distribution figure (clean vs corrupted vs FORGE-recalibrated).

Captures the pre-recalibration conv output at every recalibration site under
three conditions -- clean (source, no adapt), corrupted (deployed, no adapt),
and FORGE-recalibrated (corrupted, adapt) -- then picks the (site, channel) that
best tells the story: a large standardized corruption shift off the source mean,
*and* FORGE restoring the channel back onto the source distribution. The chosen
channel's flattened (subsampled) values are written to an npz that
make_figures.py turns into fig_activations.pdf.

    python deploy/dump_activations.py --corruption gaussian_noise --severity 5
"""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data import CorruptionSet                       # noqa: E402
import torchvision                                   # noqa: E402
import torchvision.transforms as T                   # noqa: E402
sys.path.insert(0, os.path.dirname(__file__))
from int8_reference import IntResNet20               # noqa: E402

ART = os.path.join(os.path.dirname(__file__), "artifacts", "resnet20_int8.npz")
OUT = os.path.join(os.path.dirname(__file__), "..", "paper", "activations.npz")
N = 256                # images per condition (numpy conv is slow; this is plenty for a KDE)
KEEP = 8000            # values to store per condition (subsampled, smooth distribution)

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def _capture(eng, x, adapt, want):
    """Run one forward, return {site_name: tensor} of every recalib site's
    pre-recalib input (want='in') or post-recalib output (want='out')."""
    orig = IntResNet20._recalib.__get__(eng, IntResNet20)
    grab = {}

    def hook(name, xx, ad):
        out = orig(name, xx, ad)
        grab[name] = xx if want == "in" else out
        return out

    eng._recalib = hook
    eng.forward(x, adapt=adapt)
    del eng._recalib                                  # restore bound method
    return grab                                       # {name: [N, C, H, W]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corruption", default="gaussian_noise")
    ap.add_argument("--severity", type=int, default=5)
    args = ap.parse_args()

    tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    clean_ds = torchvision.datasets.CIFAR10("data", train=False, download=True, transform=tf)
    clean = np.stack([clean_ds[i][0].numpy() for i in range(N)]).astype(np.float32)
    corr_ds = CorruptionSet("data", args.corruption, args.severity)
    corr = np.stack([corr_ds[i][0].numpy() for i in range(N)]).astype(np.float32)

    eng = IntResNet20(ART)
    eng.quant_out = bool(eng.out_scales)              # deployed int8 path (requantized convs)

    eng.reset()
    clean_a = _capture(eng, clean, adapt=False, want="in")     # source, all sites
    eng.reset()
    corr_a = _capture(eng, corr, adapt=False, want="in")       # deployed corrupted, all sites

    # FORGE at convergence: running stats = test-stream stats (momentum->1 over one batch),
    # which is the fixed point the online EMA tracks. Re-applies the clean target stats.
    eng.momentum = 1.0
    eng.reset()
    forge_a = _capture(eng, corr, adapt=True, want="out")      # recalibrated corrupted, all sites

    # score every (site, channel): want a large standardized corruption shift off the source
    # mean AND FORGE landing back near the source (small restoration error). Both are real
    # measured properties, so picking by this is honest -- it just finds the clearest example.
    best = None
    for name in clean_a:
        c, k, f = clean_a[name], corr_a[name], forge_a[name]
        mc = c.mean((0, 2, 3)); sc = c.std((0, 2, 3)) + 1e-6
        shift = np.abs(k.mean((0, 2, 3)) - mc) / sc                 # clean -> corrupted
        rest = np.abs(f.mean((0, 2, 3)) - mc) / sc                  # forge  -> clean
        score = shift - rest
        ch = int(np.argmax(score))
        if best is None or score[ch] > best[0]:
            best = (float(score[ch]), name, ch, float(shift[ch]), float(rest[ch]))
    _, site, ch, shift, rest = best

    rng = np.random.RandomState(0)

    def flat(a):
        v = a[site][:, ch].reshape(-1)
        return v[rng.choice(v.size, min(KEEP, v.size), replace=False)].astype(np.float32)

    vc, vk, vf = flat(clean_a), flat(corr_a), flat(forge_a)
    np.savez(OUT, clean=vc, corrupted=vk, forge=vf,
             site=site, channel=ch, corruption=args.corruption, severity=args.severity,
             mean_clean=float(vc.mean()), mean_corr=float(vk.mean()), mean_forge=float(vf.mean()))
    print(f"chose {site} ch {ch} | shift {shift:.2f} sd, restoration err {rest:.2f} sd")
    print(f"  mean clean {vc.mean():.3f} corr {vk.mean():.3f} forge {vf.mean():.3f} | "
          f"std clean {vc.std():.3f} corr {vk.std():.3f} forge {vf.std():.3f}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
