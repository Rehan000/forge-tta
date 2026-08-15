"""Law-of-Total-Variance decomposition of FORGE's running-variance estimate.

Reviewer LKDB observed that FORGE's EMA update v_bar <- (1-m) v_bar + m * var(x_batch)
converges to E[Var(X|Batch)] but omits the between-batch term Var(E[X|Batch]) of the
law of total variance, Var(X) = E[Var(X|Batch)] + Var(E[X|Batch]). This script measures
the omitted term, as a fraction of the true total variance, across batch sizes, on the
real deployed int8 model -- directly answering (a) how large the bias is (and thus its
effect on the scaling cancellation) and (b) why it explains the small-batch collapse.

For a channel, with per-image sufficient statistics over the H*W spatial pixels
(image mean mu_i and image variance s2_i, equal HW per image), the law-of-total-variance
terms for random batches of size B are exact in expectation -- no Monte-Carlo needed:

    sigma2_total = E_i[mu_i^2 + s2_i] - (E_i[mu_i])^2          # pooled over all pixels
    sigma2_mu    = Var_i(mu_i)                                 # variance of image means
    between(B)   = Var(E[X|Batch]) = sigma2_mu / B * (N-B)/(N-1)   # the OMITTED term
    omitted_frac = between(B) / sigma2_total
    scale_error  = 1/sqrt(1 - omitted_frac) - 1                # error in the 1/std rescale

between(B) ~ 1/B: negligible at large B, dominant at B=1. The rescale divides by an
under-estimated std, so the residual scaling error is ~ omitted_frac/2.

    python deploy/variance_decomp.py --corruption gaussian_noise --severity 5
"""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data import CorruptionSet                       # noqa: E402
import torchvision.transforms as T                   # noqa: E402
sys.path.insert(0, os.path.dirname(__file__))
from int8_reference import IntResNet20               # noqa: E402

ART = os.path.join(os.path.dirname(__file__), "artifacts", "resnet20_int8.npz")
N = 512                                  # images (more than dump: we need a stable Var_i(mu))
BSIZES = [1, 2, 4, 8, 16, 32, 64]
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def _capture_stats(eng, x):
    """Run one forward (no adapt); return {site: (mu[N,C], s2[N,C])} of per-image
    per-channel spatial mean and variance of the pre-recalibration conv output."""
    orig = IntResNet20._recalib.__get__(eng, IntResNet20)
    grab = {}

    def hook(name, xx, ad):
        grab[name] = (xx.mean((2, 3)), xx.var((2, 3)))     # reduce over H,W now -> tiny
        return orig(name, xx, ad)

    eng._recalib = hook
    eng.forward(x, adapt=False)
    del eng._recalib
    return grab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corruption", default="gaussian_noise")
    ap.add_argument("--severity", type=int, default=5)
    args = ap.parse_args()

    tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    corr_ds = CorruptionSet("data", args.corruption, args.severity)
    corr = np.stack([corr_ds[i][0].numpy() for i in range(N)]).astype(np.float32)

    eng = IntResNet20(ART)
    eng.quant_out = bool(eng.out_scales)                   # deployed int8 path
    eng.reset()
    stats = _capture_stats(eng, corr)

    # accumulate omitted fraction and implied scale error per batch size over all channels
    omit = {B: [] for B in BSIZES}
    scale = {B: [] for B in BSIZES}
    nchan = 0
    for name, (mu, s2) in stats.items():                   # mu,s2: [N, C]
        grand_mean = mu.mean(0)                            # [C]
        sigma2_total = (mu ** 2 + s2).mean(0) - grand_mean ** 2     # pooled over all pixels
        sigma2_mu = mu.var(0)                              # Var_i(image mean)
        good = sigma2_total > 1e-9
        nchan += int(good.sum())
        for B in BSIZES:
            between = (sigma2_mu / B * (N - B) / (N - 1))[good]   # OMITTED term, exact expectation
            frac = between / sigma2_total[good]
            omit[B].append(frac)
            scale[B].append(1.0 / np.sqrt(np.clip(1.0 - frac, 1e-6, None)) - 1.0)

    print(f"corruption={args.corruption} sev{args.severity}  N={N} images, "
          f"{nchan} channels over {len(stats)} recalib sites\n")
    print(f"{'batch':>6} | {'omitted Var(E[X|B]) / Var(X)':>30} | {'implied 1/std scale error':>26}")
    print("-" * 70)
    for B in BSIZES:
        f = np.concatenate(omit[B]); s = np.concatenate(scale[B])
        print(f"{B:>6} | mean {100*f.mean():6.2f}%   median {100*np.median(f):6.2f}%"
              f"   | mean {100*s.mean():6.2f}%   p90 {100*np.percentile(s,90):6.2f}%")


if __name__ == "__main__":
    main()
