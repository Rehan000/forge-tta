"""Generate the paper's figures from the result JSONs.

  fig_teaser.pdf      : Fig. 1 -- the method in one view (deploy -> recalibrate -> measure)
                        also written as fig_teaser.png for the README
  fig_gap.pdf         : Fig. 2 -- the adaptation gap (BN-preserved vs folded)
  fig_activations.pdf : Fig. 3 -- the mechanism on real measured activations
  fig_selective.pdf   : Fig. 4 -- recovery vs #layers adapted, held-out vs oracle
  fig_bscurve.pdf     : Fig. 5 -- recovery vs batch size, fixed vs window-matched momentum

Run from the repo root:  python paper/make_figures.py
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight",
                     # embed TrueType (Type-42), never Type-3 -- required by IEEE/CVF checkers
                     "pdf.fonttype": 42, "ps.fonttype": 42,
                     "font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "mathtext.fontset": "stixsans"})


# Shared palette (matches the Figure-1 teaser) and a common axes style, so all three
# figures read as one set.
INK, GRAY, SLATE = "#1f2937", "#9ca3af", "#64748b"
BLUE, GREEN, RED = "#2563eb", "#15803d", "#dc2626"


def _style_axes(ax, xlabel, ylabel):
    """Minimal academic axes: drop the top/right spines, mute the rest, light y-grid."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(SLATE)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=SLATE, labelsize=7, length=3, width=0.8)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK)
    ax.grid(False)
    ax.grid(True, axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, fontsize=8, color=INK)
    ax.set_ylabel(ylabel, fontsize=8, color=INK)


def fig_bscurve():
    d = json.load(open(os.path.join(RES, "phase2_bscurve.json")))
    rows = sorted(d["rows"], key=lambda r: -r["batch_size"])
    bs = [r["batch_size"] for r in rows]
    x = list(range(len(bs)))                      # categorical, large->small
    fixed = [r["fixed"] for r in rows]
    matched = [r["matched"] for r in rows]

    fig, ax = plt.subplots(figsize=(3.3, 2.45))
    ax.plot(x, fixed, "o--", color=RED, lw=1.8, ms=4, label="fixed momentum ($m{=}0.1$)")
    ax.plot(x, matched, "s-", color=BLUE, lw=1.8, ms=4,
            label="window-matched ($m{=}\\mathrm{bs}/640$)")
    ax.set_xticks(x); ax.set_xticklabels(bs)
    ax.annotate("fixed momentum\ncollapses at bs=1", xy=(x[-1], fixed[-1]),
                xytext=(x[-1] - 1.7, fixed[-1] + 3.5), fontsize=7, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.9))
    _style_axes(ax, "batch size (large $\\rightarrow$ single-sample)", "mean recovery (pts)")
    ax.legend(fontsize=7, loc="lower left", frameon=False)
    out = os.path.join(HERE, "fig_bscurve.pdf")
    fig.savefig(out)
    print("wrote", out)


def fig_selective():
    d = json.load(open(os.path.join(RES, "phase2_heldout.json")))
    ho = d["cum_heldout"]
    orc = d["cum_oracle"]
    full = d["full_recovery_test"]
    knee = d["knee_heldout"]
    k = list(range(1, len(ho) + 1))

    fig, ax = plt.subplots(figsize=(3.3, 2.45))
    ax.axhline(full, ls=":", color=SLATE, lw=1.0)
    ax.text(0.8, full + 0.15, "all 21 layers", ha="left", va="bottom", fontsize=6.8, color=SLATE)
    ax.plot(k, orc, "-", color=GREEN, lw=1.8, label="oracle selection")
    ax.plot(k, ho, "o-", color=BLUE, lw=1.8, ms=3.5, label="held-out selection")
    ax.axvline(knee, ls="--", color=RED, lw=1.0)
    pct = round(100 * ho[knee - 1] / full)
    ax.annotate(f"{knee} layers\n$\\to$ {pct}%", xy=(knee, ho[knee - 1]),
                xytext=(knee + 2.5, ho[knee - 1] - 3.0), ha="center", va="center",
                fontsize=7, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.9))
    _style_axes(ax, "# layers adapted (importance order)", "recovery on unseen test (pts)")
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    out = os.path.join(HERE, "fig_selective.pdf")
    fig.savefig(out)
    print("wrote", out)


def fig_teaser():
    """Figure 1: the FORGE story across three stages -- deploy (lose adaptation),
    recalibrate per channel (forward-only), run on device. The middle stage carries a real
    mechanism inset: a folded channel's activation distribution drifts under corruption and
    FORGE re-centers it onto the clean training target. Canvas aspect (ylim/h == xlim/w) keeps
    the badges circular; labels are kept >=6.2pt for the ~6pt print floor at column width."""
    import numpy as np
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

    INK, GRAY, SLATE = "#1f2937", "#9ca3af", "#64748b"
    BLUE, GREEN, RED = "#2563eb", "#15803d", "#dc2626"
    PANEL = "#f4f5f7"
    DOT = "\u00b7"
    BETA, GAMMA = "\u03b2", "\u03b3"

    fig, ax = plt.subplots(figsize=(7.0, 2.73))
    ax.set_xlim(0, 100); ax.set_ylim(0, 39); ax.axis("off")
    TITLE_Y = 34.0

    def panel(x, w):
        ax.add_patch(FancyBboxPatch((x, 2.0), w, 35.0, boxstyle="round,pad=0,rounding_size=1.3",
                                    linewidth=0, facecolor=PANEL, zorder=0))

    def header(cx, num, title, y=TITLE_Y):
        """Numbered badge + title, the (badge+title) group centered on cx; title width is
        measured from the renderer so the centering is exact."""
        t = ax.text(0, y, title, ha="left", va="center", color=INK, fontsize=8.2,
                    weight="bold", zorder=5)
        fig.canvas.draw()
        bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
        (x0, _), (x1, _) = ax.transData.inverted().transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        tw, d, gap = x1 - x0, 2.9, 1.4
        gl = cx - (d + gap + tw) / 2.0
        ax.add_patch(Circle((gl + d / 2, y), 1.45, facecolor=INK, edgecolor="none", zorder=4))
        ax.text(gl + d / 2, y + 0.05, str(num), ha="center", va="center", color="white",
                fontsize=7.5, weight="bold", zorder=5)
        t.set_position((gl + d + gap, y))

    def box(x, y, w, h, text, ec, fs=6.5, tc=None, fc="white"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.7",
                                    linewidth=1.2, edgecolor=ec, facecolor=fc, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=tc or INK, zorder=3, linespacing=1.45)

    def arrow(x1, y1, x2, y2, color=SLATE, lw=1.4, rad=0.0, ms=10):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms,
                                     lw=lw, color=color, connectionstyle=f"arc3,rad={rad}", zorder=3))

    # ===== Stage 1: deploy to the MCU  (panel [2,30], centre 16) =====
    panel(2, 28); header(16, 1, "Deploy to MCU")
    box(3.5, 25.0, 25, 6.0, "Trained CNN\nConv + BatchNorm (fp32)", SLATE)   # top 31.0 (gap 3.0)
    arrow(16, 25.0, 16, 18.5, color=INK, lw=1.3)
    ax.text(17.8, 21.7, "fold BN +\nint8 quantize", ha="left", va="center", fontsize=6.2, color=INK)
    box(3.5, 12.5, 25, 6.0, "Integer-only model\nBN fused into Conv", SLATE)
    ax.text(16, 9.0, "adaptation no longer works", ha="center", va="center", fontsize=6.2, color=RED)
    ax.text(16, 6.0, "0 points recovered", ha="center", va="center", fontsize=6.7,
            color=RED, weight="bold")

    # ===== Stage 2: per-channel recalibration  (panel [33,67], centre 50) =====
    panel(33, 34); header(50, 2, "Recalibrate per channel")
    bx0, bx1, base, H = 37.5, 62.5, 14.0, 14.0
    u = np.linspace(-3.6, 3.6, 240)
    sx = (u + 3.6) / 7.2 * (bx1 - bx0) + bx0
    g = lambda mu, sd: np.exp(-((u - mu) ** 2) / (2 * sd * sd))
    clean, shift = g(0.0, 1.0), g(1.55, 1.3) * 0.92
    ax.plot([bx0 - 1.5, bx1 + 1.5], [base, base], color=INK, lw=0.8, zorder=2)     # axis
    ax.fill_between(sx, base, base + clean * H, color=GRAY, alpha=0.30, zorder=2)  # clean target
    ax.plot(sx, base + clean * H, color=GRAY, lw=1.0, zorder=2)
    ax.plot(sx, base + shift * H, color=RED, lw=1.4, ls="--", zorder=3)            # corrupted
    ax.plot(sx, base + clean * H, color=BLUE, lw=1.8, zorder=4)                    # recalibrated
    arrow(55.0, base + 11.0, 50.8, base + 13.0, color=BLUE, lw=1.3, rad=0.4, ms=9)
    # compact legend (top-left, gap below title, clear of the rising left shoulder)
    lx, ly = 35.0, [30.8, 29.2, 27.6]
    ax.add_patch(Rectangle((lx, ly[0] - 0.45), 2.1, 0.9, facecolor=GRAY, alpha=0.45,
                           edgecolor=GRAY, lw=0.8, zorder=3))
    ax.plot([lx, lx + 2.1], [ly[1], ly[1]], color=RED, lw=1.4, ls="--", zorder=3)
    ax.plot([lx, lx + 2.1], [ly[2], ly[2]], color=BLUE, lw=1.8, zorder=3)
    ax.text(lx + 2.7, ly[0], "clean target", ha="left", va="center", fontsize=6.2, color=INK)
    ax.text(lx + 2.7, ly[1], "corrupted", ha="left", va="center", fontsize=6.2, color=RED)
    ax.text(lx + 2.7, ly[2], "re-centered", ha="left", va="center", fontsize=6.2, color=BLUE,
            weight="bold")
    ax.text(50, base - 1.9, "per-channel pre-activation", ha="center", va="center",
            fontsize=6.2, color=SLATE, style="italic")
    ax.text(50, base - 4.5, "match each channel to its clean (" + BETA + ", |" + GAMMA + "|)",
            ha="center", va="center", fontsize=6.3, color=INK)
    ax.text(50, base - 6.9, "forward-only " + DOT + " no gradients", ha="center", va="center",
            fontsize=6.2, color=SLATE)

    # ===== Stage 3: on-device result  (panel [70,98], centre 84) =====
    panel(70, 28); header(84, 3, "Run on ESP32-S3")
    ax.text(73, 30.8, "accuracy recovered (pts)", ha="left", va="center", fontsize=6.2, color=INK)
    ax.add_patch(Rectangle((73, 27.0), 0.7, 1.8, color=GRAY, zorder=2))            # deployed ~0
    ax.text(74.4, 27.9, "deployed  +0.0", ha="left", va="center", fontsize=6.2, color=SLATE)
    ax.add_patch(Rectangle((73, 22.8), 16.5, 1.8, color=GREEN, zorder=2))          # FORGE +21
    ax.text(90.5, 23.7, "+20.9", ha="left", va="center", fontsize=6.7, color=GREEN, weight="bold")
    ax.text(73, 20.6, "FORGE", ha="left", va="center", fontsize=6.2, color=GREEN, weight="bold")
    ax.plot([73, 95], [18.2, 18.2], color="#d1d5db", lw=0.8, zorder=1)
    ax.text(73, 15.2, "energy", ha="left", va="center", fontsize=6.4, color=INK)
    ax.text(82.5, 15.2, "+8.3 mJ  (6.8%)", ha="left", va="center", fontsize=6.2, color=SLATE)
    ax.text(73, 11.6, "latency", ha="left", va="center", fontsize=6.4, color=INK)
    ax.text(82.5, 11.6, "+21.9 ms  (7.6%)", ha="left", va="center", fontsize=6.2, color=SLATE)
    ax.text(84, 7.6, "measured, Nordic PPK2", ha="center", va="center", fontsize=6.2, color=SLATE,
            style="italic")

    # connectors between stages
    arrow(30.3, 20.0, 32.7, 20.0, color=SLATE, lw=1.8, ms=12)
    arrow(67.3, 20.0, 69.7, 20.0, color=SLATE, lw=1.8, ms=12)

    out = os.path.join(HERE, "fig_teaser.pdf")
    fig.savefig(out)
    print("wrote", out)
    # PNG twin for the README (GitHub cannot render PDFs inline). White background so
    # it stays legible against GitHub's dark theme.
    png = os.path.join(HERE, "fig_teaser.png")
    fig.savefig(png, dpi=220, facecolor="white")
    print("wrote", png)


def fig_gap():
    """The adaptation gap in one view: BN-adapt recovers +20.1 only on the BN-preserved
    model (not deployable in integer-only form); folding it for deployment drops recovery to
    +0.0; FORGE restores +20.9 on the same deployed folded model. Data: gap study + baselines."""
    import numpy as np
    from matplotlib.patches import FancyArrowPatch
    labels = ["source\n(no adapt)", "BN-adapt\nBN preserved", "BN-adapt\nBN folded", "FORGE\nBN folded"]
    acc  = [50.4, 70.5, 50.4, 71.3]
    rec  = ["", "+20.1", "+0.0", "+20.9"]
    cols = [GRAY, SLATE, RED, GREEN]
    src = 50.4

    fig, ax = plt.subplots(figsize=(3.4, 2.65))
    x = np.arange(4)
    ax.bar(x, acc, width=0.66, color=cols, zorder=3, edgecolor="white", linewidth=0.6)
    ax.axhline(src, ls=(0, (2, 2)), color=SLATE, lw=0.9, zorder=1)
    ax.text(0, 53.2, "corrupted baseline", ha="center", va="bottom", fontsize=5.6, color=SLATE)
    for i in range(4):
        if rec[i]:
            ax.text(i, acc[i] + 0.8, rec[i], ha="center", va="bottom", fontsize=9,
                    weight="bold", color=cols[i])
    # the gap: same method (BN-adapt) loses all its recovery once folded for deployment
    ax.add_patch(FancyArrowPatch((1.45, 68.0), (1.85, 53.0), arrowstyle="-|>", mutation_scale=10,
                                 lw=1.3, color=RED, connectionstyle="arc3,rad=-0.28", zorder=4))
    ax.text(1.92, 61.0, "folding for\ndeployment", ha="left", va="center", fontsize=5.8,
            color=RED, linespacing=1.1)
    ax.set_ylim(40, 80); ax.set_xlim(-0.65, 3.55)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.8)
    ax.set_ylabel("mean accuracy, CIFAR-10-C (%)", fontsize=7.5, color=INK)
    ax.tick_params(axis="y", labelsize=7.5, colors=SLATE, length=3)
    ax.tick_params(axis="x", length=0, labelcolor=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(SLATE); ax.spines[s].set_linewidth(0.8)
    ax.set_axisbelow(True)
    ax.grid(False)
    ax.grid(True, axis="y", alpha=0.25, lw=0.6)
    out = os.path.join(HERE, "fig_gap.pdf")
    fig.savefig(out)
    print("wrote", out)


def _kde(v, grid, bw=None):
    """Lightweight Gaussian KDE (numpy-only), Scott's-rule bandwidth by default."""
    import numpy as np
    v = np.asarray(v, dtype=np.float64)
    if bw is None:
        bw = v.std() * v.size ** (-0.2) + 1e-9
    z = (grid[:, None] - v[None, :]) / bw
    return np.exp(-0.5 * z * z).sum(1) / (v.size * bw * np.sqrt(2 * np.pi))


def fig_activations():
    """Evidence the mechanism works on real activations: the measured per-channel
    distribution of one conv channel in the deployed int8 model -- clean (source),
    corrupted (shifted off the source), and FORGE-recalibrated (snapped back onto the
    source). Data: deploy/dump_activations.py -> paper/activations.npz.
    This is Figure 3 of the published paper (Sec. 3.3)."""
    import numpy as np
    src = os.path.join(HERE, "activations.npz")
    if not os.path.exists(src):
        print("skip fig_activations (run deploy/dump_activations.py first)")
        return
    d = np.load(src)
    vc, vk, vf = d["clean"], d["corrupted"], d["forge"]
    lo = float(min(vc.min(), vk.min(), vf.min()))
    hi = float(max(vc.max(), vk.max(), vf.max()))
    pad = 0.08 * (hi - lo)
    grid = np.linspace(lo - pad, hi + pad, 400)

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    series = [(vc, SLATE, "clean (source)"),
              (vk, RED, "corrupted"),
              (vf, GREEN, "FORGE (recalibrated)")]
    ymax = 0.0
    for v, c, lab in series:
        y = _kde(v, grid)
        ymax = max(ymax, y.max())
        ax.fill_between(grid, y, color=c, alpha=0.16, zorder=2)
        ax.plot(grid, y, color=c, lw=1.7, zorder=3, label=lab)

    # the shift: corruption displaces the channel off the source mean; FORGE restores it
    mc, mk = float(vc.mean()), float(vk.mean())
    ax.annotate("", xy=(mk, ymax * 1.04), xytext=(mc, ymax * 1.04),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9))
    ax.text((mc + mk) / 2, ymax * 1.10, "corruption shift", ha="center", va="bottom",
            fontsize=6.0, color=INK)

    ax.set_ylim(0, ymax * 1.22)
    ax.set_xlabel("activation value (one conv channel, deployed int8)", fontsize=7.5, color=INK)
    ax.set_ylabel("density", fontsize=7.5, color=INK)
    ax.tick_params(labelsize=7, colors=SLATE, length=3)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(SLATE); ax.spines[s].set_linewidth(0.8)
    ax.set_axisbelow(True)
    ax.grid(False)
    ax.grid(True, axis="y", alpha=0.25, lw=0.6)
    ax.legend(fontsize=6.4, frameon=False, loc="upper left", handlelength=1.3,
              labelcolor=INK, borderaxespad=0.2)
    out = os.path.join(HERE, "fig_activations.pdf")
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    fig_teaser()
    fig_bscurve()
    fig_selective()
    fig_gap()
    fig_activations()
