"""Render the NeuroGlycemicNet architecture as a labeled block diagram (PNG).

A faithful schematic of neural_model.py::NeuroGlycemicNet.forward — availability-aware
mixture-of-experts + meal response kernel + Gaussian mixture, residual over persistence.
Committed to outputs/_r2/model_architecture.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _box(ax, xy, w, h, text, face, fg="#0b1020"):
    from matplotlib.patches import FancyBboxPatch
    x, y = xy
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.2, edgecolor="#334155", facecolor=face, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=9, color=fg, zorder=3, wrap=True)


def _arrow(ax, start, end, color="#475569", label=None):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.4), zorder=1)
    if label:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx, my + 0.06, label, ha="center", va="bottom", fontsize=7.5, color=color)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/_r2/model_architecture.png"))
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12.5, 8.0))
    ax.set_xlim(0, 12); ax.set_ylim(0, 10.5); ax.axis("off")

    blue, green, amber, grey, violet = "#dbeafe", "#dcfce7", "#fef3c7", "#e2e8f0", "#ede9fe"

    _box(ax, (0.4, 9.2), 5.0, 0.9,
         "Inputs per modality (cgm · events)\nstandardized features + feature_masks", blue)
    _box(ax, (6.6, 9.2), 5.0, 0.9,
         "availability · quality · staleness\n· clock_uncertainty", grey)

    _box(ax, (0.4, 7.9), 5.0, 0.7, "modality dropout (train only)", grey)
    _box(ax, (0.4, 6.7), 5.0, 0.8, "ModalityEncoder per modality\nMLP → embedding (obs-mask aware)", blue)
    _box(ax, (0.4, 5.5), 5.0, 0.7, "CrossModalContextLayer × N  (N=0 shipped)", blue)

    _box(ax, (0.4, 4.0), 5.0, 0.9,
         "per-modality Gaussian experts\nHorizonFiLM → (μ, σ) per horizon", violet)

    _box(ax, (6.6, 4.0), 5.0, 0.9,
         "ResponseKernelHead\nsign-constrained carb→glucose\ndelayed response (+Δ)", amber)

    _box(ax, (6.6, 5.7), 5.0, 1.0,
         "LearnedMaskedFusion\navailability · quality · staleness\n→ weights + ABSTAIN flag", green)

    _box(ax, (2.2, 2.5), 5.0, 0.9,
         "Gaussian mixture\nμ = Σ wₖμₖ ,  σ² from mixture moments", violet)

    _box(ax, (2.2, 1.2), 5.0, 0.8,
         "residual over persistence (shrunk toward last CGM)", green)

    _box(ax, (2.2, 0.1), 5.0, 0.8,
         "forecast 30/60/90/120 min — μ + 95% interval\n→ split-conformal calibration", blue)

    _box(ax, (8.3, 2.5), 3.3, 0.9, "auxiliary heads\nhypo / hyper prob.", grey)

    # arrows
    _arrow(ax, (2.9, 9.2), (2.9, 8.6))
    _arrow(ax, (2.9, 7.9), (2.9, 7.5))
    _arrow(ax, (2.9, 6.7), (2.9, 6.2))
    _arrow(ax, (2.9, 5.5), (2.9, 4.9))
    _arrow(ax, (6.6, 4.45), (5.4, 4.45), label="+Δ")
    _arrow(ax, (9.1, 9.2), (9.1, 6.7), label="gates")
    _arrow(ax, (2.9, 4.0), (4.2, 3.4))
    _arrow(ax, (9.1, 5.7), (6.0, 3.4), label="weights")
    _arrow(ax, (4.7, 2.5), (4.7, 2.0))
    _arrow(ax, (4.7, 1.2), (4.7, 0.9))
    _arrow(ax, (7.2, 2.95), (8.3, 2.95))

    ax.text(6, 10.25, "NeuroGlycemicNet — availability-aware mixture-of-experts glucose forecaster",
            ha="center", fontsize=12, fontweight="bold", color="#0b1020")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
