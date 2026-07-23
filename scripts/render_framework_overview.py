"""Render the SIMPLE system-level framework overview (PNG).

Four layers, top to bottom: what data is considered -> one shared ingestion/representation
-> parallel prediction heads (mental-health + glucose) -> the LLM that explains (never
predicts). Honest about how glucose relates to the mental-health heads. Committed to
outputs/_r2/framework_overview.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _box(ax, x, y, w, h, text, face, *, bold=False, fs=10):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.3, edgecolor="#334155", facecolor=face, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal", color="#0b1020", zorder=3)


def _band(ax, y, label):
    ax.text(0.1, y, label, ha="left", va="center", fontsize=9, style="italic", color="#64748b")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/_r2/framework_overview.png"))
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12.5, 8.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 10.6); ax.axis("off")
    blue, green, violet, amber, grey = "#dbeafe", "#dcfce7", "#ede9fe", "#fef3c7", "#e2e8f0"

    ax.text(6, 10.25, "Multimodal health-intelligence framework — one system, many signals",
            ha="center", fontsize=13, fontweight="bold", color="#0b1020")

    # 1 — DATA CONSIDERED
    _band(ax, 9.2, "1 · Data")
    for i, (t, face) in enumerate([
        ("BCI / EEG\nGalea · EMOTIV", blue),
        ("Wearables\nHR · EDA · PPG", blue),
        ("EHR + notes\nlabs · text", blue),
        ("CGM + meals\nglucose · carbs", green),
    ]):
        _box(ax, 1.5 + i * 2.62, 8.7, 2.4, 1.0, t, face)

    # 2 — SHARED INGESTION / REPRESENTATION
    _band(ax, 7.4, "2 · Ingest")
    _box(ax, 1.5, 6.9, 10.0, 1.0,
         "Canonical event schema  →  per-modality encoders\n(real LaBraM for EEG · CGM-history features · ClinicalBERT for notes)",
         grey)

    # 3 — PARALLEL PREDICTION HEADS
    _band(ax, 5.4, "3 · Heads")
    heads = [
        ("Stress", violet), ("Anxiety", violet), ("Depression", violet),
        ("Cognitive\nworkload", violet), ("Glucose\nforecast", green),
    ]
    for i, (t, face) in enumerate(heads):
        _box(ax, 1.5 + i * 2.02, 4.9, 1.9, 1.0, t, face, bold=True, fs=10)

    # 4 — LLM EXPLANATION LAYER
    _band(ax, 3.2, "4 · Explain")
    _box(ax, 1.5, 2.7, 10.0, 1.0,
         "LLM explains & grounds each prediction  —  NEVER predicts\n"
         "abstains on missing data · every number traces to the model",
         amber)

    # arrows between bands
    for cx in (2.7, 5.3, 7.9, 10.5):
        ax.annotate("", xy=(cx, 6.95), xytext=(cx, 8.65),
                    arrowprops=dict(arrowstyle="-|>", color="#475569", lw=1.3))
    ax.annotate("", xy=(6, 6.0), xytext=(6, 6.85),
                arrowprops=dict(arrowstyle="-|>", color="#475569", lw=1.6))
    for i in range(5):
        cx = 1.5 + i * 2.02 + 0.95
        ax.annotate("", xy=(cx, 3.75), xytext=(cx, 4.85),
                    arrowprops=dict(arrowstyle="-|>", color="#475569", lw=1.1))

    # honest note
    ax.text(6, 1.6,
            "Honest scope: the heads share ONE framework, but glucose is forecast from CGM history + meals —\n"
            "no open dataset records EEG + CGM on the same person, so 'glucose from stress/mood' is a "
            "scoped goal, not a validated claim.",
            ha="center", va="center", fontsize=8.8, color="#b45309",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fffbeb", edgecolor="#f59e0b"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
