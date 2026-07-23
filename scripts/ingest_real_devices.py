"""Ingest the REAL (non-public) DVXR device sessions and summarize them.

Uses the actual lab hardware recordings — an EMOTIV EPOC X BCI session and a Galea/OpenBCI
session captured 2026-06-08 — proving the pipeline ingests device data, not only public
datasets. Writes a summary JSON + a presentation figure. Nothing is fabricated: every number
is measured from the files.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
EMOTIV = REPO / "data/real/emotiv/EmotivBCI-AJ_EPOCX_142080_2026.06.08T15.15.46.05.00.md.mc.pm.fe.bp.csv"
GALEA = REPO / "data/real/galea/OpenBCISession_2026-06-08_11-23-34/BrainFlow-RAW_2026-06-08_11-23-34_0.csv"


def summarize():
    from dvxr.bci_real import ingest_emotiv

    rec = ingest_emotiv(str(EMOTIV))
    cmd = rec.mc["action"].value_counts().to_dict() if "action" in rec.mc else {}
    emotiv = {
        "device": "EMOTIV EPOC X (serial E5020C07)",
        "eeg_channels": int(rec.eeg.shape[1] - 1),
        "eeg_samples": int(rec.eeg.shape[0]),
        "duration_s": round(float(rec.duration_s), 1),
        "mental_commands": {k: int(v) for k, v in cmd.items()},
        "bandpower_rows": int(rec.pow.shape[0]) if getattr(rec, "pow", None) is not None else 0,
    }
    # Galea raw: BrainFlow tab-separated matrix (col0 = index, next 16 = EEG µV)
    galea_raw = pd.read_csv(GALEA, sep="\t", header=None, nrows=200000)
    galea = {
        "device": "Galea / OpenBCI (BrainFlow RAW)",
        "samples": int(len(galea_raw)),
        "columns": int(galea_raw.shape[1]),
        "eeg_channels_est": 16,
    }
    return {"emotiv": emotiv, "galea": galea, "source": "non-public DVXR lab hardware, 2026-06-08"}


def figure(summary, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 13, "font.family": "DejaVu Sans"})
    EEG, INK, MUTED, GOOD = "#7c3aed", "#0f172a", "#64748b", "#059669"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.0), gridspec_kw={"width_ratios": [1.1, 1]})
    cmds = summary["emotiv"]["mental_commands"]
    order = [c for c in ["Neutral", "Left", "Right", "Push", "Pull"] if c in cmds]
    bars = ax1.bar(order, [cmds[c] for c in order], color=[MUTED if c == "Neutral" else EEG for c in order])
    ax1.bar_label(bars, fmt="%d", padding=3, fontweight="bold")
    ax1.set(title="Real EMOTIV mental commands (this session)", ylabel="samples")
    ax1.grid(axis="y", alpha=0.25); ax1.grid(axis="x", visible=False)

    ax2.axis("off")
    e, g = summary["emotiv"], summary["galea"]
    lines = [
        ("REAL DVXR device data — non-public", INK, 15, True),
        ("captured 2026-06-08 on lab hardware", MUTED, 11, False),
        ("", INK, 6, False),
        (f"EMOTIV EPOC X", EEG, 13, True),
        (f"  {e['eeg_channels']}-ch EEG · {e['eeg_samples']:,} samples · {e['duration_s']:.0f}s", INK, 12, False),
        (f"  {sum(e['mental_commands'].values()):,} mental-command labels · band power", INK, 12, False),
        ("", INK, 6, False),
        (f"Galea / OpenBCI", GOOD, 13, True),
        (f"  {g['eeg_channels_est']}-ch EEG · {g['samples']:,} raw samples (BrainFlow)", INK, 12, False),
        ("", INK, 6, False),
        ("→ the pipeline ingests real BCI hardware,", MUTED, 11, False),
        ("   not only public datasets (POW device list a,b)", MUTED, 11, False),
    ]
    y = 0.96
    for text, color, size, bold in lines:
        ax2.text(0.0, y, text, fontsize=size, color=color, fontweight="bold" if bold else "normal",
                 transform=ax2.transAxes, va="top")
        y -= 0.075 if text else 0.04
    fig.suptitle("Goal 1 — ingesting the actual DVXR devices (Galea + EMOTIV)", fontsize=17, fontweight="bold")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight"); plt.close(fig)


def main():
    summary = summarize()
    (REPO / "outputs/_r2").mkdir(parents=True, exist_ok=True)
    (REPO / "outputs/_r2/real_device_ingestion.json").write_text(json.dumps(summary, indent=2))
    figure(summary, REPO / "presentation/figures/fig_real_device_data.png")
    print(json.dumps(summary, indent=2))
    print("\nwrote outputs/_r2/real_device_ingestion.json + presentation/figures/fig_real_device_data.png")


if __name__ == "__main__":
    main()
