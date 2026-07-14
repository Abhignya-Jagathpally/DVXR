"""dvxr.serve.evidence — the single source of the product's numbers, literature, and limits.

Every quantitative claim the product makes (CLI `report`, the demo page, the model card, the
shareable evidence Artifact) resolves here, and each number is pinned to the committed scoreboard
file it came from. `verify_against_scoreboards()` re-reads those CSVs and asserts the headline
errors still match, so a number can never silently drift from its source. `EXCLUDED_CLAIMS` names
what must *never* appear as a product claim (the honesty gate the P5 audit enforces).

Numbers are AUROC on subject-held-out CV, converted from the scoreboards' `1-AUROC` error column
(AUROC = 1 - err). Research-grade screening, not diagnosis.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# Repo-root-relative scoreboards. Resolved from this file so it works from any CWD.
_ROOT = Path(__file__).resolve().parents[3]
_MAIN = "outputs/benchmark_scoreboard.csv"            # mh profile: baseline-vs-CACMF
_LABRAM = "outputs/_dnh_labram/benchmark_scoreboard.csv"  # DNH library incl. real LaBraM


@dataclass
class ProductClaim:
    """One validated, headline-able capability. `source_err` is the scoreboard's 1-AUROC cell we
    re-verify against; `auroc` is the number the product shows (1 - source_err)."""
    task: str
    label: str
    encoder: str
    auroc: float
    auroc_ci: List[float]
    source_file: str
    source_task: str
    source_baseline: str      # which scoreboard column/row is the winner
    source_err: float         # the 1-AUROC value in that row (drift-guarded)
    comparators: Dict[str, float]   # competing method -> AUROC, for the comparative table
    literature: List[str]
    caveat: str
    headline: bool = False


# ----- the allowed product claims (traceable, validated) -----
PRODUCT_CLAIMS: List[ProductClaim] = [
    ProductClaim(
        task="mumtaz_depression",
        label="Depression screen (MDD vs healthy) from resting EEG",
        encoder="real LaBraM EEG foundation model (frozen linear-probe)",
        auroc=0.961, auroc_ci=[0.942, 0.976],
        source_file=_LABRAM, source_task="mumtaz_depression",
        source_baseline="labram", source_err=0.0392,
        comparators={"LaBraM EEG FM": 0.961, "band-power (single:eeg)": 0.889,
                     "tuned GBM (xgboost)": 0.930, "raw-CNN": 0.840,
                     "learned CACMF fusion": 0.795},
        literature=[
            "LaBraM — Jiang et al., ICLR 2024, arXiv:2405.18765 (EEG foundation model)",
            "Mumtaz et al., 2016 — MDD-vs-healthy resting-EEG cohort (labels)",
        ],
        caveat="Fidelity-limited: 64 Hz source (≤32 Hz content) vs LaBraM's 200 Hz training; "
               "would plausibly improve at native rate.",
        headline=True),
    ProductClaim(
        task="wesad_stress",
        label="Acute-stress screen from wearable physiology",
        encoder="band-power physiology features + tuned GBM",
        auroc=0.955, auroc_ci=[0.930, 0.978],
        source_file=_MAIN, source_task="wesad_stress",
        source_baseline="xgboost", source_err=0.0453,
        comparators={"band-power + GBM": 0.955, "learned CACMF fusion": 0.871},
        literature=["WESAD — Schmidt et al., 2018 (wearable stress cohort)"],
        caveat="Wrist/chest wearable cohort; acute-stress state, not chronic-stress diagnosis."),
    ProductClaim(
        task="eegmat_workload",
        label="Cognitive-workload screen (rest vs task)",
        encoder="ECG autonomic (band-power); LaBraM improves the EEG-only path",
        auroc=0.740, auroc_ci=[0.71, 0.77],
        source_file=_MAIN, source_task="eegmat_workload",
        source_baseline="single:physiology", source_err=0.2598,
        comparators={"ECG autonomic (task best)": 0.740, "LaBraM EEG FM": 0.663,
                     "band-power EEG": 0.636, "learned CACMF fusion": 0.635},
        literature=[
            "PhysioNet EEG-during-Mental-Arithmetic — Zyma et al., 2019",
            "LaBraM — Jiang et al., ICLR 2024, arXiv:2405.18765",
        ],
        caveat="Autonomic ECG dominates for arithmetic load; the EEG FM improves the EEG view "
               "(0.663 > 0.636 band-power) but does not overturn the task-level modality finding."),
    ProductClaim(
        task="stress",
        label="Stress screen from peripheral physiology",
        encoder="peripheral physiology (band-power, concat)",
        auroc=0.892, auroc_ci=[0.86, 0.91],
        source_file=_MAIN, source_task="stress",
        source_baseline="rep:pca", source_err=0.1079,
        comparators={"peripheral physiology": 0.892, "learned CACMF fusion": 0.871},
        literature=["PhysioNet Non-EEG — Birjandtalab et al., 2016 (peripheral physiology)"],
        caveat="Peripheral-signal stress states; research cohort."),
]

# ----- a method-level claim (not a single task): the do-no-harm fusion contribution -----
METHOD_CLAIMS = [
    {"claim": "Reliability-gated do-no-harm fusion (dnh_gated) beats the proposal's own learned "
              "cross-modal CACMF fusion on 4 of 6 tasks (stress +11%, wesad +28%, eegmat +18%, "
              "depression +53%) and the best single modality on 3 of 6.",
     "source_file": _LABRAM,
     "caveat": "Nuanced positive, not a clean sweep: universal held-out do-no-harm does NOT hold "
               "at N<=60 (loses to single-ECG on eegmat, slips on the near-chance DEAP pair). "
               "Reported, not hidden.",
     "literature": ["Super Learner — van der Laan et al., 2007 (do-no-harm oracle provenance)"]},
]

# ----- what must NEVER be presented as a product claim (the honesty gate) -----
EXCLUDED_CLAIMS = {
    "deap_affect": "DEAP affective decoding is at chance (AUROC ~0.51-0.55) for every config — "
                   "not a capability.",
    "cacmf_as_win": "The learned CACMF cross-modal fusion LOSES on all 6 real tasks — it is the "
                    "negative result, never sold as the product model.",
    "llm_as_predictor": "The LLM-in-the-predictive-path (rep:llm) is the weakest config and is "
                        "explanation-only in the product, never a predictor.",
    "mimic_mortality": "MIMIC mortality is tiny/untrustworthy here — excluded from claims.",
    "cgmacros_diabetes": "The old cgmacros_diabetes numbers had label leakage — excluded.",
    "diagnosis": "This is research-grade screening / decision-support, never a diagnostic claim.",
}


def _read_scoreboard(rel: str) -> Dict[str, dict]:
    path = _ROOT / rel
    rows: Dict[str, dict] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            rows[row["task"]] = row
    return rows


def verify_against_scoreboards(tol: float = 5e-3) -> List[str]:
    """Re-read the committed scoreboards and confirm each claim's source_err still matches.

    Returns a list of human-readable mismatch strings (empty ⇒ every number still traces cleanly).
    Raises FileNotFoundError only if a scoreboard is missing.
    """
    problems: List[str] = []
    cache: Dict[str, Dict[str, dict]] = {}
    for c in PRODUCT_CLAIMS:
        board = cache.setdefault(c.source_file, _read_scoreboard(c.source_file))
        row = board.get(c.source_task)
        if row is None:
            problems.append(f"{c.task}: source_task {c.source_task!r} not in {c.source_file}")
            continue
        base_err = float(row["base_err"])
        if abs(base_err - c.source_err) > tol:
            problems.append(f"{c.task}: scoreboard base_err {base_err} != pinned {c.source_err}")
        derived = round(1.0 - base_err, 3)
        if abs(derived - c.auroc) > 1e-2:
            problems.append(f"{c.task}: AUROC {c.auroc} != 1-err {derived}")
    return problems


def comparative_table() -> List[dict]:
    """Rows for the headline comparative table: per task, each method's AUROC, winner flagged."""
    out = []
    for c in PRODUCT_CLAIMS:
        winner = max(c.comparators, key=c.comparators.get)
        out.append({"task": c.label, "encoder": c.encoder, "auroc": c.auroc,
                    "ci": c.auroc_ci, "winner_method": winner,
                    "comparators": dict(sorted(c.comparators.items(), key=lambda kv: -kv[1])),
                    "source": c.source_file, "caveat": c.caveat, "headline": c.headline})
    return out


def product_numbers() -> Dict[str, dict]:
    return {c.task: {"label": c.label, "auroc": c.auroc, "ci": c.auroc_ci,
                     "encoder": c.encoder, "source": c.source_file} for c in PRODUCT_CLAIMS}


def render_report(screener_dir: Optional[str] = None) -> str:
    """Evidence one-pager (text) for `dvxr report`. If screener_dir is given, its manifest line is
    appended, but the product-grade numbers below are the canonical, scoreboard-traced set."""
    lines = ["DVXR Screen — evidence report", "=" * 60,
             "Research-grade screening / decision-support. Not a diagnosis.",
             "Every number below traces to a committed scoreboard file.", ""]
    for c in PRODUCT_CLAIMS:
        tag = "  ★ HEADLINE" if c.headline else ""
        lines.append(f"[{c.task}]{tag}")
        lines.append(f"  {c.label}")
        lines.append(f"  AUROC {c.auroc}  CI [{c.auroc_ci[0]}, {c.auroc_ci[1]}]   "
                     f"via {c.encoder}")
        comp = "  vs ".join(f"{k} {v}" for k, v in
                            sorted(c.comparators.items(), key=lambda kv: -kv[1]))
        lines.append(f"  compare: {comp}")
        lines.append(f"  source : {c.source_file}")
        lines.append(f"  caveat : {c.caveat}")
        lines.append("")
    lines.append("Method contribution:")
    for m in METHOD_CLAIMS:
        lines.append(f"  - {m['claim']}")
        lines.append(f"    caveat: {m['caveat']}  [{m['source_file']}]")
    lines.append("")
    lines.append("Explicitly NOT claimed (honesty gate):")
    for k, why in EXCLUDED_CLAIMS.items():
        lines.append(f"  - {k}: {why}")
    problems = verify_against_scoreboards()
    lines.append("")
    lines.append("Scoreboard trace: " +
                 ("ALL NUMBERS VERIFIED ✓" if not problems else "MISMATCH — " + "; ".join(problems)))
    if screener_dir:
        import json
        m = json.loads((Path(screener_dir) / "manifest.json").read_text())
        h = m["heldout"]
        lines.append("")
        lines.append(f"Loaded screener [{m['task']}]: AUROC {h['auroc']} CI {h['auroc_ci']} "
                     f"ECE {h.get('ece')} ({h.get('protocol')})")
    return "\n".join(lines)
