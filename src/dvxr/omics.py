from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .schemas import validate_events

# ---------------------------------------------------------------------------
# Gene symbols used in synthetic data
# ---------------------------------------------------------------------------
_GENE_SYMBOLS = [
    "TP53", "BRCA1", "EGFR", "MYC", "KRAS", "PTEN", "RB1", "APC",
    "CDH1", "VHL", "MLH1", "CDKN2A", "SMAD4", "STK11", "BRAF",
    "NRAS", "PIK3CA", "IDH1", "IDH2", "FLT3", "NPM1", "RUNX1",
    "CEBPA", "DNMT3A", "TET2", "ASXL1", "EZH2", "SF3B1", "SRSF2",
    "U2AF1", "ZRSR2", "SETBP1", "CALR", "JAK2", "MPL", "CSF3R",
    "CBL", "NF1", "WT1", "GATA2", "IKZF1", "PAX5", "NOTCH1", "FBXW7",
    "PHF6", "RPL5", "RPL10", "RPS14", "KDM6A", "KMT2A",
]

# Unit assignments by omic type
_OMIC_UNITS = {
    "gene": "expr",
    "protein": "abundance",
    "metab": "conc",
}


def _infer_unit(channel: str) -> str:
    """Infer unit from channel prefix (gene_/protein_/metab_)."""
    for prefix, unit in _OMIC_UNITS.items():
        if channel.startswith(f"{prefix}_"):
            return unit
    return "value"


def _row(
    subject_id: str,
    session_id: str,
    timestamp_utc,
    channel: str,
    value: float,
    unit: str,
    label_value: str,
) -> dict:
    return {
        "subject_id": subject_id,
        "session_id": session_id,
        "timestamp_utc": timestamp_utc,
        "source_system": "synthetic_omics",
        "device": "omics_panel",
        "modality": "omics",
        "channel": channel,
        "value": float(value),
        "unit": unit,
        "sampling_rate_hz": 0.0,
        "quality_flag": "ok",
        "label_name": "omics_risk",
        "label_value": label_value,
    }


def generate_omics_like_table(
    output_csv: str | Path,
    subjects: int = 8,
    n_genes: int = 50,
    n_proteins: int = 30,
    n_metabolites: int = 20,
    seed: int = 23,
) -> pd.DataFrame:
    """Synthesize a multi-omics panel and return validated canonical events.

    Each (subject, omic-feature) pair becomes one collection event.  A latent
    risk signal shifts ~30 % of features upward for high-risk subjects.

    Parameters
    ----------
    output_csv:
        Destination CSV path (parent directories are created as needed).
    subjects:
        Number of subjects to generate.
    n_genes:
        Number of gene-expression features per subject.
    n_proteins:
        Number of protein-abundance features per subject.
    n_metabolites:
        Number of metabolite-concentration features per subject.
    seed:
        Random-number-generator seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Validated canonical events frame.
    """
    rng = np.random.default_rng(seed)
    base_date = pd.Timestamp("2026-01-01T08:00:00Z")

    # Clamp n_genes to available gene symbols
    n_genes = min(n_genes, len(_GENE_SYMBOLS))

    # Build feature names
    gene_channels = [f"gene_{_GENE_SYMBOLS[i]}" for i in range(n_genes)]
    protein_channels = [f"protein_P{i + 1:03d}" for i in range(n_proteins)]
    metab_channels = [f"metab_M{i + 1:03d}" for i in range(n_metabolites)]
    all_channels = gene_channels + protein_channels + metab_channels

    # Assign labels: alternate high/low, then shuffle to avoid simple ordering
    labels_base = ["high_risk" if i % 2 == 0 else "low_risk" for i in range(subjects)]
    label_order = rng.permutation(subjects)
    labels = [labels_base[idx % 2] for idx in label_order]

    # Which features carry the latent risk signal (~30 %)
    n_signal = max(1, int(0.30 * len(all_channels)))
    signal_feature_idx = rng.choice(len(all_channels), size=n_signal, replace=False)
    signal_set = set(int(i) for i in signal_feature_idx)

    rows: list[dict] = []
    for subj_idx in range(subjects):
        subject_id = f"OmicsS{subj_idx + 1:02d}"
        session_id = f"omics_session_{subj_idx + 1:02d}"
        label_value = labels[subj_idx]
        risk = 1.0 if label_value == "high_risk" else 0.0
        timestamp = base_date + pd.Timedelta(hours=subj_idx * 2)

        for feat_idx, channel in enumerate(all_channels):
            unit = _infer_unit(channel)
            base_val = rng.normal(loc=1.0, scale=0.3)
            signal_shift = 0.5 * risk if feat_idx in signal_set else 0.0
            value = base_val + signal_shift
            rows.append(_row(subject_id, session_id, timestamp, channel, value, unit, label_value))

    events = validate_events(pd.DataFrame(rows))
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(output_csv, index=False)
    return events


def load_omics_table(
    path: str | Path,
    id_col: str = "subject_id",
    label_col: str | None = None,
) -> pd.DataFrame:
    """Ingest a WIDE multi-omics CSV (rows = subjects, columns = omic features).

    Column name prefixes determine omic type and unit:
    - ``gene_*``    → unit "expr"
    - ``protein_*`` → unit "abundance"
    - ``metab_*``   → unit "conc"

    Other non-id / non-label columns are treated as generic omics features
    with unit "value".

    Parameters
    ----------
    path:
        Path to wide-format CSV file.
    id_col:
        Name of the subject-identifier column (default ``"subject_id"``).
    label_col:
        Optional column name carrying per-subject labels that become
        ``label_value`` in the canonical frame.  When ``None`` no label is set.

    Returns
    -------
    pd.DataFrame
        Validated canonical events frame.
    """
    path = Path(path)
    wide = pd.read_csv(path)

    if id_col not in wide.columns:
        raise ValueError(f"id_col '{id_col}' not found in {path.name}; available: {list(wide.columns[:10])}")

    skip_cols = {id_col}
    if label_col is not None:
        if label_col not in wide.columns:
            raise ValueError(f"label_col '{label_col}' not found in {path.name}")
        skip_cols.add(label_col)

    feature_cols = [c for c in wide.columns if c not in skip_cols]
    base_date = pd.Timestamp("2026-01-01T08:00:00Z")

    rows: list[dict] = []
    for row_idx, record in enumerate(wide.itertuples(index=False)):
        subject_id = str(getattr(record, id_col))
        label_value = str(getattr(record, label_col)) if label_col is not None else ""
        timestamp = base_date + pd.Timedelta(hours=row_idx * 2)

        for col in feature_cols:
            raw = getattr(record, col)
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue  # skip non-numeric columns silently

            unit = _infer_unit(col)
            rows.append(
                {
                    "subject_id": subject_id,
                    "session_id": f"omics_session_{row_idx + 1:02d}",
                    "timestamp_utc": timestamp,
                    "source_system": "omics_import",
                    "device": "omics_panel",
                    "modality": "omics",
                    "channel": col,
                    "value": value,
                    "unit": unit,
                    "sampling_rate_hz": 0.0,
                    "quality_flag": "ok",
                    "label_name": "omics_risk" if label_col is not None else "",
                    "label_value": label_value,
                }
            )

    return validate_events(pd.DataFrame(rows))


def build_omics_features(events: pd.DataFrame) -> pd.DataFrame:
    """Pivot canonical omics events into one feature row per subject.

    Each subject becomes a single row whose columns are the omics channel
    values (numeric), plus ``session_id`` and a ``target`` column derived
    from ``label_value``.  The ``target`` column preserves the raw label
    string (e.g. ``"high_risk"`` / ``"low_risk"``).

    Parameters
    ----------
    events:
        Validated canonical events frame containing omics data.

    Returns
    -------
    pd.DataFrame
        Wide feature frame indexed by ``subject_id`` with columns:
        ``session_id``, ``target``, and one numeric column per omics channel.
        Rows are sorted by ``subject_id``.
    """
    omics_events = events[events["modality"] == "omics"].copy()
    if omics_events.empty:
        raise ValueError("No omics events (modality=='omics') found in the supplied frame.")

    # Pivot: rows = subject_id, columns = channel, values = mean value
    pivot = omics_events.pivot_table(
        index="subject_id",
        columns="channel",
        values="value",
        aggfunc="mean",
    )
    pivot.columns.name = None
    pivot = pivot.reset_index()

    # Attach session_id and target from the last event per subject (stable across subject)
    meta = (
        omics_events.sort_values(["subject_id", "timestamp_utc"])
        .groupby("subject_id", sort=False)
        .agg(session_id=("session_id", "last"), target=("label_value", "last"))
        .reset_index()
    )

    result = meta.merge(pivot, on="subject_id", how="left")
    result = result.sort_values("subject_id").reset_index(drop=True)
    return result
