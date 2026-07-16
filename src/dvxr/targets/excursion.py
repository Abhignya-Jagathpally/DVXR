"""Prospective glucose-excursion target builder (Gate 1).

For an anchor (cutoff) time ``t``, the feature window is ``[t - history, t]`` and the label asks whether
a glucose excursion occurs in the strictly-future window ``(t, t+horizon]``:

    Y_h(t) = 1  iff  any CGM sample in (t, t+h]  is  > high_mg_dl  OR  < low_mg_dl

The builder is deterministic (no wall-clock, no RNG) and records full provenance for every example:
the feature/target window bounds, the first excursion time, the realized horizon, the threshold
version, and a **censoring** status — an anchor whose future window is not observed closely enough to
``t+h`` (a CGM gap) is censored (``label=None``) rather than silently labelled 0. This prevents a
missing future from masquerading as "no excursion".

Nothing here fabricates data or crosses cohorts: it operates on a single subject's CGM timeline and is
equally valid on the synthetic fixture or a real single cohort (e.g. CGMacros). It computes only a
*label* from the future outcome; it never lets future samples enter the feature window.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExcursionThresholds:
    """Versioned outcome definition (spec §3 "thresholds live in versioned study config"). The
    ``version`` string is stamped onto every produced example so a label set is always traceable to
    the exact definition that made it."""
    version: str = "pilot-v1"
    low_mg_dl: float = 70.0             # hypoglycaemia bound
    high_mg_dl: float = 180.0          # hyperglycaemia bound
    horizons_minutes: Tuple[int, ...] = (30, 60)
    history_minutes: int = 240          # 4h feature context (spec §6 CGM)
    target_tolerance_minutes: float = 5.0   # a target window needs a sample within tol of t+h
    min_history_samples: int = 1        # an anchor needs at least this many samples in [t-H, t]


@dataclass(frozen=True)
class ExcursionExample:
    """One prospective example with full causal provenance. ``label`` is None iff ``censored``."""
    subject_id: str
    anchor_time: pd.Timestamp           # the cutoff t (== feature_window_end == target_window_start)
    feature_window_start: pd.Timestamp
    feature_window_end: pd.Timestamp
    horizon_minutes: int
    target_window_start: pd.Timestamp
    target_window_end: pd.Timestamp
    label: Optional[int]
    first_excursion_time: Optional[pd.Timestamp]
    censored: bool
    censor_reason: Optional[str]
    actual_horizon_minutes: float       # realized (last future sample - t); <= horizon under gaps
    n_history_samples: int
    threshold_version: str
    # --- outcome taxonomy (an EARLY-WARNING target must separate onset from persistence) ---
    anchor_glucose: Optional[float] = None       # the reading at t (last sample in the feature window)
    anchor_state: str = "unknown"                # in_range | hyper | hypo | unknown
    #: incident_excursion (in-range at t → new excursion), persistent_excursion (out of range at t, still
    #: out at t+h), recovery (out of range at t → back in range at t+h), no_excursion, or None if censored.
    outcome_class: Optional[str] = None
    #: the PRIMARY early-warning label: 1 = incident onset, 0 = in-range at t and stayed in range; None
    #: for anchors that were already out of range at t (excluded from the incident model) or censored.
    incident_label: Optional[int] = None

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        for k in ("anchor_time", "feature_window_start", "feature_window_end",
                  "target_window_start", "target_window_end", "first_excursion_time"):
            v = d[k]
            d[k] = None if v is None or (isinstance(v, float) and np.isnan(v)) else pd.Timestamp(v).isoformat()
        return d


def _prep(cgm: pd.DataFrame, time_col: str, glucose_col: str,
          subject_col: Optional[str]) -> pd.DataFrame:
    if time_col not in cgm.columns or glucose_col not in cgm.columns:
        raise KeyError(f"cgm must have columns {time_col!r} and {glucose_col!r}; got {list(cgm.columns)}")
    out = cgm[[c for c in (subject_col, time_col, glucose_col) if c]].copy()
    out[time_col] = pd.to_datetime(out[time_col], utc=False)
    out = out.dropna(subset=[time_col, glucose_col])
    if subject_col is None:
        out["__subject__"] = "single"
    else:
        out = out.rename(columns={subject_col: "__subject__"})
    out = out.rename(columns={time_col: "__t__", glucose_col: "__g__"})
    out["__g__"] = out["__g__"].astype(float)
    # deterministic order: subject, then time, stable
    return out.sort_values(["__subject__", "__t__"], kind="stable").reset_index(drop=True)


def history_slice(cgm: pd.DataFrame, anchor: pd.Timestamp, *,
                  thresholds: ExcursionThresholds = ExcursionThresholds(),
                  time_col: str = "timestamp", glucose_col: str = "glucose",
                  subject_col: Optional[str] = None, subject_id: Optional[str] = None) -> pd.DataFrame:
    """Return the causal feature-window samples ``[anchor - history, anchor]`` for one anchor. Never
    includes a sample after ``anchor`` — this is the slice downstream feature builders may consume."""
    p = _prep(cgm, time_col, glucose_col, subject_col)
    if subject_id is not None:
        p = p[p["__subject__"] == subject_id]
    anchor = pd.Timestamp(anchor)
    start = anchor - pd.Timedelta(minutes=thresholds.history_minutes)
    win = p[(p["__t__"] >= start) & (p["__t__"] <= anchor)]
    return win.rename(columns={"__t__": time_col, "__g__": glucose_col})[[time_col, glucose_col]]


def _label_one(future_g: np.ndarray, future_t: pd.DatetimeIndex, anchor: pd.Timestamp,
               horizon_end: pd.Timestamp, thr: ExcursionThresholds
               ) -> Tuple[Optional[int], Optional[pd.Timestamp], bool, Optional[str], float]:
    """Label a single (anchor, horizon). Returns (label, first_excursion_time, censored, reason,
    actual_horizon_minutes)."""
    if len(future_g) == 0:
        return None, None, True, "no_future_samples", 0.0
    last_t = future_t[-1]
    actual = (last_t - anchor) / pd.Timedelta(minutes=1)
    # the future window must be observed close enough to its end, else we cannot assert "no excursion"
    if (horizon_end - last_t) > pd.Timedelta(minutes=thr.target_tolerance_minutes):
        return None, None, True, "insufficient_future_coverage", float(actual)
    excursion = (future_g > thr.high_mg_dl) | (future_g < thr.low_mg_dl)
    if excursion.any():
        first = future_t[int(np.argmax(excursion))]   # argmax → first True
        return 1, pd.Timestamp(first), False, None, float(actual)
    return 0, None, False, None, float(actual)


def _anchor_state(glucose: Optional[float], thr: ExcursionThresholds) -> str:
    """The participant's glucose STATE at the anchor t (from the reading at t): in_range | hyper | hypo
    | unknown. This is what separates an early-warning (incident) anchor from one already out of range."""
    if glucose is None or (isinstance(glucose, float) and np.isnan(glucose)):
        return "unknown"
    if glucose < thr.low_mg_dl:
        return "hypo"
    if glucose > thr.high_mg_dl:
        return "hyper"
    return "in_range"


def _classify_outcome(anchor_state: str, label: Optional[int], end_in_range: Optional[bool]
                      ) -> Tuple[Optional[str], Optional[int]]:
    """Map (state at t, future-excursion label, in-range-at-t+h?) to an outcome class and the PRIMARY
    incident label. In-range-at-t → incident onset (1) vs stayed-in-range (0). Already out of range at t
    → persistent (still out at t+h) or recovery (back in range) — excluded from the incident label."""
    if label is None:                                  # censored
        return None, None
    if anchor_state == "in_range":
        return ("incident_excursion" if label == 1 else "no_excursion"), int(label == 1)
    if anchor_state in ("hyper", "hypo"):
        if end_in_range is None:
            return None, None
        return ("recovery" if end_in_range else "persistent_excursion"), None
    return None, None                                  # unknown state at t → no taxonomy


def build_excursion_labels(
    cgm: pd.DataFrame,
    *,
    thresholds: ExcursionThresholds = ExcursionThresholds(),
    anchors: Optional[Sequence[pd.Timestamp]] = None,
    time_col: str = "timestamp",
    glucose_col: str = "glucose",
    subject_col: Optional[str] = None,
    label_definition: str = "any",
) -> pd.DataFrame:
    """Build the prospective excursion label table for a CGM timeline.

    Each row is one (anchor, horizon) example. If ``anchors`` is None, every observed timestamp that
    has at least ``min_history_samples`` in its feature window is used as an anchor (deterministic).
    Censored examples are kept in the table with ``label=NaN`` and a ``censor_reason`` so callers can
    audit coverage; drop them with ``df[df.censored == False]`` for a reportable label set.

    Every row also carries the outcome taxonomy (``anchor_state``, ``outcome_class``, ``incident_label``)
    so an EARLY-WARNING model can be trained on *incident onset* rather than a persistence detector.
    ``label_definition`` selects what the primary ``label`` column means:

    * ``"any"`` (default, back-compat): label = 1 iff ANY future sample in (t, t+h] is out of range —
      this counts an already-hyperglycaemic participant who stays high as positive (persistence).
    * ``"incident"``: label = the incident onset among anchors that were IN RANGE at t; anchors already
      out of range at t are censored (``out_of_range_at_anchor``) so they leave the reportable set. Use
      this for the honest prospective early-warning claim.
    """
    if label_definition not in ("any", "incident"):
        raise ValueError(f"label_definition must be 'any' or 'incident', got {label_definition!r}")
    p = _prep(cgm, time_col, glucose_col, subject_col)
    rows: List[dict] = []
    for sid, g in p.groupby("__subject__", sort=True):
        times = g["__t__"].to_numpy()
        vals = g["__g__"].to_numpy()
        tindex = pd.DatetimeIndex(g["__t__"])
        if anchors is None:
            anchor_list = list(tindex)
        else:
            anchor_list = [pd.Timestamp(a) for a in anchors]
        for anchor in anchor_list:
            hist_start = anchor - pd.Timedelta(minutes=thresholds.history_minutes)
            hist_mask = (tindex >= hist_start) & (tindex <= anchor)
            hist_arr = hist_mask.to_numpy() if hasattr(hist_mask, "to_numpy") else np.asarray(hist_mask)
            n_hist = int(hist_arr.sum())
            if n_hist < thresholds.min_history_samples:
                continue
            # the reading AT t (last sample in the feature window) → the participant's state at the anchor
            hist_vals = vals[hist_arr]
            anchor_glucose = float(hist_vals[-1]) if len(hist_vals) else None
            anchor_state = _anchor_state(anchor_glucose, thresholds)
            for h in thresholds.horizons_minutes:
                horizon_end = anchor + pd.Timedelta(minutes=h)
                fut_mask = (tindex > anchor) & (tindex <= horizon_end)
                fut_t = tindex[fut_mask]
                fut_g = vals[fut_mask.to_numpy() if hasattr(fut_mask, "to_numpy") else fut_mask]
                label, first, censored, reason, actual = _label_one(
                    np.asarray(fut_g, dtype=float), fut_t, anchor, horizon_end, thresholds)
                # is the participant back in range at t+h? (the reading nearest the horizon end)
                end_in_range = None
                if not censored and len(fut_g):
                    ev = float(np.asarray(fut_g, dtype=float)[-1])
                    end_in_range = bool(thresholds.low_mg_dl <= ev <= thresholds.high_mg_dl)
                outcome_class, incident_label = _classify_outcome(anchor_state, label, end_in_range)
                primary_label, primary_censored, primary_reason = label, censored, reason
                if label_definition == "incident":
                    # the incident model scores only anchors that were in range at t; already-out-of-range
                    # anchors are censored so they never inflate a "persistence" as early warning.
                    if not censored and anchor_state != "in_range":
                        primary_label, primary_censored, primary_reason = None, True, "out_of_range_at_anchor"
                    else:
                        primary_label = incident_label
                ex = ExcursionExample(
                    subject_id=str(sid),
                    anchor_time=anchor,
                    feature_window_start=hist_start,
                    feature_window_end=anchor,
                    horizon_minutes=int(h),
                    target_window_start=anchor,
                    target_window_end=horizon_end,
                    label=primary_label,
                    first_excursion_time=first,
                    censored=primary_censored,
                    censor_reason=primary_reason,
                    actual_horizon_minutes=actual,
                    n_history_samples=n_hist,
                    threshold_version=thresholds.version,
                    anchor_glucose=anchor_glucose,
                    anchor_state=anchor_state,
                    outcome_class=outcome_class,
                    incident_label=incident_label,
                )
                rows.append(ex.to_dict())
    cols = list(ExcursionExample.__dataclass_fields__)
    return pd.DataFrame(rows, columns=cols)
