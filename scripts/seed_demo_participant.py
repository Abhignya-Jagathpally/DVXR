#!/usr/bin/env python
"""Seed a demo PARTICIPANT so the live Generate engine produces a genuine, traceable review.

The Generate API fails closed: a request needs (1) a registered CONSENT scope for the caller's
purpose, and (2) ingested CGM EVENTS for that (tenant, patient), plus an ACTIVE committed CGM model
(built by scripts/build_cgm_artifact.py) — all in the SAME registry db the API reads (DVXR_DB_PATH).
Out of the box none of these exist, so every identifier (including the placeholder ``PSEUDO-DEMO``)
raises ConsentError and no traceable prediction_id is minted.

This script seeds a pseudonymous participant from a REAL CGMacros subject:
  * consent scope for the requested purposes (research/participant/clinical), tenant-scoped
  * that subject's REAL CGM glucose series as API events

Honesty: glucose VALUES are the subject's real CGMacros readings — never fabricated. Only the
timestamps are re-anchored so the series ends at ``--anchor`` (default: now), because CGMacros was
recorded in 2020 and a "Generate now" request resolves its causal cutoff to the present; without
re-anchoring the forecaster finds no recent history and abstains. The re-anchoring is recorded on
every event (``time_reanchored_from``) so it is auditable, not hidden.

Usage:
    python scripts/seed_demo_participant.py --db artifacts/registry.db \
        --data data/real/cgmacros --participant CGM-DEMO-01 --source-subject cgmacros_001 \
        [--tenant default] [--purposes research participant clinical] \
        [--history-hours 12] [--build-artifact]

Then run the API with DVXR_DB_PATH pointing at the same db and generate a review for the printed id.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dvxr.loaders import load_cgmacros_dataset  # noqa: E402
from dvxr.storage.local import open_local_stores  # noqa: E402


def _cgm_series(data_root: str, source_subject: str | None, source: str = "dexcom") -> pd.DataFrame:
    ev = load_cgmacros_dataset(data_root, subjects=None, include_bio=False)
    cgm = ev[ev["modality"] == "cgm"].copy()
    if "glucose_source" in cgm.columns and source in set(cgm["glucose_source"]):
        cgm = cgm[cgm["glucose_source"] == source]
    subjects = sorted(cgm["subject_id"].unique())
    if not subjects:
        raise SystemExit(f"no CGMacros CGM data under {data_root!r}")
    sid = source_subject or subjects[0]
    if sid not in subjects:
        raise SystemExit(f"source subject {sid!r} not found; available: {subjects[:8]}…")
    one = cgm[cgm["subject_id"] == sid].sort_values("timestamp_utc").reset_index(drop=True)
    return one, sid


def _to_api_events(series: pd.DataFrame, *, tenant: str, participant: str, source_subject: str,
                   anchor: datetime, history_hours: int) -> list[dict]:
    ts = pd.to_datetime(series["timestamp_utc"], utc=True)
    # keep the most recent `history_hours` of the REAL series, then re-anchor so it ends at `anchor`
    last = ts.max()
    keep = ts >= (last - pd.Timedelta(hours=history_hours))
    series, ts = series[keep].reset_index(drop=True), ts[keep].reset_index(drop=True)
    shift = pd.Timestamp(anchor) - ts.max()
    events = []
    for i, (_, row) in enumerate(series.iterrows()):
        obs = (ts.iloc[i] + shift).isoformat()
        events.append({
            "tenant_id": tenant,
            "patient_id": participant,
            "event_id": f"{participant}-cgm-{i:06d}",
            "modality": "cgm",
            "value": float(row["value"]),
            "unit": row.get("unit", "mg/dL"),
            "observed_at_utc": obs,
            "source_system": "cgmacros",
            "source_subject": source_subject,           # provenance: which real subject
            "time_reanchored_from": ts.iloc[i].isoformat(),  # audit: original real timestamp
        })
    return events


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="seed a demo participant for a traceable live review")
    ap.add_argument("--db", default="artifacts/registry.db",
                    help="registry db the API reads via DVXR_DB_PATH (same file build_cgm_artifact uses)")
    ap.add_argument("--data", default="data/real/cgmacros")
    ap.add_argument("--participant", default="CGM-DEMO-01", help="pseudonymous identifier to type in the UI")
    ap.add_argument("--source-subject", default=None, help="real CGMacros subject to source glucose from")
    ap.add_argument("--tenant", default="default",
                    help="MUST match the API principal's tenant (default deployment: 'default'; "
                         "unsafe_dev demo principal: 'dev') — consent/events are tenant-scoped")
    ap.add_argument("--purposes", nargs="+", default=["research", "participant", "clinical"])
    ap.add_argument("--history-hours", type=int, default=12)
    ap.add_argument("--anchor", default=None, help="ISO instant the series ends at (default: now UTC)")
    ap.add_argument("--build-artifact", action="store_true",
                    help="also fit+register the active CGM artifact into --db (deploy step)")
    ap.add_argument("--artifact-root", default="artifacts")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    anchor = (datetime.fromisoformat(args.anchor) if args.anchor
              else datetime.now(timezone.utc)).astimezone(timezone.utc)

    Path(args.db).parent.mkdir(parents=True, exist_ok=True)

    if args.build_artifact:
        from build_cgm_artifact import build
        print(f"[seed] building + registering ACTIVE CGM artifact into {args.db} …")
        Path(args.artifact_root).mkdir(parents=True, exist_ok=True)
        build(args.data, args.artifact_root, args.db, seed=args.seed)

    series, sid = _cgm_series(args.data, args.source_subject)
    events = _to_api_events(series, tenant=args.tenant, participant=args.participant,
                            source_subject=sid, anchor=anchor, history_hours=args.history_hours)

    stores = open_local_stores(args.db)
    stores.consent.set_scope(args.participant, {"purposes": list(args.purposes)},
                             tenant_id=args.tenant)
    n = stores.events.append_events(events)

    print(f"[seed] db={args.db} tenant={args.tenant}")
    print(f"[seed] participant identifier to type in the UI:  {args.participant}")
    print(f"[seed] consent purposes: {args.purposes}")
    print(f"[seed] ingested {n} REAL CGM events from {sid} "
          f"(last {args.history_hours}h, re-anchored to end at {anchor.isoformat()})")
    print(f"[seed] active CGM model present: "
          f"{stores.models.active('cgm_glucose_forecast') is not None}")
    print("[seed] now run:  DVXR_DB_PATH=%s  (and artifact-root=%s) then Generate a "
          "'CGM glucose forecast' review for the id above." % (args.db, args.artifact_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
