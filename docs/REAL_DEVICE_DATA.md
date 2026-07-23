# Real (non-public) DVXR device data

To satisfy the POW's requirement that the system incorporate data from the actual lab
hardware — **a) Galea biosensing headset, b) EMOTIV EEG systems** — not only public
datasets, the pipeline ingests two real sessions captured on **2026-06-08** on DVXR lab
equipment. Every number below is measured from the files
(`scripts/ingest_real_devices.py`; summary `outputs/_r2/real_device_ingestion.json`;
figure `presentation/figures/fig_real_device_data.png`).

## EMOTIV EPOC X (serial E5020C07)

Ingested via `dvxr.bci_real.ingest_emotiv` — the pipeline's EMOTIV path — with no changes.

- **14-channel EEG @128 Hz** (AF3 F7 F3 FC5 T7 P7 O1 O2 P8 T8 FC6 F4 F8 AF4) — **175,857 samples**, **1,373 s (~23 min)**.
- **Mental-command labels** (10,991): Neutral 9,279 · Left 728 · Right 410 · Push 368 · Pull 206 — the real avatar-control signal.
- **Band power** (POW.\*, 14 ch × 5 bands), **Performance Metrics** (stress/engagement/excitement/relaxation/focus), motion, facial expression.

## Galea / OpenBCI

- **16-channel EEG**, BrainFlow RAW export, **52,548 samples** (29 columns incl. aux/accel).
  Ingestible via `scripts/convert_galea_subject.py` / the Galea path.

## Why this matters

- **Not public-only.** The framework's EEG modality is now exercised on the exact devices the
  POW names (Galea, EMOTIV EPOC X) — real subject-recorded sessions, not just WESAD/DEAP/Mumtaz.
- **Real avatar-control signal.** The EMOTIV mental commands are the genuine control stream the
  BCI digital-twin demo consumes (Neutral/Left/Right/Push/Pull).
- **Honesty caveat (unchanged):** these are single-subject, single-session recordings. The
  mental-command labels are EMOTIV's on-device engine output, not experimenter-cued intent —
  demonstration-grade, not validated neural decoding. The device's Performance-Metric stress is
  a proprietary index, useful as a real signal but not a clinical measure.

## Reproduce
```
python scripts/ingest_real_devices.py    # ingests both, writes the summary + figure
```

> Raw recordings (the 114 MB EMOTIV CSV, the Galea BrainFlow files, and the source zips) are
> kept **out of version control** — only the measured summary, figure, and this doc are committed.
