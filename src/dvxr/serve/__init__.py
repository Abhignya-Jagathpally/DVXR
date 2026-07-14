"""dvxr.serve — the product serving layer.

Turns the repo's *validated* models (real LaBraM EEG foundation model; do-no-harm fusion;
calibrated band-power heads) into a usable "fit once → save → load → predict(new subject) →
calibrated, explained, evidence-backed screening score" path — the piece that did not exist
while the winning models lived only inside the offline benchmark.

Research-grade screening / decision-support only; never a clinical diagnostic claim. Every
accuracy number a Screener reports is its own subject-held-out estimate and matches the
committed benchmark scoreboard.
"""
from dvxr.serve.screener import Screener, fit_screener, REPRESENTATION_BY_TASK  # noqa: F401

__all__ = ["Screener", "fit_screener", "REPRESENTATION_BY_TASK"]
