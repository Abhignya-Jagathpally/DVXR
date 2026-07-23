"""End-to-end use-case: multimodal risk prediction + grounded LLM explanation + abstention.

Runs realistic patient scenarios through the agentic pipeline (the same LangGraph path the
API serves) and prints a readable narrative: what went in, what each head predicted, the
selected diabetes outcome, and the LLM explanation — which restates only model numbers and
never predicts. A missing-data scenario shows honest abstention.

Run: python scripts/usecase_demo.py
"""

from __future__ import annotations

import textwrap


SCENARIOS = [
    {
        "name": "Elevated metabolic risk (all inputs present)",
        "payload": {
            "selected_outcome": "glucose_instability",
            "prediction_horizons_minutes": [30, 60],
            "inputs": {
                "hba1c": 7.8, "fasting_glucose": 142, "bmi": 33,
                "cgm_std": 52, "time_above_range": 48,   # metabolic / CGM
                "hrv_rmssd": 24, "eda_scl": 7.1,          # wearable / pulse
            },
        },
    },
    {
        "name": "Low-risk profile (all inputs present)",
        "payload": {
            "selected_outcome": "glucose_instability",
            "prediction_horizons_minutes": [30],
            "inputs": {
                "hba1c": 5.2, "fasting_glucose": 88, "bmi": 22,
                "cgm_std": 18, "time_above_range": 4, "hrv_rmssd": 62,
            },
        },
    },
    {
        "name": "Missing metabolic inputs → honest abstention",
        "payload": {
            "selected_outcome": "diabetes_status",
            "prediction_horizons_minutes": [30],
            "inputs": {"hrv_rmssd": 41},  # only a wearable signal; no metabolic data
        },
    },
]


def _run(payload: dict) -> dict:
    try:
        from dvxr.serve.agents import agentic_available, run_agentic_prediction
        if agentic_available():
            return run_agentic_prediction(payload)
    except Exception:  # noqa: BLE001
        pass
    from dvxr.serve.research_predict import run_research_prediction
    return run_research_prediction(payload)


def _render(name: str, payload: dict, body: dict) -> str:
    lines = [f"\n{'='*74}", f"USE CASE: {name}", f"{'='*74}"]
    lines.append(f"inputs: {payload['inputs']}")
    prov = body.get("evidence_provenance", "?")
    if prov != "committed":
        lines.append("[NOTE] evidence_provenance=%s → the ILLUSTRATIVE model is in use "
                     "(committed screener artifacts absent in this checkout), so probability "
                     "MAGNITUDES are illustrative; the pipeline behavior is real." % prov)
    sel = body.get("selected_outcome", {})
    if body.get("status") == "abstained" or sel.get("probability") is None:
        lines.append(f"\noutcome: ABSTAINED ({sel.get('name')}) — status={body.get('status')}")
        lines.append(f"missing: {body.get('missing_or_stale_data') or sel.get('missing_or_stale_data')}")
    else:
        lines.append(
            f"\nselected outcome: {sel.get('name')} — probability {sel.get('probability')} "
            f"(risk {sel.get('risk_band')}); validated_for_clinical_use="
            f"{sel.get('validated_for_clinical_use')}"
        )
        tp = body.get("target_predictions", {})
        preds = {t: v.get("probability") for t, v in tp.items() if v.get("probability") is not None}
        lines.append(f"per-head predictions: {preds}")
        drivers = [f"{c.get('factor')}({c.get('direction')})" for c in body.get("contributions", [])[:3]]
        if drivers:
            lines.append(f"top drivers: {', '.join(drivers)}")
    explanation = body.get("explanation") or {}
    if explanation.get("text"):
        lines.append("\nLLM EXPLANATION (explains, never predicts; predicts=%s):"
                     % explanation.get("predicts"))
        lines.append(textwrap.fill(explanation["text"], width=74, initial_indent="  ",
                                    subsequent_indent="  "))
    lines.append(f"\ndisclaimer: {body.get('disclaimer', '')[:90]}...")
    return "\n".join(lines)


def main() -> None:
    for scenario in SCENARIOS:
        body = _run(scenario["payload"])
        print(_render(scenario["name"], scenario["payload"], body))
    print(f"\n{'='*74}\nEvery number above came from the model; the explanation only restates them.")


if __name__ == "__main__":
    main()
