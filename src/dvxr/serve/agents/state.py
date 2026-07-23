"""Typed state passed between orchestration-graph nodes.

The graph is an *orchestration* layer: every node delegates the numeric work to an
existing ``dvxr.serve.research_predict`` helper. Nodes accumulate their outputs into this
state; the ``calibration_gate`` node assembles the exact response body
``run_research_prediction`` produces, and the explanation node adds a grounded narrative
plus a per-node ``trace``. No node computes, imputes, or edits a prediction number.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class PipelineState(TypedDict, total=False):
    # inputs
    payload: dict
    screener_root: Any
    # resolved objects carried between nodes (kept out of the response body)
    req: Any
    heads: Any
    meta: Any
    # ingestion
    prediction_id: str
    provenance: str
    available_modalities: List[str]
    input_quality: Dict[str, Any]
    # per-modality encoder/predict outputs
    target_predictions: Dict[str, Any]
    prob_stack: Dict[str, float]
    # fusion / selected task
    selected_outcome: Dict[str, Any]
    contributions: List[Dict[str, Any]]
    forecast: Dict[str, Any]
    # gate + assembled body
    body: Dict[str, Any]
    abstained: bool
    # explanation (LLM explains, never predicts) + audit trace
    explanation: Dict[str, Any]
    trace: List[Dict[str, Any]]
    error: Optional[str]
