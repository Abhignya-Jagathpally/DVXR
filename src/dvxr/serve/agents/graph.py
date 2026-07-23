"""LangGraph orchestration spine for the research-prediction pipeline.

The graph wraps existing ``research_predict`` helpers as nodes; it changes control flow,
never numbers. Wiring:

    ingestion_availability -> encode_predict -> fuse_select -> forecast
        -> calibration_gate --(abstained?)--> explain_abstention --> END
                             \\--(ok)--------> explain_prediction --> END

The conditional edge after ``calibration_gate`` is the fail-closed branch: an abstaining
request is routed to the abstention explainer and never revives a prediction.
"""

from __future__ import annotations

from functools import lru_cache

from dvxr.serve.agents import nodes
from dvxr.serve.agents.state import PipelineState


def build_graph():
    """Construct and compile the orchestration graph."""
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(PipelineState)
    graph.add_node("ingestion_availability", nodes.ingestion_availability)
    graph.add_node("encode_predict", nodes.encode_predict)
    graph.add_node("fuse_select", nodes.fuse_select)
    graph.add_node("forecast", nodes.forecast)
    graph.add_node("calibration_gate", nodes.calibration_gate)
    graph.add_node("explain_prediction", nodes.explain_prediction)
    graph.add_node("explain_abstention", nodes.explain_abstention)

    graph.add_edge(START, "ingestion_availability")
    graph.add_edge("ingestion_availability", "encode_predict")
    graph.add_edge("encode_predict", "fuse_select")
    graph.add_edge("fuse_select", "forecast")
    graph.add_edge("forecast", "calibration_gate")
    graph.add_conditional_edges(
        "calibration_gate",
        nodes.route_after_gate,
        {"explain_prediction": "explain_prediction",
         "explain_abstention": "explain_abstention"},
    )
    graph.add_edge("explain_prediction", END)
    graph.add_edge("explain_abstention", END)
    return graph.compile()


@lru_cache(maxsize=1)
def compiled_graph():
    """Compile once and reuse (the graph is stateless across invocations)."""
    return build_graph()
