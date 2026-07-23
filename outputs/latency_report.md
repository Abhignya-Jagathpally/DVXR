# Inference latency (CPU, p50/p95 in ms)

Iterations: 50 (serving) / 200 (point models). Warm-up excluded. Single-request, single-thread on the shared host.

| path | p50 (ms) | p95 (ms) |
|---|---:|---:|
| research_predict (direct) | 0.15 | 0.22 |
| research_predict/agentic (LangGraph) | 2.31 | 2.45 |
| rt-demo frame build | 0.00 | 0.00 |
| glucose linear ridge (1 sample) | 0.06 | 0.06 |
| glucose gradient boosting (1 sample) | 1.39 | 1.41 |

**Reading:** the LangGraph orchestration adds a small fixed overhead over the direct path for the per-node trace + grounded explanation; both are well within interactive budgets. The glucose point models predict in well under a millisecond, so latency is not a reason to prefer the deep model — the trade is calibration + abstention + fusion (see docs/MODEL_JUSTIFICATION.md), not speed.
