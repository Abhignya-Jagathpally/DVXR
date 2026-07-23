# Interoperability + interpretation

Every cross-system surface carries not just numbers but an **interpretation payload** (risk
band + grounded reason + honesty flags), so a downstream device or record never receives a
bare value it could misread.

## Devices in / out (Galea · EMOTIV · CGM · wearables)

- **LSL stream contracts** (`neuroglycemic-sentinel/config/lsl_streams.json`) define three
  logical modalities for real-time ingest: `eeg` (Emotiv/Galea/Muse), `wearable`
  (Empatica/Galea — PPG/BVP/EDA/temp/HR/SpO₂), and `reference_glucose` (CGM). XDF capture via
  LabRecorder; replay via `lsl-replay` / `lsl-session-replay`. This is the plug-and-play
  ingestion contract the POW calls for.
- **RT-Demo stream** (`docs/RT_DEMO_STREAM_CONTRACT.md`, `rt-demo-v1`) is the outbound
  real-time contract shared by the web and Unity clients — each frame carries the decoded
  command **and** its confidence, the stress index, the glucose channel **with its
  `evidence_status`** (abstains rather than emitting a number when data is insufficient), and
  an always-present `experimental` flag.

## Clinical record interoperability (FHIR)

`neuroglycemic-sentinel/src/neuroglycemic/fhir.py::neural_forecast_observation` exports a
glucose forecast as a **FHIR Observation** resource (LOINC-coded glucose, per-horizon
components, the prediction interval, and the research-only status), so a forecast can enter an
EHR as a structured, interpretable record rather than a loose number. Emitted on demand by the
`neural-case` CLI (`--fhir-patient-reference`).

## Interpretation travels with the value

Across all three surfaces the same invariants hold:
- a **risk band** and, where applicable, a **grounded reason** accompany every probability;
- **abstention is explicit** — a missing-data state is transmitted as such, never as a
  fabricated value or a silent zero;
- `validated_for_clinical_use=False` and a research-stage disclaimer are attached at the
  boundary, so no consumer can mistake a research estimate for a cleared clinical result.

## How latency and hallucination are handled at the boundary

- **Latency** (`outputs/latency_report.md`): the serving paths return in <3 ms (direct 0.15 ms,
  LangGraph-orchestrated 2.3 ms, RT frame ~0 ms), and the glucose point models predict in
  <1.5 ms — real-time-safe for streaming.
- **Hallucination**: the numeric body is frozen before the explanation node; the LLM/narrative
  may only restate it, enforced by `_validate_llm_payload` and
  `tests/test_no_hallucinated_numbers.py`. Cross-system payloads therefore cannot carry an
  ungrounded number introduced by a language model.
