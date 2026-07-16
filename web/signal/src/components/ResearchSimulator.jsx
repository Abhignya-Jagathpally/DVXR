import { useCallback, useMemo, useRef, useState } from "react";
import InputModeSelector from "./InputModeSelector.jsx";
import MetabolicInputs from "./MetabolicInputs.jsx";
import PhysiologicalInputs from "./PhysiologicalInputs.jsx";
import NeuralInputs from "./NeuralInputs.jsx";
import ClinicalInputs from "./ClinicalInputs.jsx";
import SignalUpload from "./SignalUpload.jsx";
import SignalReadiness from "./SignalReadiness.jsx";
import TargetSelector from "./TargetSelector.jsx";
import PredictionProgress from "./PredictionProgress.jsx";
import ResearchResult from "./ResearchResult.jsx";
import HumanSignalCanvas from "./HumanSignalCanvas.jsx";
import { defaultInputs, applySample, SAMPLES } from "../lib/researchModel.js";
import { TARGET_ORDER } from "../lib/researchPredictionTypes.js";
import { useResearchPrediction } from "../lib/useResearchPrediction.js";
import { DEMO_MODE, SYNTHETIC_BADGE } from "../lib/researchPredictionApi.js";
import { prefersReducedMotion } from "../lib/useReveal.js";

export default function ResearchSimulator() {
  const [inputs, setInputs] = useState(defaultInputs);
  const [mode, setMode] = useState("manual");
  const [outcome, setOutcome] = useState("diabetes_status");
  const [activeSample, setActiveSample] = useState(null);
  const sessionId = useRef("sess_" + Math.random().toString(36).slice(2, 10)).current;
  const resultsRef = useRef(null);
  const { status, result, error, run, reset, isLoading, isDone } = useResearchPrediction();

  const setField = useCallback((groupKey, fieldKey, next) => {
    setInputs((prev) => ({
      ...prev,
      [groupKey]: { ...prev[groupKey], [fieldKey]: next },
    }));
  }, []);

  const resetAll = useCallback(() => {
    setInputs(defaultInputs());
    setActiveSample(null);
  }, []);

  const chooseSample = useCallback((id) => {
    setInputs(applySample(id));
    setActiveSample(id);
    setMode("manual");
  }, []);

  const onGenerate = useCallback(() => {
    run({ sessionId, inputMode: mode, selectedOutcome: outcome, inputs, targets: TARGET_ORDER });
    // Scroll the stage into view once results begin.
    requestAnimationFrame(() => {
      if (resultsRef.current)
        resultsRef.current.scrollIntoView({
          behavior: prefersReducedMotion() ? "auto" : "smooth",
          block: "start",
        });
    });
  }, [run, sessionId, mode, outcome, inputs]);

  const onBack = useCallback(() => {
    reset();
  }, [reset]);

  const modeName = useMemo(
    () => ({ manual: "Manual values", upload: "Signal upload", sample: "Sample profile" }[mode]),
    [mode]
  );

  return (
    <section id="simulator" className="pad-y">
      <div className="wrap">
        <div className="kicker reveal">
          <span className="eyebrow">Research simulator</span>
        </div>
        <h2 className="display chapter-h reveal">
          Compose a profile.<br />Watch the model reason.
        </h2>
        <p className="sim-lede reveal">
          Enter research-profile inputs across modalities and generate a transparent, fully computed
          estimate. {DEMO_MODE ? "Running the in-page synthetic model — no data leaves your browser." : "Connected to the research backend."}
        </p>
        <div className="sim-topflags reveal">
          <span className="sim-synbadge lg">{SYNTHETIC_BADGE}</span>
          <span className="sim-proto-line">Research prototype — not a diagnosis.</span>
        </div>

        {isLoading ? (
          <div className="sim-stage" ref={resultsRef}>
            <PredictionProgress />
          </div>
        ) : isDone && result ? (
          <div className="sim-stage" ref={resultsRef}>
            <ResearchResult result={result} inputs={inputs} onBack={onBack} />
          </div>
        ) : (
          <div className="sim-shell" ref={resultsRef}>
            {/* LEFT — inputs */}
            <div className="sim-col sim-col-inputs">
              <InputModeSelector mode={mode} onMode={setMode} />

              {mode === "manual" && (
                <div className="sim-inputs" aria-label="Research profile inputs">
                  <div className="sim-inputs-bar">
                    <span>Research profile inputs</span>
                    <button type="button" className="sim-reset-all" onClick={resetAll}>
                      Reset all
                    </button>
                  </div>
                  <MetabolicInputs inputs={inputs} onChange={setField} />
                  <PhysiologicalInputs inputs={inputs} onChange={setField} />
                  <NeuralInputs inputs={inputs} onChange={setField} />
                  <ClinicalInputs inputs={inputs} onChange={setField} />
                </div>
              )}

              {mode === "upload" && <SignalUpload />}

              {mode === "sample" && (
                <div className="sim-samples">
                  {SAMPLES.map((s) => (
                    <button
                      key={s.id}
                      type="button"
                      className={"sim-samplecard" + (activeSample === s.id ? " is-active" : "")}
                      onClick={() => chooseSample(s.id)}
                    >
                      <b>{s.label}</b>
                      <small>{s.blurb}</small>
                    </button>
                  ))}
                  <p className="sim-samples-note">
                    Selecting a profile loads its synthetic values into the manual inputs.
                  </p>
                </div>
              )}
            </div>

            {/* CENTER — human signal viz */}
            <div className="sim-col sim-col-viz">
              <div className="sim-viz-frame">
                <HumanSignalCanvas inputs={inputs} />
                <div className="sim-viz-legend">
                  <span data-mod="neural">Neural</span>
                  <span data-mod="physiological">Cardio</span>
                  <span data-mod="clinical">EDA</span>
                  <span data-mod="metabolic">Metabolic</span>
                </div>
              </div>
              <p className="sim-viz-cap mono">
                Regions respond live to the inputs on the left · {modeName}
              </p>
            </div>

            {/* RIGHT — readiness + target + generate */}
            <div className="sim-col sim-col-right">
              <SignalReadiness inputs={inputs} />
              <TargetSelector selected={outcome} onSelect={setOutcome} />
              {status === "validation-failure" && (
                <div className="sim-alert" role="alert">
                  {error}
                </div>
              )}
              {status === "api-unavailable" && (
                <div className="sim-alert" role="alert">
                  {error} The synthetic demo mode can run this offline.
                </div>
              )}
              <button type="button" className="btn solid sim-generate" onClick={onGenerate}>
                Generate profile <span className="arw">→</span>
              </button>
              <p className="sim-generate-note">
                Synthetic demonstration · numbers are computed live from your inputs.
              </p>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
