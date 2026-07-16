import TargetConstellation from "./TargetConstellation.jsx";
import ContributionWaterfall from "./ContributionWaterfall.jsx";
import GlucoseForecast from "./GlucoseForecast.jsx";
import EvidenceStatusBadge from "./EvidenceStatusBadge.jsx";
import AbstentionPanel from "./AbstentionPanel.jsx";
import { SYNTHETIC_BADGE } from "../lib/researchPredictionApi.js";
import { OUTCOME_LABEL, TARGET_ORDER, TARGET_NAMES } from "../lib/researchPredictionTypes.js";

const pct = (x) => Math.round(x * 100);
const cleanBand = (b) => (b || "").replace("research-", "");

export default function ResearchResult({ result, inputs, onBack }) {
  const r = result;
  const sel = r.selected;
  const outLabel = OUTCOME_LABEL[r.selected_outcome];

  return (
    <div className="sim-results" role="region" aria-label="Research profile result">
      <div className="sim-res-flag">
        <span className="sim-synbadge">{SYNTHETIC_BADGE}</span>
        <span>Model output, not real clinical results.</span>
      </div>

      {r.abstained ? (
        <AbstentionPanel inputs={inputs} outcomeLabel={outLabel} />
      ) : (
        <div className="sim-res-primary">
          <div className="sim-res-kicker">Multimodal research profile</div>
          <div className="sim-res-dialwrap">
            <div className="sim-res-dial" style={{ "--p": pct(sel.probability) }}>
              <div className="sim-res-dial-in">
                <b>
                  {pct(sel.probability)}
                  <i>%</i>
                </b>
                <small>{outLabel}</small>
              </div>
            </div>
            <div className="sim-res-primary-meta">
              <div className={"sim-res-band band-" + cleanBand(sel.risk_band)}>
                research · {cleanBand(sel.risk_band)}
              </div>
              <div className="sim-res-conf">
                <span>Confidence</span>
                <div className="sim-ctrack">
                  <i style={{ width: pct(sel.confidence) + "%" }} />
                </div>
                <b>{pct(sel.confidence)}%</b>
              </div>
              <p className="sim-res-say">
                The entered profile produced a{" "}
                <b>{sel.probability > 0.5 ? "raised" : "lower"}</b> {outLabel.toLowerCase()} research
                estimate. This is not a diagnosis.
              </p>
              <EvidenceStatusBadge status={sel.evidence_status} />
            </div>
          </div>
        </div>
      )}

      <div className="sim-res-grid">
        <div className="sim-res-card">
          <div className="sim-res-h">
            Associated research signals
            <EvidenceStatusBadge status="component-model" />
          </div>
          <TargetConstellation result={r} />
          <div className="sim-const-note">
            Node size = probability · opacity = confidence · link = model contribution to the estimate
          </div>
        </div>

        <div className="sim-res-card">
          <div className="sim-res-h">
            What moved the estimate
            <EvidenceStatusBadge status={sel && sel.evidence_status} />
          </div>
          <ContributionWaterfall contributions={r.contributions} />
        </div>
      </div>

      {r.forecast && (
        <div className="sim-res-card sim-fc-card">
          <div className="sim-res-h">
            Near-term glucose outlook
            <EvidenceStatusBadge status="metabolic-model" />
          </div>
          <GlucoseForecast forecast={r.forecast} />
        </div>
      )}

      <div className="sim-tcards">
        {TARGET_ORDER.map((t) => {
          const tp = r.target_predictions[t];
          return (
            <div className="sim-tcard" key={t}>
              <span className="sim-tc-name">{TARGET_NAMES[t]}</span>
              <b className="sim-tc-p">{pct(tp.probability)}%</b>
              <span className={"sim-tc-band b-" + tp.risk_band}>{tp.risk_band}</span>
              <EvidenceStatusBadge status={tp.evidence_status} className="tc-ev" />
            </div>
          );
        })}
      </div>

      <div className="sim-res-foot">
        <span className="mono">
          {(sel && sel.model_version) || "research-sim"} · {r.prediction_id}
        </span>
        <span className="mono">{r.disclaimer}</span>
      </div>

      <p className="sim-res-proto">
        Research prototype — not a diagnosis. All values are a synthetic demonstration of the modeling
        flow.
      </p>

      <div className="sim-res-actions">
        <button className="btn ghost" type="button" onClick={onBack}>
          ← Adjust inputs
        </button>
      </div>
    </div>
  );
}
