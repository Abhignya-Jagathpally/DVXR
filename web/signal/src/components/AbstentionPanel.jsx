import { GROUP_KEYS } from "../lib/researchPredictionTypes.js";
import { FIELDS, groupPresence } from "../lib/researchModel.js";

// Responsible abstention: shown when the model declines to produce a fused
// estimate. Lists which modalities were available vs. missing.
export default function AbstentionPanel({ inputs, outcomeLabel }) {
  const rows = [
    ...GROUP_KEYS.map((g) => {
      const p = groupPresence(inputs, g);
      return { label: FIELDS[g].label, available: p !== "none", partial: p === "partial" };
    }),
    { label: "Molecular", available: false, partial: false },
  ];
  const available = rows.filter((r) => r.available).map((r) => r.label);
  const missing = rows.filter((r) => !r.available).map((r) => r.label);

  return (
    <div className="sim-abstain" role="alert">
      <span className="sim-abstain-chip">Insufficient synchronized evidence</span>
      <h3 className="sim-abstain-h">{outcomeLabel} estimate was not generated.</h3>
      <p className="sim-abstain-p">
        An approved same-subject multimodal model artifact was unavailable for this request, so no
        fused probability was produced. Responsible abstention is a feature, not a failure — the
        system will not invent a result it cannot support.
      </p>
      <div className="sim-abstain-mods">
        <div className="sim-abstain-col">
          <span className="sim-abstain-lab">Available</span>
          <div className="sim-abstain-tags">
            {available.length ? (
              available.map((m) => (
                <span key={m} className="sim-abstain-tag on">
                  {m}
                </span>
              ))
            ) : (
              <span className="sim-abstain-tag none">none</span>
            )}
          </div>
        </div>
        <div className="sim-abstain-col">
          <span className="sim-abstain-lab">Missing</span>
          <div className="sim-abstain-tags">
            {missing.map((m) => (
              <span key={m} className="sim-abstain-tag off">
                {m}
              </span>
            ))}
          </div>
        </div>
      </div>
      <p className="sim-abstain-codes mono">
        status: abstained · risk: null (no fabricated probability)
      </p>
    </div>
  );
}
