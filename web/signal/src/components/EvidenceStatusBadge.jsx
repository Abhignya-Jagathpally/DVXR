// Small, deliberately subtle per-card evidence label. Never conveys risk by
// color alone — it is a plain text status.
const TITLES = {
  "component-model": "Component sub-model estimate.",
  experimental: "Experimental research estimate — not validated for clinical use.",
  "metabolic-model": "Metabolic-only research model.",
  abstained: "No estimate produced — insufficient evidence.",
};

export default function EvidenceStatusBadge({ status, className = "" }) {
  if (!status) return null;
  return (
    <span
      className={"sim-ev-badge " + className}
      data-status={status}
      title={TITLES[status] || "Research evidence status."}
    >
      {status}
    </span>
  );
}
