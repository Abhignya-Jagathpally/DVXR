// "WHAT MOVED THE ESTIMATE" — signed contribution rows (factor / value /
// direction / method). Includes the required non-causal disclosure note.
export default function ContributionWaterfall({ contributions }) {
  const c = contributions || [];
  if (!c.length) {
    return (
      <div className="sim-waterfall">
        <div className="sim-wf-empty">No contribution provided for this outcome.</div>
      </div>
    );
  }
  const max = Math.max(...c.map((x) => Math.abs(x.signed_contribution))) || 1;
  return (
    <div className="sim-waterfall">
      {c.slice(0, 8).map((x) => {
        const pos = x.signed_contribution >= 0;
        const w = (Math.abs(x.signed_contribution) / max) * 50;
        return (
          <div className="sim-wf-row" key={x.factor}>
            <span className="sim-wf-f">{x.factor}</span>
            <div className="sim-wf-bar">
              <i
                className={pos ? "pos" : "neg"}
                style={pos ? { width: w + "%", left: "50%" } : { width: w + "%", right: "50%" }}
              />
            </div>
            <b className={"sim-wf-v " + (pos ? "pos" : "neg")} title={`method: ${x.method} · ${x.direction}`}>
              {pos ? "+" : ""}
              {x.signed_contribution.toFixed(2)}
            </b>
          </div>
        );
      })}
      <p className="sim-contrib-note">
        Contribution indicates how a variable influenced this model estimate. It does not establish
        that the variable caused the outcome.
      </p>
    </div>
  );
}
