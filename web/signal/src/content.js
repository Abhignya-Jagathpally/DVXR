// SIGNAL — single source of truth for copy + VERIFIED evidence numbers.
// Every figure below resolves to a committed scoreboard (source noted per item).

export const brand = {
  name: "SIGNAL",
  tagline: "Adaptive Multimodal Health Intelligence",
  nav: [
    { label: "Vision", href: "#problem" },
    { label: "Intelligence", href: "#engine" },
    { label: "Experience", href: "#experience" },
    { label: "Evidence", href: "#evidence" },
    { label: "Research", href: "#researcher" },
  ],
};

export const hero = {
  tag: "Adaptive Multimodal Health Intelligence",
  headline: ["See risk", "before it", "speaks."],
  subcopy:
    "Health does not emerge from one measurement. It emerges from the interaction between mind, physiology, metabolism, behavior, and clinical history.",
  primary: { label: "Explore the system", href: "#engine" },
  secondary: { label: "View research evidence", href: "#evidence" },
  qualification:
    "Research prototype. Designed for investigation and decision support — not diagnosis.",
};

export const problem = {
  lines: ["Health is multimodal.", "Most models are not."],
  paras: [
    "EEG captures neural state. Wearables reveal physiological response. Continuous glucose monitors expose metabolic dynamics. Electronic health records provide clinical context. Molecular data describes biological susceptibility.",
    "Individually, each signal tells part of the story.",
  ],
  connect: "SIGNAL connects the story.",
};

export const worlds = [
  {
    idx: "01", name: "Neural", viz: "neural",
    statement: "Read the brain's response.",
    inputs: ["Galea biosensing streams", "EMOTIV EEG recordings", "Spectral power", "Functional connectivity", "Cognitive-workload markers", "Neural-state embeddings"],
    functions: ["Stress-state characterization", "Cognitive-workload estimation", "Neural-response monitoring", "Depression & anxiety risk research", "Cross-modal association analysis"],
  },
  {
    idx: "02", name: "Physiological", viz: "physio",
    statement: "Measure what the body doesn't say.",
    inputs: ["Heart rate", "Heart-rate variability", "Electrodermal activity", "Respiration", "Skin temperature", "Motion & activity", "Sleep-related signals"],
    functions: ["Autonomic stress detection", "Recovery-state analysis", "Physiological anomaly identification", "Behavioral-context modeling", "Individual baseline comparison"],
  },
  {
    idx: "03", name: "Metabolic", viz: "metabolic",
    statement: "Follow glucose as a trajectory — not a number.",
    inputs: ["Continuous glucose monitoring", "Glucose trend & velocity", "Glucose variability", "Time in range", "Meal & activity context", "Diabetes-related biomarkers"],
    functions: ["Glucose-instability estimation", "Short-horizon forecasting", "Hypo- / hyperglycemia risk research", "Personalized deviation detection", "Metabolic-stress analysis"],
    note: "Today: CGM-only forecasting, validated. Synchronized EEG–wearable–CGM fusion is a future gate — not yet claimed.",
  },
  {
    idx: "04", name: "Clinical", viz: "clinical",
    statement: "Put every signal in context.",
    inputs: ["Diagnoses", "Medications", "Laboratory measurements", "Procedures", "Clinical notes", "Patient history", "Longitudinal encounters"],
    functions: ["Clinical-context embeddings", "Temporal event representation", "Risk-factor extraction", "Patient-history summarization", "Evidence-grounded explanation"],
  },
  {
    idx: "05", name: "Molecular", viz: "molecular", frontier: true,
    statement: "Understand the biology beneath the signal.",
    inputsLabel: "Potential inputs",
    inputs: ["Gene-expression profiles", "Genomic variants", "Proteomic measurements", "Pathway-level representations", "Multi-omics embeddings"],
    functionsLabel: "Research objective",
    functions: ["Investigate whether molecular context can improve personalization, stratification, and precision-risk modeling beyond physiological and clinical measurements alone."],
    note: "Not yet an operational, patient-facing capability. Shown as a research direction, not a claim.",
  },
];

export const engineSteps = [
  { n: "01", title: "Capture", sig: "capture", body: "Receive neural, physiological, metabolic, clinical & molecular observations through standardized interfaces.", items: ["Device adapters", "Batch ingestion", "Streaming-ready", "Schema validation", "Signal-quality checks"] },
  { n: "02", title: "Align", sig: "align", body: "Organize observations on a common patient timeline while blocking future information from past predictions.", items: ["Timestamp normalization", "Patient-level alignment", "Causal cutoffs", "Window construction", "Missingness tracking"] },
  { n: "03", title: "Encode", sig: "encode", body: "Convert each modality into a representation suited to its underlying structure.", items: ["EEG signal encoder", "Wearable time-series", "CGM temporal encoder", "Clinical language", "Structured EHR · omics"] },
  { n: "04", title: "Fuse", sig: "fuse", body: "Fusion is an experimental choice, not a single magical model — many strategies, one honest comparison.", items: ["Weighted late fusion", "Confidence-aware", "Cross-modal attention", "Missing-aware fusion", "Single-modality fallback"] },
  { n: "05", title: "Predict", sig: "predict", body: "Task-specific heads emit calibrated risk with explicit confidence and uncertainty.", items: ["Stress · workload", "Glucose trajectory", "Glucose instability", "Mental-health risk", "Confidence · uncertainty"] },
  { n: "06", title: "Explain", sig: "explain", body: "Translate model evidence into a research report — without letting the language model invent the prediction.", items: ["Evidence-grounded text", "Signal attribution", "Missing-data disclosure", "Model provenance", "Abstention rationale"] },
];

export const fusionStrategies = ["Weighted late fusion", "Confidence-aware aggregation", "Intermediate fusion", "Cross-modal attention", "Missing-modality-aware", "Single-modality fallback"];
export const fusionNote = "Fusion is evaluated against the strongest unimodal baseline. It is retained only when it provides measurable, reproducible value — and on our real cohorts, learned fusion currently does not clear that bar. We report that openly below.";

export const features = [
  { n: "01", size: "big", title: "Temporal by design", body: "Every prediction is anchored to an explicit cutoff. Future observations can never influence an earlier prediction — leakage is a design constraint, not an afterthought." },
  { n: "02", size: "med", title: "Modality-aware", body: "Each source is processed by its native structure, not flattened into one table." },
  { n: "03", size: "sm", title: "Missing-data resilient", body: "Unavailable signals are identified; the path adapts and never invents evidence." },
  { n: "04", size: "med", title: "Confidence before conclusion", body: "Predictions carry uncertainty, calibration, data-quality indicators, and abstention conditions." },
  { n: "05", size: "med", title: "Evidence-grounded language", body: "The language model communicates structured model evidence. It does not independently calculate clinical risk." },
  { n: "06", size: "med", title: "Artifact-backed inference", body: "Production inference loads versioned, committed model artifacts rather than retraining during a request." },
  { n: "07", size: "big", title: "Traceable research outputs", body: "Each result can retain its full lineage.", tags: ["model version", "dataset lineage", "input window", "prediction horizon", "data cutoff", "missing modalities", "quality checks", "explanation provenance"] },
  { n: "08", size: "big", title: "Privacy-aware architecture", body: "Access is governed, scoped, and auditable.", tags: ["consent verification", "role-based access", "patient isolation", "tenant isolation", "minimum-necessary data", "auditability"], footnote: "Architectural controls — not a claim of HIPAA certification." },
];

export const stories = [
  { idx: "Story 01", title: "When stress becomes physiology", body: "Explore how neural activity, autonomic response, respiration, and behavior may jointly characterize an evolving stress state — and when a single modality is enough on its own.", audience: ["Behavioral-health researchers", "Cognitive scientists", "Human-performance teams", "Digital-health investigators"] },
  { idx: "Story 02", title: "When the number isn't the trajectory", body: "Investigate whether physiological context can improve early glucose-instability detection beyond CGM history alone — tested against a strong forecasting baseline, not assumed.", audience: ["Diabetes researchers", "Endocrinology investigators", "Remote-monitoring teams", "Digital-biomarker researchers"] },
  { idx: "Story 03", title: "When the clinical record is not enough", body: "Combine structured observations, clinical language, biosignals, and longitudinal measurements to construct a richer research representation of patient state.", audience: ["Clinical-informatics researchers", "Health-system innovation groups", "Translational investigators", "Precision-health teams"] },
];

// VERIFIED metrics — see source per row.
export const metrics = [
  { cap: "Depression · MDD vs healthy, resting EEG", value: "0.961", detail: "AUROC · 95% CI 0.942–0.976 · real LaBraM", pct: 96.1, src: "docs/MODEL_CARD.md" },
  { cap: "Acute stress · wearable physiology", value: "0.955", detail: "AUROC · 95% CI 0.930–0.978 · band-power+GBM", pct: 95.5, src: "WESAD · subject-held-out" },
  { cap: "Clinical notes · surgery vs rest", value: "0.910", detail: "AUROC · Bio_ClinicalBERT · MTSamples 4,499", pct: 91.0, src: "clinical_notes_scoreboard" },
  { cap: "Glucose · 30-min forecast (CGMacros)", value: "10.27", unit: " mg/dL", detail: "MAE · coverage 0.877 (target 0.90) · CGM-only", pct: 70, src: "glucose_forecast_cgmacros" },
];

export const evidenceCategories = [
  { title: "Benchmarking", items: ["Compare against unimodal baselines", "Compare simple and learned fusion", "Evaluate across prediction horizons", "Report confidence intervals", "Report calibration — not only accuracy"] },
  { title: "Ablation", items: ["EEG removed · wearables removed", "CGM removed · clinical context removed", "Fusion replaced by unimodal model", "Explanation layer removed", "Personalization disabled"] },
  { title: "Robustness", items: ["Missing modality · noisy sensor", "Reduced observation window", "Distribution shift · new participant", "Device variation", "Delayed measurements"] },
  { title: "Responsible failure", items: ["Insufficient data · unsupported task", "Missing artifact", "Out-of-distribution input", "Low confidence", "No synchronized cohort"] },
];

export const honestNegative = {
  lead: "A complex multimodal model is not successful merely because it is more sophisticated. On our real cohorts, under leakage-controlled evaluation, learned cross-modal fusion does not beat the strongest simpler baseline. We report it plainly — that is the standard the science is held to.",
  rows: [
    { task: "Stress", metric: "1 − AUROC", baseline: "0.108 (concat)", fusion: "0.129", rer: "−19.9%" },
    { task: "Glucose", metric: "MAE mg/dL", baseline: "10.66 (concat)", fusion: "13.09", rer: "−22.8%" },
    { task: "Mortality", metric: "1 − AUROC", baseline: "0.178", fusion: "0.360", rer: "−101.7%" },
  ],
  holds: [
    { v: "~35%", l: "What does hold up: multimodality beats the best single modality on stress — via straightforward concatenation, not learned fusion." },
    { v: "~17%", l: "A simple learned model beats 30-minute glucose persistence — modest, significant, honest." },
  ],
  thesis: "The contribution is not another fusion architecture. It is turning a rigorously honest benchmark into a live, evidence-forward research instrument — one that abstains when the evidence isn't there.",
};

export const roadmap = [
  { state: "Completed / active", cls: "done", ph: "Phase 01", title: "Foundation", items: ["Standardized modality schemas", "Real public-dataset pipelines", "Single-modality benchmarks", "FastAPI research wrapper", "Versioned artifacts", "Leakage-aware evaluation", "Evidence-grounded reports"] },
  { state: "Next experimental gate", cls: "next", ph: "Phase 02", title: "Synchronization", items: ["Same-subject EEG + wearable + CGM acquisition", "Shared timestamps & protocol", "Participant-level train/test separation", "Device-quality controls", "Multimodal alignment validation"] },
  { state: "Scientific question", cls: "", ph: "Phase 03", title: "Incremental value", question: "Does neural or wearable context improve prediction beyond the strongest unimodal model?", items: ["Prespecified hypotheses", "Ablation matrix", "Negative controls", "Calibration evaluation", "Statistical uncertainty"] },
  { state: "Adaptation", cls: "", ph: "Phase 04", title: "Personalization", items: ["Individual baselines", "Subject adaptation", "Longitudinal calibration", "Drift detection", "Personalized uncertainty"] },
  { state: "Translation", cls: "", ph: "Phase 05", title: "Translation", items: ["Prospective evaluation", "Clinician-centered usability", "Human factors", "Ethical review", "Deployment governance"] },
];

export const researcher = {
  initials: "AJ",
  name: "Abhignya Jagathpally",
  statement: "Research at the intersection of signals, language, and health.",
  bio: "Abhignya Jagathpally is a PhD researcher in Information Science with a Data Science concentration. His work investigates multimodal learning, longitudinal health intelligence, biosignal analytics, clinical language models, and evidence-grounded AI systems.",
  pillars: [
    { title: "Multimodal learning", body: "Connecting heterogeneous biomedical signals." },
    { title: "Temporal intelligence", body: "Modeling health as an evolving state, not a static snapshot." },
    { title: "Responsible translation", body: "Systems that communicate uncertainty and abstain when evidence is insufficient." },
  ],
};

export const campaign = {
  l1: "The body is always speaking.",
  l2: ["The question is whether", "we can listen responsibly."],
  cta: { label: "Explore the prototype", href: "#experience" },
};

export const disclaimer =
  "A multimodal health-intelligence research prototype. Not intended for diagnosis, treatment, or emergency decision-making. Research-grade decision-support only — a raised risk is a prompt to consult a qualified clinician, never a conclusion. All demonstrations use synthetic or sample data; validated figures apply to their stated research cohorts, not arbitrary recordings.";
