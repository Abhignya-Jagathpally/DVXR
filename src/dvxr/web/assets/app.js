"use strict";

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const state = {
  config: null,
  authenticated: false,
  report: null,
  lastRequest: null,
};

const elements = {
  form: $("#riskForm"),
  reportShell: $("#reportShell"),
  reportEmpty: $("#reportEmpty"),
  loading: $("#loadingState"),
  reportView: $("#reportView"),
  error: $("#errorState"),
  accessDialog: $("#accessDialog"),
  accessForm: $("#accessForm"),
  signOut: $("#signOutButton"),
  connection: $("#connectionState"),
  toast: $("#toast"),
};

const modalityLabels = {
  cgm: "Continuous glucose monitoring",
  eeg: "EEG / neural state",
  wearable_phys: "Wearable physiology",
  wearable: "Wearable physiology",
  ehr: "Clinical context",
  behavior: "Behavioral context",
  omics: "Multi-omics context",
};

const reportHints = {
  stress_glucose_risk: "Research fusion is expected to abstain until synchronized same-subject data and a validated artifact exist.",
  glucose_risk: "Returns a calibrated CGM excursion risk only when an approved, registered CGM artifact and admissible patient history are available.",
  cgm_glucose_forecast: "Returns a CGM-only point forecast and interval only when a registered forecasting artifact is provisioned.",
};

const illustrativeReport = {
  illustrative: true,
  request_id: "req_illustrative_ui_only",
  prediction_id: "pred_illustrative_ui_only",
  status: "completed",
  reused: false,
  grounding_complete: true,
  retrieval_status: "complete",
  protocol_grounding_complete: true,
  model_version: "illustrative-interface/no-model-executed",
  feature_version: "illustrative-interface",
  disclaimer: "Illustrative interface data. No live model was executed. Research-grade decision support, not a diagnosis.",
  prediction: {
    patient_id: "PSEUDO-DEMO",
    report_type: "glucose_risk",
    risk: { excursion_30m: 0.38, excursion_60m: 0.57 },
    risk_category: "elevated",
    confidence: 0.81,
    data_quality: "acceptable",
    missing_modalities: ["eeg"],
    stale_modalities: [],
    abstained: false,
    abstain_reason: null,
    model_version: "illustrative-interface/no-model-executed",
    feature_version: "illustrative-interface",
    calibration_version: "illustrative-only",
    data_cutoff_at: "2026-07-16T09:00:00+00:00",
    snapshot_id: "snap_illustrative_ui_only",
    prediction_id: "pred_illustrative_ui_only",
    forecast: {
      glucose_30m: { point: 142, lower: 124, upper: 161 },
      glucose_60m: { point: 158, lower: 129, upper: 188 },
    },
  },
  evidence: {
    contributions: { cgm: 0.44, wearable_phys: 0.17, ehr: 0.08 },
    modality_quality: { cgm: 0.94, wearable_phys: 0.82, ehr: 0.71 },
    missing_data_effects: ["EEG was unavailable; the report does not make a neural-state claim."],
    uncertainty: 0.19,
    evidence_records: [
      { evidence_id: "ev_demo_1", feature: "recent CGM slope", value: 0.44, method: "illustrative" },
      { evidence_id: "ev_demo_2", feature: "autonomic stress representation", value: 0.17, method: "illustrative" },
    ],
  },
  action: {
    action_id: "VERIFY_AND_CONTINUE_MONITORING",
    policy_id: "neuroglycemic-policy",
    policy_version: "illustrative",
    reason_codes: ["ELEVATED_RISK", "ACCEPTABLE_DATA", "RESEARCH_REVIEW"],
    requires_clinician_review: false,
    system_action_id: "CONTINUE_MONITORING",
  },
  explanation: {
    risk_summary: "Illustrative glucose-excursion risk — 30 minutes: 38%; 60 minutes: 57%.",
    prediction_horizon_minutes: [30, 60],
    supporting_factors: [
      { statement: "Recent CGM dynamics contribute most strongly to the illustrative result.", source_id: "ev_demo_1" },
      { statement: "Wearable physiology adds contextual evidence, but EEG is unavailable.", source_id: "ev_demo_2" },
    ],
    missing_or_stale_data: ["eeg"],
    uncertainty_statement: "Illustrative confidence 0.81. This value was created only to demonstrate the interface.",
    action_id: "VERIFY_AND_CONTINUE_MONITORING",
    action_explanation: "Verify current data quality and continue the approved monitoring workflow. This interface does not prescribe treatment.",
    citations: [],
    limitations: [
      "Illustrative data only; no patient data or model inference was used.",
      "Research-grade decision support, not a diagnosis.",
    ],
  },
};

function setVisible(active) {
  for (const [name, node] of Object.entries({ empty: elements.reportEmpty, loading: elements.loading, report: elements.reportView, error: elements.error })) {
    node.hidden = name !== active;
  }
  elements.reportShell.setAttribute("aria-busy", active === "loading" ? "true" : "false");
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("show");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => elements.toast.classList.remove("show"), 2400);
}

function setConnection(label, kind = "ok") {
  elements.connection.className = `connection-state ${kind === "ok" ? "" : kind}`.trim();
  $("b", elements.connection).textContent = label;
}

function formatPercent(value) {
  return Number.isFinite(Number(value)) ? `${Math.round(Number(value) * 100)}%` : "—";
}

function formatNumber(value, digits = 2) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—";
}

function humanize(value) {
  return String(value || "").replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function formatDate(value) {
  if (!value) return "Not reported";
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? value : new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(date);
}

async function jsonFetch(url, options = {}) {
  const response = await fetch(url, {
    credentials: "same-origin",
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  let payload = {};
  try { payload = await response.json(); } catch { payload = {}; }
  if (!response.ok) {
    const error = new Error(payload.error || payload.detail || `Request failed (${response.status})`);
    error.status = response.status;
    throw error;
  }
  return payload;
}

async function bootstrap() {
  try {
    const [config, session] = await Promise.all([
      jsonFetch("/ui/config", { headers: {} }),
      jsonFetch("/ui/session", { headers: {} }),
    ]);
    state.config = config;
    state.authenticated = Boolean(session.authenticated);
    setConnection("Service available");
    elements.signOut.hidden = !state.authenticated || config.unsafe_dev;
  } catch (error) {
    setConnection("Service unavailable", "error");
    console.error(error);
  }

  try {
    const health = await fetch("/health", { credentials: "same-origin" });
    if (!health.ok) throw new Error("health check failed");
  } catch {
    setConnection("API health unavailable", "offline");
  }
}

function ensureAccess() {
  if (!state.config?.auth_required || state.authenticated) return true;
  elements.accessDialog.showModal();
  setTimeout(() => $("#accessCode").focus(), 50);
  return false;
}

async function submitAccess(event) {
  event.preventDefault();
  const errorNode = $("#dialogError");
  const button = $("#accessButton");
  errorNode.hidden = true;
  button.disabled = true;
  try {
    await jsonFetch("/ui/session", { method: "POST", body: JSON.stringify({ access_code: $("#accessCode").value }) });
    state.authenticated = true;
    elements.signOut.hidden = false;
    elements.accessDialog.close();
    $("#accessCode").value = "";
    showToast("Research workspace unlocked");
    if (state.lastRequest === "generate") elements.form.requestSubmit();
  } catch (error) {
    errorNode.textContent = error.message;
    errorNode.hidden = false;
  } finally {
    button.disabled = false;
  }
}

async function signOut() {
  try { await jsonFetch("/ui/session", { method: "DELETE" }); } catch { /* local state still clears */ }
  state.authenticated = false;
  elements.signOut.hidden = true;
  showToast("Signed out");
}

function requestPayload() {
  const horizons = $$('input[name="horizon"]:checked').map((node) => Number(node.value));
  if (!horizons.length) throw new Error("Select at least one prediction horizon.");
  const patient = $("#patientId").value.trim();
  if (!patient) throw new Error("Enter a pseudonymous patient identifier.");
  const nonce = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  return {
    patient_id: patient,
    report_type: $("#reportType").value,
    prediction_horizons_minutes: horizons,
    question: $("#question").value.trim() || null,
    idempotency_key: `web-${nonce}`,
  };
}

async function generateReport(event) {
  event.preventDefault();
  state.lastRequest = "generate";
  if (!ensureAccess()) return;

  let payload;
  try { payload = requestPayload(); }
  catch (error) { showError("Review the request", error.message); return; }

  setVisible("loading");
  $("#generateButton").disabled = true;
  try {
    const report = await jsonFetch("/v1/risk-reports", { method: "POST", body: JSON.stringify(payload) });
    state.report = report;
    renderReport(report);
  } catch (error) {
    if (error.status === 401) {
      state.authenticated = false;
      elements.signOut.hidden = true;
      setVisible("empty");
      ensureAccess();
      return;
    }
    const titles = {
      403: "Authorization or consent was not verified",
      409: "This request conflicts with an earlier review",
      500: "The report service did not complete the request",
    };
    showError(titles[error.status] || "The review could not be generated", error.message);
  } finally {
    $("#generateButton").disabled = false;
  }
}

function showError(title, message) {
  $("#errorTitle").textContent = title;
  $("#errorMessage").textContent = message;
  setVisible("error");
}

function renderReport(report) {
  const pred = report.prediction || {};
  const explanation = report.explanation || {};
  const evidence = report.evidence || {};
  const action = report.action || {};
  const abstained = Boolean(pred.abstained || report.status === "abstained");
  const category = pred.risk_category || (abstained ? "abstained" : "not reported");

  $("#illustrativeBanner").hidden = !report.illustrative;
  $("#reportStatusChip").textContent = abstained ? "Abstained safely" : humanize(category);
  $("#reportTitle").textContent = `${humanize(pred.report_type || "risk review")} · ${pred.patient_id || "Unknown patient"}`;
  $("#reportSubtitle").textContent = `Data cutoff ${formatDate(pred.data_cutoff_at)}`;

  const statusPanel = $("#statusPanel");
  statusPanel.className = `status-panel ${abstained ? "abstained" : category === "high" ? "high" : ""}`.trim();
  $("#statusEyebrow").textContent = abstained ? "SAFE ABSTENTION" : "CURRENT MODEL STATE";
  $("#statusHeadline").textContent = abstained ? "No risk estimate was issued." : `${humanize(category)} risk`;
  $("#statusSummary").textContent = abstained
    ? (pred.abstain_reason || explanation.risk_summary || "The available evidence did not satisfy the requirements for a reliable estimate.")
    : (explanation.risk_summary || "A calibrated prediction is available.");

  const confidence = pred.confidence;
  $("#confidenceValue").textContent = confidence == null ? "—" : formatPercent(confidence);
  $("#confidenceRing").style.setProperty("--confidence", `${Math.max(0, Math.min(100, Number(confidence || 0) * 100))}%`);

  renderMetrics(pred, abstained);
  renderModalities(pred, evidence);
  renderSupporting(explanation, evidence);
  renderAction(report, action, explanation);
  renderProvenance(report, pred, action, explanation);
  setVisible("report");
  elements.reportShell.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderMetrics(pred, abstained) {
  const cards = [];
  const risks = pred.risk || {};
  for (const [key, value] of Object.entries(risks)) {
    const horizon = key.match(/(\d+)m/)?.[1];
    cards.push({ label: horizon ? `${horizon}-minute excursion risk` : humanize(key), value: abstained ? "—" : formatPercent(value), note: "Calibrated probability" });
  }
  for (const [key, value] of Object.entries(pred.forecast || {})) {
    const horizon = key.match(/(\d+)m/)?.[1];
    const point = value?.point;
    const lower = value?.lower;
    const upper = value?.upper;
    cards.push({ label: horizon ? `${horizon}-minute glucose forecast` : humanize(key), value: point == null ? "—" : `${Math.round(point)} mg/dL`, note: lower != null && upper != null ? `Interval ${Math.round(lower)}–${Math.round(upper)} mg/dL` : "Interval not reported" });
  }
  if (!cards.length) {
    cards.push(
      { label: "Risk estimate", value: "Not issued", note: abstained ? "The model abstained" : "No numerical output" },
      { label: "Data quality", value: humanize(pred.data_quality || "unknown"), note: "Input readiness" },
      { label: "Missing modalities", value: String((pred.missing_modalities || []).length), note: "Explicitly represented" },
    );
  }
  $("#metricGrid").innerHTML = cards.slice(0, 6).map((card) => `
    <article class="metric-card"><span>${escapeHtml(card.label)}</span><strong>${escapeHtml(card.value)}</strong><small>${escapeHtml(card.note)}</small></article>
  `).join("");
}

function renderModalities(pred, evidence) {
  const missing = new Set(pred.missing_modalities || []);
  const stale = new Set(pred.stale_modalities || []);
  const quality = evidence.modality_quality || {};
  const keys = new Set(["cgm", "wearable_phys", "eeg", "ehr", ...Object.keys(quality), ...missing, ...stale]);
  $("#modalityList").innerHTML = [...keys].map((key) => {
    const status = missing.has(key) ? "missing" : stale.has(key) ? "stale" : "available";
    const detail = quality[key] == null ? humanize(status) : `${formatPercent(quality[key])} quality`;
    return `<div class="modality-row ${status}"><span><i></i>${escapeHtml(modalityLabels[key] || humanize(key))}</span><b>${escapeHtml(detail)}</b></div>`;
  }).join("");
  $("#dataQualityNote").textContent = `Overall data quality: ${humanize(pred.data_quality || "unknown")}. Missing and stale inputs are not silently imputed by the interface.`;
}

function renderSupporting(explanation, evidence) {
  const factors = explanation.supporting_factors || [];
  const fallback = Object.entries(evidence.contributions || {}).map(([key, value]) => ({ statement: `${humanize(key)} contribution: ${formatNumber(value, 2)}.` }));
  const displayed = factors.length ? factors : fallback;
  $("#supportingFactors").innerHTML = displayed.length
    ? displayed.map((factor) => `<div class="support-item">${escapeHtml(typeof factor === "string" ? factor : factor.statement || JSON.stringify(factor))}</div>`).join("")
    : '<div class="support-item">No contribution narrative was issued.</div>';
  $("#uncertaintyStatement").textContent = explanation.uncertainty_statement || "Uncertainty was not reported.";
}

function renderAction(report, action, explanation) {
  $("#actionHeadline").textContent = humanize(action.action_id || "No action issued");
  $("#actionExplanation").textContent = explanation.action_explanation || "The policy engine did not return explanatory action text.";
  $("#reasonCodes").innerHTML = (action.reason_codes || []).map((code) => `<span>${escapeHtml(humanize(code))}</span>`).join("");

  const buttonWrap = $("#actionButtons");
  buttonWrap.innerHTML = "";
  if (report.illustrative || !report.prediction_id) return;
  const actions = [
    ["Acknowledge", "acknowledge", "button-secondary"],
    ["Escalate", "escalate", "button-primary"],
    ["Dismiss", "dismiss", "button-ghost"],
  ];
  for (const [label, operation, style] of actions) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `button ${style}`;
    button.textContent = label;
    button.addEventListener("click", () => updateAlert(report.prediction_id, operation, button));
    buttonWrap.append(button);
  }
}

async function updateAlert(predictionId, operation, button) {
  button.disabled = true;
  try {
    const payload = await jsonFetch(`/v1/alerts/${encodeURIComponent(predictionId)}/${operation}`, { method: "POST", body: JSON.stringify({ note: "Updated from Sentinel web interface" }) });
    showToast(`Alert ${payload.alert?.state || operation}`);
  } catch (error) {
    showToast(error.message);
  } finally {
    button.disabled = false;
  }
}

function renderProvenance(report, pred, action, explanation) {
  const items = [
    ["Request ID", report.request_id],
    ["Prediction ID", report.prediction_id],
    ["Snapshot ID", pred.snapshot_id],
    ["Model", report.model_version || pred.model_version],
    ["Feature version", report.feature_version || pred.feature_version],
    ["Calibration", pred.calibration_version],
    ["Policy", `${action.policy_id || "—"} ${action.policy_version || ""}`.trim()],
    ["Retrieval", report.retrieval_status || "disabled"],
    ["Grounding complete", String(Boolean(report.grounding_complete))],
    ["Data cutoff", pred.data_cutoff_at],
  ];
  $("#provenanceGrid").innerHTML = items.map(([label, value]) => `<div class="provenance-item"><span>${escapeHtml(label)}</span><code title="${escapeHtml(value || "Not reported")}">${escapeHtml(value || "Not reported")}</code></div>`).join("");
  const limitations = [...(explanation.limitations || []), report.disclaimer].filter(Boolean);
  $("#limitations").innerHTML = limitations.map((item) => `<p>• ${escapeHtml(item)}</p>`).join("");
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[character]));
}

function openIllustrativeReport() {
  state.report = structuredClone(illustrativeReport);
  renderReport(state.report);
}

function copyPredictionId() {
  const id = state.report?.prediction_id;
  if (!id) return;
  navigator.clipboard?.writeText(id).then(() => showToast("Prediction ID copied"), () => showToast(id));
}

function updateReportHint() {
  $("#reportTypeHint").textContent = reportHints[$("#reportType").value] || "";
}

elements.form.addEventListener("submit", generateReport);
elements.accessForm.addEventListener("submit", submitAccess);
elements.signOut.addEventListener("click", signOut);
$("#sampleReportButton").addEventListener("click", openIllustrativeReport);
$("#copyReportId").addEventListener("click", copyPredictionId);
$("#reportType").addEventListener("change", updateReportHint);
$("#retryButton").addEventListener("click", () => setVisible("empty"));
elements.accessDialog.addEventListener("close", () => { if (!state.authenticated) state.lastRequest = null; });

bootstrap();
