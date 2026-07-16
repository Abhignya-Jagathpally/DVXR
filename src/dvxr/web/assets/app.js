"use strict";

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

// Served same-origin by the FastAPI app: default the engine to this deployment's own origin, so the
// UI calls /ui/token, /v1, /health on the same host with no hard-coded deployment URL. A different
// engine can still be entered under Engine settings for a cross-origin artifact host.
const DEFAULT_API = (typeof window !== "undefined" && window.location && window.location.origin) || "";
const STORAGE_KEY = "ngs_artifact_settings_v1";
const TOKEN_KEY = "ngs_artifact_token_v1";
const memoryStorage = new Map();

function storageGet(area, key) {
  try { return window[area]?.getItem(key); } catch { return memoryStorage.get(`${area}:${key}`) || null; }
}

function storageSet(area, key, value) {
  try { window[area]?.setItem(key, value); } catch { memoryStorage.set(`${area}:${key}`, value); }
}

function storageRemove(area, key) {
  try { window[area]?.removeItem(key); } catch { memoryStorage.delete(`${area}:${key}`); }
}

const state = {
  view: "overview",
  mode: "live",
  apiBase: DEFAULT_API,
  timeout: 30000,
  autoDemo: true,
  token: storageGet("sessionStorage", TOKEN_KEY) || "",
  connected: false,
  report: null,
  lastRequest: null,
  loadingTimer: null,
};

const reportHints = {
  stress_glucose_risk: "Requires synchronized same-subject evidence and a registered fusion artifact.",
  glucose_risk: "Requires an approved CGM risk artifact and admissible recent glucose history.",
  cgm_glucose_forecast: "Requires an approved forecasting artifact and a valid causal CGM window.",
};

const modalityNames = {
  cgm: "Continuous glucose monitoring",
  eeg: "EEG / neural state",
  wearable: "Wearable physiology",
  wearable_phys: "Wearable physiology",
  ehr: "Clinical context",
  behavior: "Behavioral context",
  omics: "Multi-omics context",
};

const demoReport = {
  illustrative: true,
  request_id: "req_illustrative_ngs_2026",
  prediction_id: "pred_illustrative_ngs_2026",
  status: "completed",
  grounding_complete: true,
  retrieval_status: "complete",
  protocol_grounding_complete: true,
  model_version: "illustrative-interface/no-model-executed",
  feature_version: "illustrative-interface",
  disclaimer: "Illustrative interface data. No patient data or live model inference was used.",
  prediction: {
    patient_id: "PSEUDO-DEMO",
    report_type: "stress_glucose_risk",
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
    data_cutoff_at: "2026-07-16T09:00:00-05:00",
    snapshot_id: "snap_illustrative_ngs_2026",
    prediction_id: "pred_illustrative_ngs_2026",
    forecast: {
      glucose_30m: { point: 142, lower: 124, upper: 161 },
      glucose_60m: { point: 158, lower: 129, upper: 188 },
    },
  },
  evidence: {
    contributions: { cgm: 0.44, wearable_phys: 0.17, ehr: 0.08 },
    modality_quality: { cgm: 0.94, wearable_phys: 0.82, ehr: 0.71 },
    missing_data_effects: ["EEG was unavailable; no neural-state claim was made."],
    uncertainty: 0.19,
    evidence_records: [
      { evidence_id: "ev_demo_cgm_slope", feature: "recent CGM slope", value: 0.44, method: "illustrative" },
      { evidence_id: "ev_demo_autonomic", feature: "autonomic stress representation", value: 0.17, method: "illustrative" },
    ],
  },
  action: {
    action_id: "VERIFY_AND_CONTINUE_MONITORING",
    policy_id: "neuroglycemic-policy",
    policy_version: "illustrative-v1",
    reason_codes: ["ELEVATED_RISK", "ACCEPTABLE_DATA", "EEG_UNAVAILABLE"],
    requires_clinician_review: false,
    system_action_id: "CONTINUE_MONITORING",
  },
  explanation: {
    risk_summary: "Illustrative glucose-excursion risk is 38% at 30 minutes and 57% at 60 minutes.",
    prediction_horizon_minutes: [30, 60],
    supporting_factors: [
      { statement: "The recent glucose trajectory is the largest contributor to the illustrative result.", source_id: "ev_demo_cgm_slope" },
      { statement: "Wearable physiology adds autonomic context, while EEG is explicitly marked unavailable.", source_id: "ev_demo_autonomic" },
      { statement: "The 60-minute forecast interval widens, indicating increasing uncertainty with horizon.", source_id: "pred_illustrative_ngs_2026" },
    ],
    missing_or_stale_data: ["eeg"],
    uncertainty_statement: "Illustrative confidence is 81%. The wider 60-minute interval and missing EEG reduce certainty.",
    action_id: "VERIFY_AND_CONTINUE_MONITORING",
    action_explanation: "Verify current sensor status and continue the approved monitoring workflow. This interface does not prescribe medication or treatment.",
    citations: [],
    limitations: [
      "Illustrative data only; no patient data or live model was used.",
      "The multimodal fusion claim requires synchronized same-subject validation.",
      "Research-stage decision support, not a diagnosis or treatment recommendation.",
    ],
  },
};

function safeJsonParse(value, fallback) {
  try {
    const parsed = JSON.parse(value);
    return parsed && typeof parsed === "object" ? parsed : fallback;
  } catch { return fallback; }
}

function loadSettings() {
  const saved = safeJsonParse(storageGet("localStorage", STORAGE_KEY), {});
  state.apiBase = String(saved.apiBase || DEFAULT_API).replace(/\/$/, "");
  state.timeout = Number(saved.timeout || 30000);
  state.autoDemo = saved.autoDemo !== false;
  $("#apiEndpoint").value = state.apiBase;
  $("#settingsEndpoint").value = state.apiBase;
  $("#requestTimeout").value = String(state.timeout);
  $("#autoDemo").checked = state.autoDemo;
}

function saveSettings() {
  storageSet("localStorage", STORAGE_KEY, JSON.stringify({
    apiBase: state.apiBase,
    timeout: state.timeout,
    autoDemo: state.autoDemo,
  }));
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  }[character]));
}

function humanize(value) {
  return String(value || "").replaceAll("_", " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function formatPercent(value) {
  return Number.isFinite(Number(value)) ? `${Math.round(Number(value) * 100)}%` : "—";
}

function formatDate(value) {
  if (!value) return "Not reported";
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) return String(value);
  return new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(date);
}

function showToast(message) {
  const node = $("#toast");
  node.textContent = message;
  node.classList.add("show");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => node.classList.remove("show"), 2600);
}

function setService(stateName, label) {
  const pill = $("#servicePill");
  pill.dataset.state = stateName;
  $("#serviceLabel").textContent = label;
  $("#openAccess").textContent = state.connected ? "Connected" : "Connect";
}

function navigate(view) {
  state.view = view;
  $$("[data-view-panel]").forEach((panel) => panel.classList.toggle("active", panel.dataset.viewPanel === view));
  $$(".nav-link").forEach((button) => button.classList.toggle("active", button.dataset.view === view));
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function setMode(mode) {
  state.mode = mode;
  $$("[data-mode]").forEach((button) => button.classList.toggle("active", button.dataset.mode === mode));
  if (mode === "demo") setService("demo", "Guided demonstration");
  else checkHealth();
}

function timeoutFetch(url, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), state.timeout);
  return fetch(url, { ...options, signal: controller.signal }).finally(() => clearTimeout(timer));
}

async function requestJson(path, options = {}) {
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  const response = await timeoutFetch(`${state.apiBase}${path}`, { mode: "cors", ...options, headers });
  let payload = {};
  try { payload = await response.json(); } catch { payload = {}; }
  if (!response.ok) {
    const error = new Error(payload.detail || payload.error || `Request failed (${response.status})`);
    error.status = response.status;
    throw error;
  }
  return payload;
}

async function checkHealth() {
  if (state.mode === "demo") return;
  setService("checking", "Checking engine");
  try {
    const response = await timeoutFetch(`${state.apiBase}/health`, { mode: "cors", headers: { Accept: "application/json" } });
    if (!response.ok) throw new Error("Health endpoint unavailable");
    state.connected = Boolean(state.token);
    setService("online", state.token ? "Engine connected" : "Engine available");
  } catch (error) {
    state.connected = false;
    setService("offline", "Engine unavailable");
  }
}

async function connectEngine(event) {
  event.preventDefault();
  const endpoint = $("#apiEndpoint").value.trim().replace(/\/$/, "");
  const accessCode = $("#accessCode").value;
  const button = $("#connectButton");
  const message = $("#accessMessage");
  message.hidden = true;
  button.disabled = true;
  button.textContent = "Establishing session…";

  try {
    state.apiBase = endpoint;
    const response = await timeoutFetch(`${state.apiBase}/ui/token`, {
      method: "POST",
      mode: "cors",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ access_code: accessCode }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || payload.error || `Connection failed (${response.status})`);
    if (!payload.access_token) throw new Error("The engine did not return a session token. Apply the included FastAPI artifact bridge.");

    state.token = payload.access_token;
    state.connected = true;
    storageSet("sessionStorage", TOKEN_KEY, state.token);
    saveSettings();
    $("#accessCode").value = "";
    $("#accessDialog").close();
    setMode("live");
    setService("online", "Engine connected");
    showToast(`Secure session established for ${Math.round((payload.expires_in || 900) / 60)} minutes`);
  } catch (error) {
    message.textContent = error.name === "AbortError" ? "The engine did not respond before the timeout." : error.message;
    message.hidden = false;
  } finally {
    button.disabled = false;
    button.innerHTML = 'Establish secure session <span>→</span>';
  }
}

function clearSecureSession() {
  state.token = "";
  state.connected = false;
  storageRemove("sessionStorage", TOKEN_KEY);
  $("#settingsDialog").close();
  setService("checking", "Checking engine");
  checkHealth();
  showToast("Secure session cleared");
}

function updateReportHint() {
  $("#reportHint").textContent = reportHints[$("#reportType").value] || "";
}

function buildRequest() {
  const patientId = $("#patientId").value.trim();
  const horizons = $$("input[name='horizon']:checked").map((node) => Number(node.value));
  if (!patientId) throw new Error("Enter a pseudonymous participant identifier.");
  if (!horizons.length) throw new Error("Select at least one forecast horizon.");
  const nonce = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  return {
    patient_id: patientId,
    report_type: $("#reportType").value,
    prediction_horizons_minutes: horizons,
    question: $("#question").value.trim() || null,
    idempotency_key: `artifact-${nonce}`,
  };
}

function setStage(name) {
  const mapping = {
    empty: $("#emptyStage"),
    loading: $("#loadingStage"),
    error: $("#errorStage"),
    report: $("#report"),
  };
  Object.entries(mapping).forEach(([key, node]) => { node.hidden = key !== name; });
  $("#reviewStage").setAttribute("aria-busy", name === "loading" ? "true" : "false");
}

function animateLoading() {
  const headlines = [
    "Verifying authorization and signal readiness.",
    "Aligning authorized evidence at one causal cutoff.",
    "Resolving the registered model or abstention contract.",
    "Validating explanation, provenance, and policy action.",
  ];
  let step = 0;
  clearInterval(state.loadingTimer);
  const advance = () => {
    $("#loadingHeadline").textContent = headlines[step];
    $$("#loadingSteps li").forEach((item, index) => item.classList.toggle("active", index === step));
    step = (step + 1) % headlines.length;
  };
  advance();
  state.loadingTimer = setInterval(advance, 900);
}

function stopLoading() {
  clearInterval(state.loadingTimer);
  state.loadingTimer = null;
}

async function generateReview(event) {
  event.preventDefault();
  let payload;
  try { payload = buildRequest(); } catch (error) { return showError("Review the request", error.message); }
  state.lastRequest = payload;
  navigate("review");

  if (state.mode === "demo") {
    loadDemoReport(payload.patient_id, payload.report_type);
    return;
  }

  if (!state.token) {
    $("#accessDialog").showModal();
    setTimeout(() => $("#accessCode").focus(), 40);
    return;
  }

  setStage("loading");
  animateLoading();
  $("#generateReview").disabled = true;
  try {
    const report = await requestJson("/v1/risk-reports", { method: "POST", body: JSON.stringify(payload) });
    state.report = report;
    renderReport(report);
    setService("online", "Engine connected");
  } catch (error) {
    if (error.status === 401) {
      state.token = "";
      state.connected = false;
      storageRemove("sessionStorage", TOKEN_KEY);
    }
    showError(
      error.status === 403 ? "Authorization or consent was not verified" : "The live review did not complete",
      error.name === "AbortError" ? "The engine did not respond before the configured timeout." : error.message,
    );
  } finally {
    stopLoading();
    $("#generateReview").disabled = false;
  }
}

function showError(title, message) {
  stopLoading();
  $("#errorTitle").textContent = title;
  $("#errorMessage").textContent = message;
  setStage("error");
}

function loadDemoReport(patientId = "PSEUDO-DEMO", reportType = "stress_glucose_risk") {
  const report = typeof structuredClone === "function" ? structuredClone(demoReport) : JSON.parse(JSON.stringify(demoReport));
  report.prediction.patient_id = patientId || "PSEUDO-DEMO";
  report.prediction.report_type = reportType;
  state.report = report;
  state.mode = "demo";
  $$("[data-mode]").forEach((button) => button.classList.toggle("active", button.dataset.mode === "demo"));
  setService("demo", "Guided demonstration");
  navigate("review");
  renderReport(report);
}

function renderReport(report) {
  const prediction = report.prediction || {};
  const evidence = report.evidence || {};
  const explanation = report.explanation || {};
  const action = report.action || {};
  const abstained = Boolean(prediction.abstained || report.status === "abstained");
  const category = prediction.risk_category || (abstained ? "abstained" : "not reported");
  const risks = prediction.risk || {};
  const selectedRisk = Number(risks.excursion_60m ?? risks.excursion_30m);
  const riskPercent = Number.isFinite(selectedRisk) && !abstained ? Math.round(selectedRisk * 100) : 0;

  $("#demoRibbon").hidden = !report.illustrative;
  $("#stageContext").textContent = `${humanize(prediction.report_type || "risk review")} · ${prediction.patient_id || "Unknown participant"}`;
  $("#statusChip").textContent = abstained ? "Abstained safely" : humanize(category);
  $("#statusChip").className = `status-chip ${category === "high" ? "high" : abstained ? "abstained" : ""}`.trim();
  $("#reportTypeLabel").textContent = humanize(prediction.report_type || "risk review").toUpperCase();
  $("#reportPatient").textContent = prediction.patient_id || "Unknown participant";
  $("#reportCutoff").textContent = `Data cutoff ${formatDate(prediction.data_cutoff_at)}`;
  $("#riskDial").style.setProperty("--risk", String(riskPercent));
  $("#riskValue").textContent = abstained ? "—" : `${riskPercent}%`;
  $("#riskHorizon").textContent = abstained ? "no estimate issued" : risks.excursion_60m != null ? "60-minute risk" : "30-minute risk";
  $("#statusHeadline").textContent = abstained ? "No risk estimate was issued" : `${humanize(category)} near-term risk`;
  $("#statusSummary").textContent = abstained
    ? (prediction.abstain_reason || explanation.risk_summary || "The available evidence did not satisfy the requirements for a reliable estimate.")
    : (explanation.risk_summary || "A calibrated model result is available.");

  const confidence = Number(prediction.confidence);
  const confidencePercent = Number.isFinite(confidence) ? Math.round(confidence * 100) : 0;
  $("#confidenceValue").textContent = Number.isFinite(confidence) ? `${confidencePercent}%` : "—";
  $("#confidenceBar").style.width = `${confidencePercent}%`;

  renderMetrics(prediction, abstained);
  renderForecast(prediction, abstained);
  renderModalities(prediction, evidence);
  renderEvidence(explanation, evidence);
  renderAction(report, action, explanation);
  renderProvenance(report, prediction, action, explanation);
  setStage("report");
}

function renderMetrics(prediction, abstained) {
  const metrics = [];
  const risk = prediction.risk || {};
  Object.entries(risk).forEach(([key, value]) => {
    const horizon = key.match(/(\d+)m/)?.[1];
    metrics.push({
      label: horizon ? `${horizon}-minute excursion risk` : humanize(key),
      value: abstained ? "Not issued" : formatPercent(value),
      note: abstained ? "Formal abstention" : "Calibrated probability",
    });
  });
  metrics.push({ label: "Model confidence", value: prediction.confidence == null ? "—" : formatPercent(prediction.confidence), note: "Reported confidence" });
  metrics.push({ label: "Data quality", value: humanize(prediction.data_quality || "unknown"), note: "Input readiness" });
  metrics.push({ label: "Unavailable inputs", value: String((prediction.missing_modalities || []).length), note: "Explicitly represented" });
  $("#metricLedger").innerHTML = metrics.slice(0, 4).map((metric) => `
    <div class="metric"><span>${escapeHtml(metric.label)}</span><strong>${escapeHtml(metric.value)}</strong><small>${escapeHtml(metric.note)}</small></div>
  `).join("");
}

function renderForecast(prediction, abstained) {
  const forecast = prediction.forecast || {};
  const entries = Object.entries(forecast);
  $("#trendBadge").textContent = abstained ? "No forecast" : state.report?.illustrative ? "Illustrative" : "Model output";
  if (!entries.length || abstained) {
    $("#forecastReadouts").innerHTML = '<div><span>Forecast</span><b>Not issued</b></div><div><span>Reason</span><b>Model abstention</b></div>';
    $("#forecastLine").style.opacity = ".18";
    $("#forecastBand").style.opacity = ".08";
    $("#point30").style.opacity = "0";
    $("#point60").style.opacity = "0";
    return;
  }
  $("#forecastLine").style.opacity = "1";
  $("#forecastBand").style.opacity = "1";
  $("#point30").style.opacity = "1";
  $("#point60").style.opacity = "1";
  $("#forecastReadouts").innerHTML = entries.slice(0, 2).map(([key, value]) => {
    const horizon = key.match(/(\d+)m/)?.[1] || humanize(key);
    const point = value?.point;
    const interval = value?.lower != null && value?.upper != null ? `${Math.round(value.lower)}–${Math.round(value.upper)} mg/dL interval` : "Interval not reported";
    return `<div><span>${escapeHtml(horizon)} minute forecast</span><b>${point == null ? "—" : `${Math.round(point)} mg/dL`}</b><small>${escapeHtml(interval)}</small></div>`;
  }).join("");
}

function renderModalities(prediction, evidence) {
  const missing = new Set(prediction.missing_modalities || []);
  const stale = new Set(prediction.stale_modalities || []);
  const quality = evidence.modality_quality || {};
  const keys = new Set(["cgm", "wearable_phys", "eeg", "ehr", ...Object.keys(quality), ...missing, ...stale]);
  $("#modalityLedger").innerHTML = [...keys].map((key) => {
    const status = missing.has(key) ? "missing" : stale.has(key) ? "stale" : "available";
    const detail = quality[key] == null ? humanize(status) : `${formatPercent(quality[key])} quality`;
    return `<div class="modality-row ${status}"><span><i></i>${escapeHtml(modalityNames[key] || humanize(key))}</span><b>${escapeHtml(detail)}</b></div>`;
  }).join("");
  $("#dataQualityNote").textContent = `Overall data quality: ${humanize(prediction.data_quality || "unknown")}. Missing and stale inputs remain explicit in the review.`;
}

function renderEvidence(explanation, evidence) {
  const factors = explanation.supporting_factors || [];
  const fallback = Object.entries(evidence.contributions || {}).map(([key, value]) => ({
    statement: `${humanize(key)} contribution: ${Number(value).toFixed(2)}.`,
    source_id: "model-evidence",
  }));
  const items = factors.length ? factors : fallback;
  $("#factorList").innerHTML = items.length
    ? items.map((factor) => `<div class="factor">${escapeHtml(typeof factor === "string" ? factor : factor.statement || JSON.stringify(factor))}<small>${escapeHtml(factor.source_id || "Structured model evidence")}</small></div>`).join("")
    : '<div class="factor">No contribution narrative was issued.<small>Explanation unavailable</small></div>';
  $("#uncertaintyStatement").textContent = explanation.uncertainty_statement || "Uncertainty was not reported.";
}

function renderAction(report, action, explanation) {
  $("#actionTitle").textContent = humanize(action.action_id || "No action issued");
  $("#actionExplanation").textContent = explanation.action_explanation || "The policy engine did not return an explanatory action statement.";
  $("#reasonCodes").innerHTML = (action.reason_codes || []).map((code) => `<span>${escapeHtml(humanize(code))}</span>`).join("");
  const wrap = $("#actionButtons");
  wrap.innerHTML = "";
  if (report.illustrative || !report.prediction_id) {
    wrap.innerHTML = '<button class="secondary-button" type="button" disabled>Illustrative action only</button>';
    return;
  }
  [["Acknowledge", "acknowledge", "secondary-button"], ["Escalate", "escalate", "primary-button"], ["Dismiss", "dismiss", "secondary-button"]]
    .forEach(([label, operation, className]) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = className;
      button.textContent = label;
      button.addEventListener("click", () => updateAlert(report.prediction_id, operation, button));
      wrap.append(button);
    });
}

async function updateAlert(predictionId, operation, button) {
  button.disabled = true;
  try {
    const response = await requestJson(`/v1/alerts/${encodeURIComponent(predictionId)}/${operation}`, {
      method: "POST",
      body: JSON.stringify({ note: "Updated from the NeuroGlycemic Sentinel artifact" }),
    });
    showToast(`Alert ${response.alert?.state || operation}`);
  } catch (error) {
    showToast(error.message);
  } finally {
    button.disabled = false;
  }
}

function renderProvenance(report, prediction, action, explanation) {
  const items = [
    ["Request", report.request_id],
    ["Prediction", report.prediction_id],
    ["Snapshot", prediction.snapshot_id],
    ["Model", report.model_version || prediction.model_version],
    ["Features", report.feature_version || prediction.feature_version],
    ["Calibration", prediction.calibration_version],
    ["Policy", `${action.policy_id || "—"} ${action.policy_version || ""}`.trim()],
    ["Retrieval", report.retrieval_status || "disabled"],
    ["Grounding", String(Boolean(report.grounding_complete))],
  ];
  $("#provenanceGrid").innerHTML = items.map(([label, value]) => `<div class="provenance-item"><span>${escapeHtml(label)}</span><code title="${escapeHtml(value || "Not reported")}">${escapeHtml(value || "Not reported")}</code></div>`).join("");
  const limitations = [...(explanation.limitations || []), report.disclaimer].filter(Boolean);
  $("#limitations").innerHTML = limitations.length ? limitations.map((item) => `<p>• ${escapeHtml(item)}</p>`).join("") : "<p>• No limitations were supplied.</p>";
}

function toggleProvenance() {
  const button = $("#provenanceToggle");
  const body = $("#provenanceBody");
  const open = button.getAttribute("aria-expanded") === "true";
  button.setAttribute("aria-expanded", String(!open));
  body.hidden = open;
  $("i", button).textContent = open ? "＋" : "−";
}

function copyPredictionId() {
  const id = state.report?.prediction_id;
  if (!id) return showToast("No prediction identifier is available");
  navigator.clipboard?.writeText(id).then(() => showToast("Prediction identifier copied"), () => showToast(id));
}

function openSettings() {
  $("#settingsEndpoint").value = state.apiBase;
  $("#requestTimeout").value = String(state.timeout);
  $("#autoDemo").checked = state.autoDemo;
  $("#settingsDialog").showModal();
}

function saveSettingsFromDialog(event) {
  event.preventDefault();
  state.apiBase = $("#settingsEndpoint").value.trim().replace(/\/$/, "") || DEFAULT_API;
  state.timeout = Number($("#requestTimeout").value || 30000);
  state.autoDemo = $("#autoDemo").checked;
  $("#apiEndpoint").value = state.apiBase;
  saveSettings();
  $("#settingsDialog").close();
  checkHealth();
  showToast("Engine settings saved");
}

function initializeEvents() {
  $$(".nav-link").forEach((button) => button.addEventListener("click", () => navigate(button.dataset.view)));
  $$('[data-go]').forEach((button) => button.addEventListener("click", () => navigate(button.dataset.go)));
  $$("[data-mode]").forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode)));
  $("#openDemo").addEventListener("click", () => loadDemoReport());
  $("#emptyDemo").addEventListener("click", () => loadDemoReport($("#patientId").value, $("#reportType").value));
  $("#fallbackDemo").addEventListener("click", () => loadDemoReport($("#patientId").value, $("#reportType").value));
  $("#reviewForm").addEventListener("submit", generateReview);
  $("#reportType").addEventListener("change", updateReportHint);
  $("#openAccess").addEventListener("click", () => $("#accessDialog").showModal());
  $("#accessForm").addEventListener("submit", connectEngine);
  $("#configureEndpoint").addEventListener("click", openSettings);
  $("#settingsForm").addEventListener("submit", saveSettingsFromDialog);
  $("#clearSession").addEventListener("click", clearSecureSession);
  $("#retryReview").addEventListener("click", () => setStage("empty"));
  $("#printReview").addEventListener("click", () => window.print());
  $("#copyReviewId").addEventListener("click", copyPredictionId);
  $("#provenanceToggle").addEventListener("click", toggleProvenance);
}

function bootstrap() {
  loadSettings();
  initializeEvents();
  updateReportHint();
  if (state.token) {
    state.connected = true;
    setService("online", "Session restored");
  }
  checkHealth();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", bootstrap, { once: true });
} else {
  bootstrap();
}
