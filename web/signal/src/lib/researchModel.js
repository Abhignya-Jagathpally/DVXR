// SIGNAL — Transparent SYNTHETIC research model.
// Ported from the artifact reference engine (sim.js). Every number below is
// COMPUTED from the entered research-profile inputs — nothing is hardcoded.
// This is an illustrative demonstration model, never a real clinical result.
import { COLORS } from "./signals.js";
import {
  GROUP_KEYS,
  TARGET_ORDER,
  ACCEPTS,
  EVIDENCE,
  MODEL_VERSION,
  OUTCOME_LABEL,
  DISCLAIMER,
} from "./researchPredictionTypes.js";

const { neural: NEU, physio: PHY, metabolic: MET, clinical: CLI } = COLORS;

/* ---------- field spec (min/max/step/default/help per field) ---------- */
export const FIELDS = {
  metabolic: {
    color: MET,
    label: "Metabolic",
    fields: [
      { k: "current_glucose_mg_dl", l: "Current glucose", u: "mg/dL", min: 40, max: 400, step: 1, d: 148, h: "Most recent CGM reading" },
      { k: "mean_glucose_mg_dl", l: "Mean glucose", u: "mg/dL", min: 40, max: 300, step: 1, d: 136, h: "Average over the window" },
      { k: "glucose_std_mg_dl", l: "Glucose SD", u: "mg/dL", min: 0, max: 100, step: 1, d: 31, h: "Dispersion of glucose" },
      { k: "glucose_cv_percent", l: "Glucose CV", u: "%", min: 0, max: 60, step: 0.1, d: 22.8, h: "Coefficient of variation" },
      { k: "glucose_slope_mg_dl_min", l: "Glucose trend", u: "mg/dL/min", min: -5, max: 5, step: 0.1, d: 1.4, h: "Rate of change" },
      { k: "time_above_180_percent", l: "Time above 180", u: "%", min: 0, max: 100, step: 1, d: 17, h: "Time in hyperglycemia" },
      { k: "time_below_70_percent", l: "Time below 70", u: "%", min: 0, max: 50, step: 1, d: 2, h: "Time in hypoglycemia" },
      { k: "hba1c_percent", l: "HbA1c", u: "%", min: 4, max: 14, step: 0.1, d: 6.2, h: "Glycated hemoglobin" },
      { k: "fasting_glucose_mg_dl", l: "Fasting glucose", u: "mg/dL", min: 50, max: 300, step: 1, d: 118, h: "Fasting plasma glucose" },
      { k: "bmi", l: "BMI", u: "kg/m²", min: 15, max: 50, step: 0.1, d: 28.4, h: "Body-mass index" },
    ],
  },
  physiological: {
    color: PHY,
    label: "Physiological",
    fields: [
      { k: "heart_rate_bpm", l: "Resting heart rate", u: "bpm", min: 40, max: 180, step: 1, d: 83, h: "Beats per minute" },
      { k: "hrv_rmssd_ms", l: "HRV RMSSD", u: "ms", min: 5, max: 150, step: 1, d: 31, h: "Parasympathetic tone (higher = calmer)" },
      { k: "hrv_sdnn_ms", l: "HRV SDNN", u: "ms", min: 5, max: 200, step: 1, d: 42, h: "Overall variability" },
      { k: "eda_microsiemens", l: "Electrodermal activity", u: "µS", min: 0, max: 30, step: 0.1, d: 4.8, h: "Sympathetic arousal" },
      { k: "respiration_rate_bpm", l: "Respiration rate", u: "breaths/min", min: 6, max: 40, step: 1, d: 18, h: "Breathing rate" },
      { k: "skin_temperature_c", l: "Skin temperature", u: "°C", min: 28, max: 40, step: 0.1, d: 33.1, h: "Peripheral temperature" },
      { k: "activity_index", l: "Activity level", u: "", min: 0, max: 1, step: 0.01, d: 0.35, h: "0 rest → 1 active" },
      { k: "sleep_hours", l: "Sleep duration", u: "h", min: 0, max: 12, step: 0.1, d: 6.1, h: "Last night's sleep" },
    ],
  },
  neural: {
    color: NEU,
    label: "Neural",
    fields: [
      { k: "delta_relative_power", l: "Delta power", u: "rel", min: 0, max: 1, step: 0.01, d: 0.21, h: "Relative δ band" },
      { k: "theta_relative_power", l: "Theta power", u: "rel", min: 0, max: 1, step: 0.01, d: 0.24, h: "Relative θ band" },
      { k: "alpha_relative_power", l: "Alpha power", u: "rel", min: 0, max: 1, step: 0.01, d: 0.29, h: "Relative α band" },
      { k: "beta_relative_power", l: "Beta power", u: "rel", min: 0, max: 1, step: 0.01, d: 0.26, h: "Relative β band" },
      { k: "beta_alpha_ratio", l: "Beta / alpha ratio", u: "", min: 0, max: 3, step: 0.01, d: 0.9, h: "Cortical arousal proxy" },
      { k: "signal_quality", l: "Signal quality", u: "", min: 0, max: 1, step: 0.01, d: 0.88, h: "EEG quality (gates confidence)" },
    ],
  },
  clinical: {
    color: CLI,
    label: "Clinical",
    fields: [
      { k: "age", l: "Age", u: "years", min: 18, max: 90, step: 1, d: 39, h: "Participant age" },
      { k: "medication_count", l: "Medication count", u: "", min: 0, max: 15, step: 1, d: 1, h: "Number of active medications" },
      { k: "known_diabetes", l: "Known diabetes context", u: "0/1", min: 0, max: 1, step: 1, d: 0, h: "Prior diabetes indicator", toggle: true },
    ],
  },
};

/* ---------- math helpers ---------- */
export const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const sig = (zv) => 1 / (1 + Math.exp(-zv));
const zc = (v, center, scale) => (v === null ? 0 : (v - center) / scale);

/* ---------- default input state ---------- */
export function defaultInputs() {
  const state = {};
  GROUP_KEYS.forEach((g) => {
    state[g] = {};
    FIELDS[g].fields.forEach((f) => {
      state[g][f.k] = { value: f.d, present: true };
    });
  });
  return state;
}

/** Read a field value, or null when the participant marked it "not provided". */
function val(inputs, g, k) {
  const s = inputs[g] && inputs[g][k];
  return s && s.present ? s.value : null;
}

export function groupPresence(inputs, g) {
  const fs = FIELDS[g].fields;
  let p = 0;
  fs.forEach((f) => {
    if (inputs[g][f.k].present) p++;
  });
  if (p === 0) return "none";
  if (p === fs.length) return "full";
  return "partial";
}

/* ---------- transparent scoring ---------- */
function conf(inputs, target) {
  const groups = ACCEPTS[target] || [];
  if (!groups.length) return 0.7;
  let c = 0, n = 0;
  groups.forEach((g) => {
    FIELDS[g].fields.forEach((f) => {
      n++;
      if (inputs[g][f.k].present) c++;
    });
  });
  let base = 0.55 + 0.35 * (n ? c / n : 0);
  if (groups.indexOf("neural") >= 0) {
    const q = val(inputs, "neural", "signal_quality");
    if (q !== null) base *= 0.6 + 0.4 * q;
  }
  return clamp(base, 0.35, 0.95);
}

export function band(p) {
  return p < 0.25 ? "lower" : p < 0.5 ? "moderate" : p < 0.75 ? "elevated" : "high";
}

function targetProb(inputs, t) {
  const v = (g, k) => val(inputs, g, k);
  if (t === "stress") {
    const s =
      0.15 +
      1.15 * zc(v("physiological", "hrv_rmssd_ms"), 40, -25) +
      0.9 * zc(v("physiological", "eda_microsiemens"), 3, 4) +
      0.6 * zc(v("physiological", "heart_rate_bpm"), 70, 20) +
      0.4 * zc(v("physiological", "respiration_rate_bpm"), 16, 6) +
      0.3 * zc(v("physiological", "sleep_hours"), 7, -2);
    return sig(s);
  }
  if (t === "anxiety") {
    const s =
      0.0 +
      0.7 * zc(v("physiological", "hrv_rmssd_ms"), 40, -25) +
      0.6 * zc(v("physiological", "eda_microsiemens"), 3, 4) +
      0.7 * zc(v("neural", "beta_alpha_ratio"), 0.9, 0.6) +
      0.3 * zc(v("physiological", "respiration_rate_bpm"), 16, 6);
    return sig(s);
  }
  if (t === "depression") {
    const s =
      -0.1 +
      0.6 * zc(v("neural", "alpha_relative_power"), 0.28, 0.1) +
      0.4 * zc(v("neural", "theta_relative_power"), 0.22, 0.08) -
      0.3 * zc(v("neural", "beta_alpha_ratio"), 0.9, 0.6);
    return sig(s);
  }
  if (t === "cognitive_workload") {
    const s =
      -0.1 +
      1.0 * zc(v("neural", "beta_alpha_ratio"), 0.9, 0.6) +
      0.5 * zc(v("neural", "theta_relative_power"), 0.22, 0.08) +
      0.3 * zc(v("neural", "beta_relative_power"), 0.24, 0.1);
    return sig(s);
  }
  if (t === "glucose_instability") {
    const s =
      -0.2 +
      1.1 * zc(v("metabolic", "glucose_cv_percent"), 18, 8) +
      0.6 * zc(v("metabolic", "glucose_std_mg_dl"), 25, 12) +
      0.7 * zc(v("metabolic", "time_above_180_percent"), 10, 12) +
      0.4 * Math.abs(zc(v("metabolic", "glucose_slope_mg_dl_min"), 0, 2));
    return sig(s);
  }
  return 0.5;
}

// Diabetes meta model. Mental-health probabilities enter as CONTEXT (small
// weights), framed as associated predictors — never as causes.
function diabetesModel(inputs, mhProbs) {
  const v = (g, k) => val(inputs, g, k);
  const terms = [];
  const push = (factor, contribution) => {
    if (contribution === 0 && factor.indexOf("context") < 0) return;
    terms.push({
      factor,
      signed_contribution: Math.round(contribution * 100) / 100,
      direction: contribution >= 0 ? "increases_estimate" : "decreases_estimate",
      method: "linear",
    });
  };
  const bias = -0.4;
  let sum = bias;
  const metabPresent = groupPresence(inputs, "metabolic") !== "none";
  const pairs = [
    ["HbA1c", 1.3 * zc(v("metabolic", "hba1c_percent"), 5.7, 0.8)],
    ["Glucose instability", 0.55 * (mhProbs.glucose_instability - 0.4) * 2],
    ["BMI", 0.55 * zc(v("metabolic", "bmi"), 25, 5)],
    ["Fasting glucose", 0.6 * zc(v("metabolic", "fasting_glucose_mg_dl"), 100, 25)],
    ["Mean glucose", 0.45 * zc(v("metabolic", "mean_glucose_mg_dl"), 110, 25)],
    ["Time above 180", 0.4 * zc(v("metabolic", "time_above_180_percent"), 10, 12)],
    ["Age", 0.3 * zc(v("clinical", "age"), 45, 15)],
    // mental-health CONTEXT (small, associated predictors, never causes)
    ["Stress context", 0.24 * (mhProbs.stress - 0.4)],
    ["Anxiety context", 0.12 * (mhProbs.anxiety - 0.4)],
    ["Depression context", 0.06 * (mhProbs.depression - 0.4)],
    ["Workload context", 0.05 * (mhProbs.cognitive_workload - 0.4)],
    ["Activity", -0.2 * zc(v("physiological", "activity_index"), 0.4, 0.3)],
  ];
  const kd = v("clinical", "known_diabetes");
  if (kd !== null && kd >= 1) pairs.push(["Known diabetes context", 0.9]);
  pairs.forEach((p) => {
    if (isFinite(p[1])) {
      sum += p[1];
      push(p[0], p[1]);
    }
  });
  terms.sort((a, b) => Math.abs(b.signed_contribution) - Math.abs(a.signed_contribution));
  return { p: sig(sum), terms, metabPresent };
}

function metabContribs(inputs) {
  const v = (g, k) => val(inputs, g, k);
  const t = [];
  const p = (f, c) => {
    if (isFinite(c))
      t.push({
        factor: f,
        signed_contribution: Math.round(c * 100) / 100,
        direction: c >= 0 ? "increases_estimate" : "decreases_estimate",
        method: "linear",
      });
  };
  p("Glucose CV", 1.1 * zc(v("metabolic", "glucose_cv_percent"), 18, 8));
  p("Glucose SD", 0.6 * zc(v("metabolic", "glucose_std_mg_dl"), 25, 12));
  p("Time above 180", 0.7 * zc(v("metabolic", "time_above_180_percent"), 10, 12));
  p("Glucose trend", 0.4 * Math.abs(zc(v("metabolic", "glucose_slope_mg_dl_min"), 0, 2)));
  t.sort((a, b) => Math.abs(b.signed_contribution) - Math.abs(a.signed_contribution));
  return t;
}

function forecastFrom(inputs) {
  const cur = val(inputs, "metabolic", "current_glucose_mg_dl");
  let slope = val(inputs, "metabolic", "glucose_slope_mg_dl_min");
  let sd = val(inputs, "metabolic", "glucose_std_mg_dl");
  if (cur === null) return null;
  slope = slope === null ? 0 : slope;
  sd = sd === null ? 20 : sd;
  const pt = (h) => {
    const p = clamp(cur + slope * h, 40, 400);
    const w = sd * 0.5 + h * 0.25 * (1 + Math.abs(slope));
    return {
      point_mg_dl: Math.round(p),
      lower_mg_dl: Math.round(clamp(p - w, 40, 400)),
      upper_mg_dl: Math.round(clamp(p + w, 40, 400)),
    };
  };
  return { history_last: Math.round(cur), 30: pt(30), 60: pt(60) };
}

function inputQuality(inputs) {
  let c = 0, n = 0;
  GROUP_KEYS.forEach((g) => {
    FIELDS[g].fields.forEach((f) => {
      n++;
      if (inputs[g][f.k].present) c++;
    });
  });
  const score = n ? c / n : 0;
  const missing = [];
  GROUP_KEYS.forEach((g) => {
    if (groupPresence(inputs, g) === "none") missing.push(g);
  });
  return {
    overall: score > 0.75 ? "acceptable" : score > 0.4 ? "partial" : "limited",
    score: Math.round(score * 100) / 100,
    missing_modalities: missing.concat(["molecular"]),
  };
}

function hashState(inputs, selectedOutcome) {
  const s = JSON.stringify(inputs) + selectedOutcome;
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0;
  return h;
}

/**
 * Run the full transparent model for a request.
 * @param {import("./researchPredictionTypes.js").ResearchInputs} inputs
 * @param {import("./researchPredictionTypes.js").SelectedOutcome} selectedOutcome
 * @returns {import("./researchPredictionTypes.js").ResearchPredictionResponse}
 */
export function runModel(inputs, selectedOutcome) {
  const mh = {
    stress: targetProb(inputs, "stress"),
    anxiety: targetProb(inputs, "anxiety"),
    depression: targetProb(inputs, "depression"),
    cognitive_workload: targetProb(inputs, "cognitive_workload"),
    glucose_instability: targetProb(inputs, "glucose_instability"),
  };
  const target_predictions = {};
  TARGET_ORDER.forEach((t) => {
    target_predictions[t] = {
      probability: mh[t],
      risk_band: band(mh[t]),
      confidence: conf(inputs, t),
      model_version: MODEL_VERSION[t],
      evidence_status: EVIDENCE[t],
    };
  });

  const out = {
    prediction_id: "sim_" + Math.abs(hashState(inputs, selectedOutcome)).toString(36),
    status: "completed",
    selected_outcome: selectedOutcome,
    target_predictions,
    input_quality: inputQuality(inputs),
    forecast: forecastFrom(inputs),
    disclaimer: DISCLAIMER,
    synthetic: true,
    abstained: false,
    contributions: [],
    selected: null,
    node_contrib: {},
  };

  if (selectedOutcome === "glucose_instability") {
    out.selected = {
      name: "glucose_instability",
      probability: mh.glucose_instability,
      risk_band: "research-" + band(mh.glucose_instability),
      confidence: conf(inputs, "glucose_instability"),
      model_version: MODEL_VERSION.glucose_instability,
      evidence_status: "metabolic-model",
      validated_for_clinical_use: false,
    };
    out.contributions = metabContribs(inputs);
    if (groupPresence(inputs, "metabolic") === "none") {
      out.abstained = true;
      out.abstain_reason = "metabolic_unavailable";
      out.selected.probability = null;
      out.selected.evidence_status = "abstained";
    }
  } else {
    // diabetes_status or diabetes_complication -> meta model
    const dm = diabetesModel(inputs, mh);
    if (!dm.metabPresent) {
      out.abstained = true;
      out.abstain_reason = "metabolic_unavailable";
      out.selected = {
        name: selectedOutcome,
        probability: null,
        evidence_status: "abstained",
        validated_for_clinical_use: false,
      };
    } else {
      out.selected = {
        name: selectedOutcome,
        probability: dm.p,
        risk_band: "research-" + band(dm.p),
        confidence: conf(inputs, "glucose_instability") * 0.9 + 0.05,
        model_version: MODEL_VERSION[selectedOutcome],
        evidence_status: EVIDENCE[selectedOutcome],
        validated_for_clinical_use: false,
      };
      out.contributions = dm.terms;
    }
  }

  // node contributions for constellation links — derived from model terms,
  // NEVER from probability alone.
  const findc = (name) => {
    const term = (out.contributions || []).find((x) => x.factor === name);
    return term ? Math.abs(term.signed_contribution) : 0;
  };
  out.node_contrib = {
    stress: findc("Stress context"),
    anxiety: findc("Anxiety context"),
    depression: findc("Depression context"),
    cognitive_workload: findc("Workload context"),
    glucose_instability:
      findc("Glucose instability") || (selectedOutcome === "glucose_instability" ? 0.3 : 0),
  };

  // final status reflects abstention / partial evidence
  if (out.abstained) out.status = "abstention";
  else if (out.input_quality.overall === "limited") out.status = "partial";
  else out.status = "completed";

  return out;
}

export { OUTCOME_LABEL };

/* ---------- sample research profiles ---------- */
export const SAMPLES = [
  {
    id: "low",
    label: "Lower metabolic instability",
    blurb: "Calm physiology, stable glucose, healthy variability.",
    set: {
      metabolic: { current_glucose_mg_dl: 104, mean_glucose_mg_dl: 101, glucose_std_mg_dl: 14, glucose_cv_percent: 13.9, glucose_slope_mg_dl_min: 0.2, time_above_180_percent: 2, time_below_70_percent: 1, hba1c_percent: 5.3, fasting_glucose_mg_dl: 92, bmi: 23.1 },
      physiological: { hrv_rmssd_ms: 58, eda_microsiemens: 2.1, heart_rate_bpm: 64, respiration_rate_bpm: 14, sleep_hours: 7.8, activity_index: 0.5 },
      neural: { beta_alpha_ratio: 0.7, alpha_relative_power: 0.31, theta_relative_power: 0.2, signal_quality: 0.9 },
    },
  },
  {
    id: "stress_stable",
    label: "Elevated stress · stable glucose",
    blurb: "High arousal and low HRV, but metabolism stays in range.",
    set: {
      metabolic: { current_glucose_mg_dl: 112, mean_glucose_mg_dl: 108, glucose_std_mg_dl: 16, glucose_cv_percent: 14.8, glucose_slope_mg_dl_min: 0.3, time_above_180_percent: 3, time_below_70_percent: 1, hba1c_percent: 5.5, fasting_glucose_mg_dl: 98, bmi: 24.6 },
      physiological: { hrv_rmssd_ms: 24, eda_microsiemens: 7.9, heart_rate_bpm: 92, respiration_rate_bpm: 21, sleep_hours: 5.4, activity_index: 0.3 },
      neural: { beta_alpha_ratio: 1.5, alpha_relative_power: 0.22, theta_relative_power: 0.27, signal_quality: 0.85 },
    },
  },
  {
    id: "stress_unstable",
    label: "Elevated stress · glucose instability",
    blurb: "Arousal and glucose variability rise together.",
    set: {
      metabolic: { current_glucose_mg_dl: 168, mean_glucose_mg_dl: 151, glucose_std_mg_dl: 44, glucose_cv_percent: 29.1, glucose_slope_mg_dl_min: 1.9, time_above_180_percent: 31, time_below_70_percent: 3, hba1c_percent: 6.8, fasting_glucose_mg_dl: 129, bmi: 30.7 },
      physiological: { hrv_rmssd_ms: 21, eda_microsiemens: 8.6, heart_rate_bpm: 95, respiration_rate_bpm: 22, sleep_hours: 5.1, activity_index: 0.28 },
      neural: { beta_alpha_ratio: 1.6, alpha_relative_power: 0.2, theta_relative_power: 0.28, signal_quality: 0.83 },
    },
  },
];

/** Apply a sample profile onto a fresh input state. */
export function applySample(sampleId) {
  const sample = SAMPLES.find((s) => s.id === sampleId);
  const state = defaultInputs();
  if (!sample) return state;
  GROUP_KEYS.forEach((g) => {
    FIELDS[g].fields.forEach((f) => {
      const v = sample.set[g] && sample.set[g][f.k];
      if (v !== undefined) state[g][f.k] = { value: v, present: true };
    });
  });
  return state;
}
