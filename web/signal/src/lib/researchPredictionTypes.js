// SIGNAL — Research prediction API contract (typedefs + enums).
// SYNTHETIC DEMONSTRATION ONLY. These types describe the request/response
// contract shared by the mock adapter and the (optional) real backend.
//
// A .js file with JSDoc typedefs is used rather than .ts because the project
// ships no TypeScript tooling; the shapes below are the single source of truth.

/**
 * @typedef {"metabolic"|"physiological"|"neural"|"clinical"} GroupKey
 * @typedef {"manual"|"upload"|"sample"} InputMode
 * @typedef {"diabetes_status"|"glucose_instability"|"diabetes_complication"} SelectedOutcome
 * @typedef {"stress"|"anxiety"|"depression"|"cognitive_workload"|"glucose_instability"} TargetName
 * @typedef {"increases_estimate"|"decreases_estimate"} ContributionDirection
 * @typedef {"idle"|"loading"|"validation-failure"|"api-unavailable"|"partial"|"abstention"|"completed"} PredictionState
 */

/**
 * A single research-profile field value with an explicit "provided" flag.
 * @typedef {Object} FieldValue
 * @property {number} value
 * @property {boolean} present
 */

/**
 * @typedef {Object.<string, FieldValue>} GroupInputs
 * @typedef {Object} ResearchInputs
 * @property {GroupInputs} metabolic
 * @property {GroupInputs} physiological
 * @property {GroupInputs} neural
 * @property {GroupInputs} clinical
 */

/**
 * Request posted to `${VITE_RESEARCH_API_URL}/v1/research/predict`.
 * @typedef {Object} ResearchPredictionRequest
 * @property {string} session_id
 * @property {InputMode} input_mode
 * @property {SelectedOutcome} selected_outcome
 * @property {number[]} prediction_horizons_minutes
 * @property {ResearchInputs} inputs
 * @property {TargetName[]} targets
 */

/**
 * @typedef {Object} InputQuality
 * @property {"acceptable"|"partial"|"limited"} overall
 * @property {number} score  // 0..1
 * @property {string[]} missing_modalities
 */

/**
 * @typedef {Object} TargetPrediction
 * @property {number} probability   // 0..1
 * @property {"lower"|"moderate"|"elevated"|"high"} risk_band
 * @property {number} confidence    // 0..1
 * @property {string} model_version
 * @property {string} evidence_status
 */

/**
 * @typedef {Object} Contribution
 * @property {string} factor
 * @property {number} signed_contribution
 * @property {ContributionDirection} direction
 * @property {string} method
 */

/**
 * @typedef {Object} ForecastPoint
 * @property {number} point_mg_dl
 * @property {number} lower_mg_dl
 * @property {number} upper_mg_dl
 */

/**
 * @typedef {Object} Forecast
 * @property {number} history_last
 * @property {ForecastPoint} "30"
 * @property {ForecastPoint} "60"
 */

/**
 * @typedef {Object} SelectedEstimate
 * @property {SelectedOutcome} name
 * @property {number|null} probability
 * @property {string} [risk_band]
 * @property {number} [confidence]
 * @property {string} [model_version]
 * @property {string} evidence_status
 * @property {boolean} validated_for_clinical_use
 */

/**
 * Response returned by the mock adapter or the real backend.
 * @typedef {Object} ResearchPredictionResponse
 * @property {string} prediction_id
 * @property {PredictionState} status
 * @property {InputQuality} input_quality
 * @property {Object.<TargetName, TargetPrediction>} target_predictions
 * @property {SelectedOutcome} selected_outcome
 * @property {SelectedEstimate} selected
 * @property {Contribution[]} contributions
 * @property {Forecast|null} forecast
 * @property {Object.<TargetName, number>} node_contrib
 * @property {boolean} abstained
 * @property {string} [abstain_reason]
 * @property {boolean} synthetic
 * @property {string} disclaimer
 */

export const GROUP_KEYS = /** @type {GroupKey[]} */ ([
  "metabolic",
  "physiological",
  "neural",
  "clinical",
]);

export const TARGET_ORDER = /** @type {TargetName[]} */ ([
  "stress",
  "anxiety",
  "depression",
  "cognitive_workload",
  "glucose_instability",
]);

export const TARGET_NAMES = {
  stress: "Stress",
  anxiety: "Anxiety",
  depression: "Depression",
  cognitive_workload: "Cognitive load",
  glucose_instability: "Glucose instability",
};

/** Three selectable research outcomes for the primary estimate. */
export const OUTCOMES = /** @type {{id: SelectedOutcome, label: string, help: string}[]} */ ([
  {
    id: "diabetes_status",
    label: "Diabetes-related status",
    help: "Meta-model over metabolic predictors, with mental-health estimates entering only as context.",
  },
  {
    id: "glucose_instability",
    label: "Glucose instability",
    help: "Metabolic-only research estimate of short-window glucose variability.",
  },
  {
    id: "diabetes_complication",
    label: "Complication-risk estimate",
    help: "Exploratory complication-risk research estimate built on the same context model.",
  },
]);

export const OUTCOME_LABEL = {
  diabetes_status: "Diabetes-related status",
  glucose_instability: "Glucose instability",
  diabetes_complication: "Complication-risk research estimate",
};

export const EVIDENCE = {
  stress: "component-model",
  anxiety: "experimental",
  depression: "component-model",
  cognitive_workload: "component-model",
  glucose_instability: "metabolic-model",
  diabetes_status: "experimental",
  diabetes_complication: "experimental",
};

export const MODEL_VERSION = {
  stress: "stress-sim-v1",
  anxiety: "anxiety-sim-v1",
  depression: "depression-sim-v1",
  cognitive_workload: "workload-sim-v1",
  glucose_instability: "glucose-instability-sim-v1",
  diabetes_status: "diabetes-context-sim-v1",
  diabetes_complication: "complication-context-sim-v1",
};

/** Which input groups each per-task target consumes (EEG only moves neural-accepting targets). */
export const ACCEPTS = {
  stress: ["physiological"],
  anxiety: ["physiological", "neural"],
  depression: ["neural"],
  cognitive_workload: ["neural"],
  glucose_instability: ["metabolic"],
};

export const DISCLAIMER =
  "Research simulation only. This output is not a medical diagnosis.";
