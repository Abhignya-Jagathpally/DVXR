"""dvxr.config — the single source of truth for every CACMF hyperparameter.

Nothing in the framework should hard-code a constant; it lives here (ARCHITECTURE
§A7) and is loaded via ``CACMFConfig``. Supports YAML (if pyyaml present) and a
JSON fallback so config I/O never requires a heavy dependency.

Real-weight policy (per user directive "MAKE SURE TO USE REAL WEIGHTS"):
``foundation_models`` maps each canonical modality to a VERIFIED, publicly
downloadable, CPU-runnable checkpoint. Where the POW's originally-named model has
no usable open weights (EEG-X, GluFormer, Med-BERT, PH-LLM), a documented real
substitute is used and the original is recorded in ``originally_selected`` so the
swap is auditable and reversible.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

MODALITIES: List[str] = ["eeg", "wearable_phys", "cgm", "ehr", "omics", "behavior"]

FUSION_STRATEGIES: List[str] = [
    "early", "intermediate", "late_weighted", "attention", "cross_modal",
]
AGGREGATIONS: List[str] = ["weighted_late", "ensemble_avg", "confidence_weighted"]


@dataclass(frozen=True)
class FoundationModel:
    """One modality's real-weight adapter spec: primary -> fallback -> baseline.

    Adapters try ``primary_id`` (via ``primary_loader``); if its weights/deps are
    unavailable they try ``fallback_id``; if that also fails they use the
    always-runnable ``baseline`` (no network, no GPU). Every hop is logged.
    """
    modality: str
    primary_id: str                 # HF repo id / Figshare / "local:<name>"
    primary_loader: str             # transformers|momentfm|braindecode|chronos|repo|local
    fallback_id: str
    fallback_loader: str
    baseline: str                   # always-runnable fallback (bundled)
    license: str
    weights_status: str             # "open" | "train_local" | "substitute:<reason>"
    originally_selected: str        # POW's named model
    note: str = ""


# Verified July 2026 against live HF/GitHub/Figshare (see docs/MASTER_BRIEF.md §1.3).
FOUNDATION_MODELS: Dict[str, FoundationModel] = {
    "eeg": FoundationModel(
        "eeg", "braindecode/labram-pretrained", "braindecode",
        "figshare:eegpt_mcae_58chs_4s_large4E.ckpt", "repo",
        "bandpower+per-channel-stats -> VQBiosignalEncoder", "BSD-3", "open",
        "EEG-X (→LaBraM)", "EEG-X repo 404/unreleased; LaBraM public, EEGPT fallback. "
        "NOT WIRED AT RUNTIME: no braindecode loader in make_primary_backend, so this "
        "returns None and the bandpower+VQ baseline runs. Wiring needs braindecode[hug] "
        "+ a raw-signal path (LaBraM cannot consume the summary-stat table)."),
    "wearable_phys": FoundationModel(
        "wearable_phys", "AutonLab/MOMENT-1-large", "momentfm",
        "google/timesfm-2.0-500m-pytorch", "transformers",
        "NeuralBiosignalEncoder / PCA", "MIT", "open",
        "PAT (→MOMENT)", "PAT weights are TF/Keras .h5 (not torch); MOMENT primary for pure-torch"),
    "cgm": FoundationModel(
        "cgm", "CRUISEResearchGroup/CGM-JEPA", "transformers",
        "amazon/chronos-bolt-small", "chronos",
        "conformalized Ridge forecaster + latent summary", "MIT", "open",
        "CGM-JEPA (GluFormer had no weights)", "CGM-JEPA now public; Chronos-Bolt fallback. "
        "NOT WIRED AT RUNTIME: no HF-text-loadable weights for the transformers loader, so "
        "this returns None and the conformal-Ridge baseline runs."),
    "ehr": FoundationModel(
        "ehr", "local:cehrbert_style", "local",
        "emilyalsentzer/Bio_ClinicalBERT", "transformers",
        "tokenized-code timeline features", "MIT", "train_local",
        "Med-BERT/BEHRT/CEHR-BERT", "CEHR-BERT has no public weights -> train locally on MIMIC code timelines; Bio_ClinicalBERT for notes"),
    "omics": FoundationModel(
        "omics", "ctheodoris/Geneformer", "transformers",
        "ctheodoris/Geneformer", "transformers",
        "build_omics_features -> linear proj", "Apache-2.0", "open",
        "(none in POW)", "Geneformer public"),
    "behavior": FoundationModel(
        "behavior", "AutonLab/MOMENT-1-large", "momentfm",
        "google/timesfm-2.0-500m-pytorch", "transformers",
        "VR/AR behavior features -> linear proj", "MIT", "open",
        "(none in POW)", "behavioral time-series via MOMENT"),
}

# LLM insight layer: hosted real weights (Anthropic Claude) with local open-weight
# + deterministic-offline fallbacks. Reasoning model ids verified ungated.
LLM_INSIGHT = {
    "provider": "anthropic",
    "model": "claude-opus-4-8",             # override via env DVXR_LLM_MODEL
    "local_open_fallback": "Qwen/Qwen2.5-7B-Instruct",   # ungated, apache-2.0, GGUF
    "local_medical_fallback": "BioMistral/BioMistral-7B",
    "offline_fallback": "deterministic-template",
    "originally_selected": "PH-LLM/PHIA/Health-LLM (all closed/unreleased)",
}


@dataclass
class CACMFConfig:
    """Every tunable constant for CACMF (ARCHITECTURE §A7). Defaults are the spec."""

    # --- latent / codebook ---
    d: int = 64                         # per-modality latent width
    d_f: int = 128                      # fused hidden width
    codebook_size: int = 512            # K
    commitment_beta: float = 0.25       # VQ commitment weight
    gumbel: bool = False                # soft-assignment path toggle
    temperature: float = 1.0            # τ (softmax / gumbel)

    # --- fusion transformer ---
    n_fusion_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.1
    fusion_strategy: str = "cross_modal"
    aggregation: str = "confidence_weighted"

    # --- windowing ---
    window_seconds: float = 30.0
    window_step: float = 30.0
    mask_ratio: float = 0.3

    # --- optimization ---
    epochs: int = 30
    batch_size: int = 64
    lr_encoder: float = 1e-3
    lr_fusion: float = 5e-4
    weight_decay: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.999
    warmup_frac: float = 0.08           # linear warmup then cosine decay
    grad_clip: float = 1.0
    use_ema: bool = False

    # --- relative loss weights (§A6) ---
    lambda_task: float = 1.0
    lambda_vq: float = 1.0
    lambda_recon: float = 0.5
    lambda_align: float = 0.1           # InfoNCE cross-modal alignment
    align_temperature: float = 0.1      # τ_a
    uncertainty_weighting: bool = False  # Kendall learned σ_t

    # --- real-weight policy ---
    use_real_weights: bool = True       # user directive: default ON
    allow_download: bool = True         # fetch checkpoints from HF/GitHub if missing
    weights_cache_dir: str = "~/.cache/dvxr_weights"

    # --- reproducibility ---
    seed: int = 7

    # --- bookkeeping (not tuned) ---
    modalities: List[str] = field(default_factory=lambda: list(MODALITIES))

    # ---------- validation ----------
    def __post_init__(self) -> None:
        if self.fusion_strategy not in FUSION_STRATEGIES:
            raise ValueError(
                f"fusion_strategy {self.fusion_strategy!r} not in {FUSION_STRATEGIES}")
        if self.aggregation not in AGGREGATIONS:
            raise ValueError(
                f"aggregation {self.aggregation!r} not in {AGGREGATIONS}")
        if not (0.0 <= self.mask_ratio < 1.0):
            raise ValueError("mask_ratio must be in [0, 1)")
        if self.d <= 0 or self.d_f <= 0 or self.codebook_size <= 0:
            raise ValueError("d, d_f, codebook_size must be positive")

    # ---------- serialization ----------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def with_(self, **overrides: Any) -> "CACMFConfig":
        """Return a copy with overrides (validated)."""
        return replace(self, **overrides)

    def save(self, path: str | Path) -> Path:
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore
                path.write_text(yaml.safe_dump(data, sort_keys=False))
                return path
            except Exception:
                path = path.with_suffix(".json")
        path.write_text(json.dumps(data, indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "CACMFConfig":
        path = Path(path).expanduser()
        text = path.read_text()
        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore
                data = yaml.safe_load(text)
            except Exception:
                data = json.loads(text)
        else:
            data = json.loads(text)
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in (data or {}).items() if k in known})

    def foundation_model(self, modality: str) -> Optional[FoundationModel]:
        return FOUNDATION_MODELS.get(modality)


# Frozen defaults instance (import and copy via .with_(...), never mutate).
DEFAULTS = CACMFConfig()
