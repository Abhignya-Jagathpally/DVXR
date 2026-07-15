"""dvxr.cohort — cohort synchrony registry + the fusion honesty gate (spec §1.B, §4).

The central scientific-integrity rule of the whole system: a *multimodal fusion claim* — e.g. "EEG
adds value to CGM glucose forecasting" — is only valid on data where those modalities are
**synchronized on the same subject over the same time interval**. Public component datasets are
separate cohorts and are never cross-joined; each co-registers only its own modality set. Crucially,
**no public cohort co-registers EEG together with CGM**, so the glucose product's headline fusion claim
cannot be validated on public data — the product stays research-stage until synchronized pilot data
exists (`dvxr.serve.evidence.PRODUCT_VISION`).

`require_synchronized_for_fusion` makes that rule executable: it raises before any fused prediction is
computed over a modality set a cohort does not genuinely co-register. Single-modality use is never
gated; within-cohort fusion of a genuinely co-registered set (e.g. DEAP EEG+peripheral) is permitted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, Optional, Union


class SynchronyError(RuntimeError):
    """Raised when a fusion claim is attempted over modalities a cohort does not co-register."""


@dataclass(frozen=True)
class CohortSpec:
    """What a cohort genuinely co-registers. ``synchronized_modalities`` is the set of modalities
    recorded on the SAME subjects over the SAME sessions — the only set a fusion claim may span."""
    cohort_id: str
    synchronized_modalities: FrozenSet[str]
    fusion_claim_permitted: bool = True
    note: str = ""

    @property
    def synchronized_same_subject(self) -> bool:
        """True iff this cohort co-registers more than one modality on the same subjects."""
        return len(self.synchronized_modalities) > 1

    def covers(self, modalities: Iterable[str]) -> bool:
        return set(modalities) <= set(self.synchronized_modalities)


# The genuinely co-registered modality set per public cohort. NOTE: no cohort contains {eeg, cgm}
# together — that is the exact gap that keeps the glucose fusion claim research-stage.
COHORT_REGISTRY: Dict[str, CohortSpec] = {
    "synthetic": CohortSpec("synthetic",
                            frozenset({"eeg", "wearable_phys", "cgm", "ehr", "omics", "behavior"}),
                            note="Synthetic co-registered fixture (all modalities same synthetic subject)."),
    "deap": CohortSpec("deap", frozenset({"eeg", "wearable_phys"}),
                       note="32-ch EEG + peripheral physiology, same subject/session (no CGM)."),
    "eegmat": CohortSpec("eegmat", frozenset({"eeg", "wearable_phys"}),
                         note="EEG + ECG during mental arithmetic, same subject/session (no CGM)."),
    "mumtaz": CohortSpec("mumtaz", frozenset({"eeg"}),
                         note="Resting EEG only (single modality)."),
    "wesad": CohortSpec("wesad", frozenset({"wearable_phys"}),
                        note="Chest+wrist wearable physiology, same subject (no EEG, no CGM)."),
    "noneeg": CohortSpec("noneeg", frozenset({"wearable_phys"}),
                         note="Peripheral physiology only."),
    "cgmacros": CohortSpec("cgmacros", frozenset({"cgm", "wearable_phys", "behavior", "ehr"}),
                           note="CGM + Fitbit + meal macros + bio labs, same subject (no EEG)."),
    "shanghai_cgm": CohortSpec("shanghai_cgm", frozenset({"cgm", "ehr"}),
                               note="CGM + minimal clinical context (no EEG)."),
    "mimic_demo": CohortSpec("mimic_demo", frozenset({"ehr"}),
                             note="Structured EHR only."),
}

#: The glucose product's target fusion set — deliberately unmatched by any public cohort.
GLUCOSE_FUSION_MODALITIES = frozenset({"eeg", "cgm", "wearable_phys"})


def cohort_synchrony(cohort_id: str) -> Optional[CohortSpec]:
    """Registered synchrony spec for a cohort id, or None if unknown."""
    return COHORT_REGISTRY.get(cohort_id)


def can_fuse(cohort: Union[str, CohortSpec], modalities: Iterable[str]) -> bool:
    """True iff a fusion claim over ``modalities`` is scientifically valid on this cohort."""
    mods = set(modalities)
    if len(mods) <= 1:
        return True                       # single modality is never a fusion claim
    spec = cohort if isinstance(cohort, CohortSpec) else cohort_synchrony(str(cohort))
    if spec is None:
        return False                      # unknown cohort ⇒ cannot assert synchrony ⇒ deny
    return spec.fusion_claim_permitted and spec.covers(mods)


def require_synchronized_for_fusion(cohort: Union[str, CohortSpec],
                                    modalities: Iterable[str]) -> None:
    """Raise SynchronyError unless a fusion claim over ``modalities`` is valid on ``cohort``.

    Call this at every point that computes a *reportable multimodal* prediction. Single-modality
    predictions pass through untouched.
    """
    mods = set(modalities)
    if len(mods) <= 1:
        return
    spec = cohort if isinstance(cohort, CohortSpec) else cohort_synchrony(str(cohort))
    cohort_id = spec.cohort_id if isinstance(spec, CohortSpec) else str(cohort)
    if spec is None:
        raise SynchronyError(
            f"Fusion over {sorted(mods)} refused: cohort {cohort_id!r} is not a registered "
            f"synchronized cohort — its same-subject synchrony cannot be asserted.")
    if not spec.fusion_claim_permitted:
        raise SynchronyError(
            f"Fusion refused on cohort {cohort_id!r}: fusion_claim_permitted is False.")
    missing = mods - set(spec.synchronized_modalities)
    if missing:
        raise SynchronyError(
            f"Fusion over {sorted(mods)} refused on cohort {cohort_id!r}: it does not co-register "
            f"{sorted(missing)} on the same subjects (synchronized set = "
            f"{sorted(spec.synchronized_modalities)}). A fused claim spanning these modalities "
            f"requires synchronized same-subject pilot data (spec §1.B).")
