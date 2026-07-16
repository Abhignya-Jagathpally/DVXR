"""dvxr.targets — prospective, causally-defined prediction targets (spec §5, §6, Gate 1).

The product predicts a FUTURE glucose excursion: given data observed up to a cutoff ``t``, will an
excursion occur in ``(t, t+horizon]``? This package builds that label deterministically, with explicit
censoring and threshold-version provenance. It is intentionally separate from the retrospective
within-window ``glucose_instability`` proxy in ``dvxr.clinical_tasks`` (which measures variability of
the OBSERVED window and must never be used as the future target).
"""
from dvxr.targets.excursion import (  # noqa: F401
    ExcursionExample,
    ExcursionThresholds,
    build_excursion_labels,
    history_slice,
)
