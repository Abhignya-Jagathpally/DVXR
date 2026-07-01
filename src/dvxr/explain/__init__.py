"""dvxr.explain — explainability bundle."""
from .linear import *  # noqa: F401,F403
from .attention_maps import attention_table, export_attention  # noqa: F401
from .codebook_usage import (  # noqa: F401
    codebook_histogram,
    codebook_perplexity,
    top_codes_per_label,
)
from .report import explain_prediction  # noqa: F401
# re-export biomarkers under dvxr.explain for the bundle (per ARCHITECTURE §A8)
from dvxr.biomarkers import (  # noqa: F401
    neural_biomarker_saliency,
    physiological_biomarkers,
)
