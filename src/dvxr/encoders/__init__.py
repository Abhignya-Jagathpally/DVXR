"""dvxr.encoders — modality encoders (baseline + VQ + adapters)."""
from .baseline import *  # noqa: F401,F403
from .codebook import (  # noqa: F401
    VQBiosignalEncoder,
    VQOutput,
    get_vector_quantizer_class,
)
from .base import (  # noqa: F401
    BaseAdapter,
    EncoderProtocol,
    ModalityEncoderRegistry,
    z_frame,
)
from .eeg_adapter import EEGAdapter  # noqa: F401
from .biosignal_adapter import BiosignalAdapter  # noqa: F401
from .cgm_adapter import CGMAdapter  # noqa: F401
from .ehr_adapter import EHRAdapter  # noqa: F401
from .omics_adapter import OmicsAdapter  # noqa: F401
from .behavior_adapter import BehaviorAdapter  # noqa: F401

# modality -> adapter class (consumed by ModalityEncoderRegistry)
ADAPTERS = {
    "eeg": EEGAdapter,
    "wearable_phys": BiosignalAdapter,
    "cgm": CGMAdapter,
    "ehr": EHRAdapter,
    "omics": OmicsAdapter,
    "behavior": BehaviorAdapter,
}
