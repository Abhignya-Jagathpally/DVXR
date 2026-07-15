"""dvxr.storage — repository interfaces (Protocols) + lightweight local implementations.

The system's stateful stores (spec §6, §11) are defined here as `typing.Protocol` contracts so model
and serving code depends on the interface, not a vendor. For the initial research deployment
(spec §10: a modular monolith is fine for hundreds of participants) the local impls in
`dvxr.storage.local` back them with sqlite / flat files — swap for postgres / a managed vector store
at scale without touching model logic.
"""
from dvxr.storage.base import (  # noqa: F401
    AuditStore,
    ClinicalStore,
    ConsentStore,
    EventStore,
    FeatureStore,
    ModelRegistry,
    PredictionStore,
    RawStore,
    VectorStore,
)
from dvxr.storage.local import (  # noqa: F401
    LocalAuditStore,
    LocalConsentStore,
    LocalModelRegistry,
    LocalPredictionStore,
    open_local_stores,
)
