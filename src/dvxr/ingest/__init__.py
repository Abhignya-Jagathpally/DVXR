"""dvxr.ingest — canonical ingestion: loaders re-export + data profiling."""
from dvxr.loaders import *  # noqa: F401,F403
try:
    from .profile import profile_data_dir  # noqa: F401
except Exception:  # pragma: no cover - profile added in Stage 1b
    pass
