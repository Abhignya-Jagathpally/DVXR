"""Backward-compat shim: goal1_pipeline.clinical_tasks -> dvxr.clinical_tasks. Auto-generated; do not edit."""
from dvxr import clinical_tasks as _m
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
del _m
