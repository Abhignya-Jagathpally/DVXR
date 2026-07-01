"""Backward-compat shim: goal1_pipeline.personalization -> dvxr.personalization. Auto-generated; do not edit."""
from dvxr import personalization as _m
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
del _m
