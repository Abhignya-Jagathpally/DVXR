"""Backward-compat shim: goal1_pipeline.streaming -> dvxr.streaming. Auto-generated; do not edit."""
from dvxr import streaming as _m
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
del _m
