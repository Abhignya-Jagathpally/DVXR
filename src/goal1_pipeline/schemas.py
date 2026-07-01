"""Backward-compat shim: goal1_pipeline.schemas -> dvxr.schemas. Auto-generated; do not edit."""
from dvxr import schemas as _m
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
del _m
