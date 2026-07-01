"""Backward-compat shim: goal1_pipeline.sample_data -> dvxr.sample_data. Auto-generated; do not edit."""
from dvxr import sample_data as _m
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
del _m
