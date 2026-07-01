"""Backward-compat shim: goal1_pipeline.models -> dvxr.models. Auto-generated; do not edit."""
from dvxr import models as _m
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
del _m
