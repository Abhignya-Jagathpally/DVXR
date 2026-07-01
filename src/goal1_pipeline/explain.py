"""Backward-compat shim: goal1_pipeline.explain -> dvxr.explain. Auto-generated; do not edit."""
from dvxr import explain as _m
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
del _m
