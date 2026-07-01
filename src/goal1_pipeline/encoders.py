"""Backward-compat shim: goal1_pipeline.encoders -> dvxr.encoders. Auto-generated; do not edit."""
from dvxr import encoders as _m
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
del _m
