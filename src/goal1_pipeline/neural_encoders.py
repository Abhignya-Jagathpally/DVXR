"""Backward-compat shim: goal1_pipeline.neural_encoders -> dvxr.neural_encoders. Auto-generated; do not edit."""
from dvxr import neural_encoders as _m
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
del _m
