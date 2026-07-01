"""Backward-compat shim: goal1_pipeline.calibration -> dvxr.calibration. Auto-generated; do not edit."""
from dvxr import calibration as _m
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
del _m
