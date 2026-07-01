"""Backward-compat shim: goal1_pipeline.bci_real -> dvxr.bci_real. Auto-generated; do not edit."""
from dvxr import bci_real as _m
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
del _m
