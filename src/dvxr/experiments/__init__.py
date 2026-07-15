"""dvxr.experiments — EXPERIMENTAL, non-clinical research code.

Nothing in this package is part of the delivered product path. Modules here explore
research questions (e.g. the LLM-in-the-predictive-path probe) and are held to the
project's honesty gate as *negative / non-product* results: they must never be wired into
the serving path, the request lifecycle, the dashboard defaults, or any product claim. The
honesty audit (`tests/test_honesty_audit.py`) enforces that the LLM is explanation-only in
the product and never a predictor.
"""
