#!/usr/bin/env python3
"""PostToolUse honesty gate for DVXR.

Runs the fast, torch-free honesty-invariant suite after an edit to source that could
regress it (``src/**`` or ``neuroglycemic-sentinel/src/**``). Blocks (exit 2) only when
the suite fails, feeding the failure back to the model. Non-source edits pass through.

Wired from ``.claude/settings.json`` as a PostToolUse hook on Edit|Write|MultiEdit.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

WATCHED = ("src/", "neuroglycemic-sentinel/src/")


def _edited_path(event: dict) -> str:
    tool_input = event.get("tool_input") or {}
    return tool_input.get("file_path") or tool_input.get("path") or ""


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except Exception:  # noqa: BLE001 — no parsable event: do not block
        return 0

    path = _edited_path(event)
    if not path:
        return 0
    try:
        rel = str(Path(path).resolve())
    except Exception:  # noqa: BLE001
        rel = path
    if not any(w in rel.replace("\\", "/") for w in WATCHED):
        return 0

    cwd = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "tests.test_honesty_audit"],
        cwd=str(cwd),
        env={"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "PATH": _path_env()},
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(
            "DVXR honesty gate FAILED after this edit — the invariant suite "
            "(tests.test_honesty_audit) regressed. Fix before continuing:\n"
            + (proc.stderr or proc.stdout)[-2000:]
        )
        return 2
    return 0


def _path_env() -> str:
    import os

    return os.environ.get("PATH", "/usr/bin:/bin")


if __name__ == "__main__":
    raise SystemExit(main())
