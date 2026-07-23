"""Bridge from ``dvxr`` to the ``neuroglycemic-sentinel`` glucose forecaster.

The sentinel package is a separate installable project with its own dependency pins and
an unusual ``src``-as-package layout. Importing it into the ``dvxr`` interpreter would
collide with ``dvxr``'s own ``src/`` and risk pin conflicts, so this bridge drives it
**out-of-process**: it builds the exact ``main.py`` argv, runs it with the sentinel repo
as the working directory, and reads the auditable artifacts the CLI writes into the
external runtime workspace.

Honesty boundary: this module launches a process and reads files. It never computes,
imputes, or edits a forecast number. Every metric returned here was written by the
sentinel CLI, which enforces patient-disjoint splits and superiority gates. Artifacts
live outside the repository (in ``neuroglycemic-runtime/``) and are never promoted into
``dvxr``'s committed ``outputs/`` scoreboards.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

# repo_root/src/dvxr/integrations/glucose_forecasting.py -> parents[3] == repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]

#: Artifact filenames written by ``run_neural_train`` into ``runs/<run>/``.
_TRAIN_ARTIFACTS = (
    "test_metrics.json",
    "training_acceptance.json",
    "test_predictions.csv",
    "patient_split.csv",
    "missing_modality_ablation.csv",
    "feature_schema.json",
    "model_card.json",
    "training_losses.csv",
)
#: Additional artifacts written by ``run_neural_evaluate``.
_EVALUATE_ARTIFACTS = (
    "reloaded_test_metrics.json",
    "reloaded_test_predictions.csv",
    "reloaded_missing_modality_ablation.csv",
    "evaluation_reproducibility.json",
)
#: Figures emitted by the sentinel training run (under ``runs/<run>/figures/``).
_TRAIN_FIGURES = (
    "training_loss.png",
    "held_out_forecasts.png",
    "fusion_weights.png",
)


class SentinelCommandError(RuntimeError):
    """Raised when a sentinel CLI invocation exits non-zero."""

    def __init__(self, argv: Sequence[str], returncode: int, stdout: str, stderr: str):
        self.argv = list(argv)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        tail = (stderr or stdout or "").strip().splitlines()[-20:]
        super().__init__(
            f"sentinel command failed (exit {returncode}): "
            f"{' '.join(self.argv)}\n" + "\n".join(tail)
        )


@dataclass(frozen=True)
class GlucoseRunArtifacts:
    """Resolved paths + parsed metrics for one sentinel run directory."""

    run_name: str
    run_directory: Path
    artifacts: Mapping[str, Path]
    figures: Mapping[str, Path]
    checkpoint: Path | None
    metrics: Mapping[str, Any] | None
    acceptance: Mapping[str, Any] | None
    reproducibility: Mapping[str, Any] | None

    def exists(self) -> bool:
        return self.run_directory.is_dir()


@dataclass(frozen=True)
class GlucoseForecastingBridge:
    """Drive the ``neuroglycemic-sentinel`` CLI from ``dvxr``.

    Parameters default to the sibling-directory layout of this repository, so
    ``GlucoseForecastingBridge()`` works with no arguments on a standard checkout.
    """

    sentinel_repo: Path = _REPO_ROOT / "neuroglycemic-sentinel"
    runtime_root: Path = _REPO_ROOT / "neuroglycemic-runtime"
    python_executable: str = sys.executable
    thread_cap: int = int(os.environ.get("DVXR_GLUCOSE_THREAD_CAP", "2"))
    extra_env: Mapping[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------ helpers
    def _resolved(self) -> tuple[Path, Path]:
        repo = Path(self.sentinel_repo).expanduser().resolve()
        runtime = Path(self.runtime_root).expanduser().resolve()
        if not (repo / "main.py").is_file():
            raise FileNotFoundError(
                f"sentinel repo not found (no main.py at {repo}). "
                "Pass sentinel_repo=... to GlucoseForecastingBridge."
            )
        return repo, runtime

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        cap = str(max(1, int(self.thread_cap)))
        # Shared multi-user host: cap BLAS/OMP threads so heavy runs do not thrash.
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            env.setdefault(var, cap)
        env.update({k: str(v) for k, v in self.extra_env.items()})
        return env

    def run_cli(
        self,
        args: Sequence[str],
        *,
        timeout: float | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """Run ``python main.py <args>`` in the sentinel repo, capturing output."""
        repo, _ = self._resolved()
        argv = [self.python_executable, "main.py", *[str(a) for a in args]]
        proc = subprocess.run(
            argv,
            cwd=str(repo),
            env=self._env(),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if check and proc.returncode != 0:
            raise SentinelCommandError(argv, proc.returncode, proc.stdout, proc.stderr)
        return proc

    def _abs(self, path: Path | str) -> str:
        return str(Path(path).expanduser().resolve())

    # ------------------------------------------------------------------ commands
    def prepare_diatrend(
        self,
        source_dir: Path | str,
        source_timezone: str,
        *,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Build causal DiaTrend windows into ``runtime/aligned/``."""
        _, runtime = self._resolved()
        return self.run_cli(
            [
                "prepare-diatrend",
                "--workspace", self._abs(runtime),
                "--source-dir", self._abs(source_dir),
                "--source-timezone", source_timezone,
            ],
            timeout=timeout,
        )

    def prepare_big_ideas(
        self,
        source_dir: Path | str,
        source_timezone: str,
        *,
        horizons_minutes: Sequence[int] | None = None,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Build causal BIG-IDEAS wearable/CGM windows into ``runtime/aligned/``."""
        _, runtime = self._resolved()
        args = [
            "prepare-big-ideas",
            "--workspace", self._abs(runtime),
            "--source-dir", self._abs(source_dir),
            "--source-timezone", source_timezone,
        ]
        if horizons_minutes:
            args += ["--horizons-minutes", *[str(h) for h in horizons_minutes]]
        return self.run_cli(args, timeout=timeout)

    def train(
        self,
        data: Path | str,
        config: Path | str,
        run_name: str,
        *,
        batch_size: int | None = None,
        pretrain_epochs: int | None = None,
        timeout: float | None = None,
    ) -> GlucoseRunArtifacts:
        """Run ``train-neural`` and return the resolved run artifacts."""
        _, runtime = self._resolved()
        args = [
            "train-neural",
            "--data", self._abs(data),
            "--config", self._abs(config),
            "--workspace", self._abs(runtime),
            "--run-name", run_name,
        ]
        if batch_size is not None:
            args += ["--batch-size", str(batch_size)]
        if pretrain_epochs is not None:
            args += ["--pretrain-epochs", str(pretrain_epochs)]
        self.run_cli(args, timeout=timeout)
        return self.artifacts(run_name)

    def evaluate(
        self,
        data: Path | str,
        config: Path | str,
        run_name: str,
        *,
        batch_size: int | None = None,
        timeout: float | None = None,
    ) -> GlucoseRunArtifacts:
        """Run ``evaluate-neural`` (deterministic re-evaluation) and return artifacts."""
        _, runtime = self._resolved()
        args = [
            "evaluate-neural",
            "--data", self._abs(data),
            "--config", self._abs(config),
            "--workspace", self._abs(runtime),
            "--run-name", run_name,
        ]
        if batch_size is not None:
            args += ["--batch-size", str(batch_size)]
        self.run_cli(args, timeout=timeout)
        return self.artifacts(run_name)

    # ------------------------------------------------------------------ readers
    def run_directory(self, run_name: str) -> Path:
        _, runtime = self._resolved()
        return runtime / "runs" / run_name

    def checkpoint_path(self, run_name: str) -> Path:
        _, runtime = self._resolved()
        return runtime / "models" / f"{run_name}.pt"

    @staticmethod
    def _load_json(path: Path) -> Any | None:
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def artifacts(self, run_name: str) -> GlucoseRunArtifacts:
        """Resolve every known artifact/figure for ``run_name`` that exists on disk."""
        run_dir = self.run_directory(run_name)
        figure_dir = run_dir / "figures"
        artifacts = {
            name: run_dir / name
            for name in (*_TRAIN_ARTIFACTS, *_EVALUATE_ARTIFACTS)
            if (run_dir / name).is_file()
        }
        figures = {
            name: figure_dir / name
            for name in _TRAIN_FIGURES
            if (figure_dir / name).is_file()
        }
        # Include any extra figures a later step (e.g. the DiaTrend overview) added.
        if figure_dir.is_dir():
            for extra in sorted(figure_dir.glob("*.png")):
                figures.setdefault(extra.name, extra)
        checkpoint = self.checkpoint_path(run_name)
        return GlucoseRunArtifacts(
            run_name=run_name,
            run_directory=run_dir,
            artifacts=artifacts,
            figures=figures,
            checkpoint=checkpoint if checkpoint.is_file() else None,
            metrics=self._load_json(run_dir / "test_metrics.json"),
            acceptance=self._load_json(run_dir / "training_acceptance.json"),
            reproducibility=self._load_json(run_dir / "evaluation_reproducibility.json"),
        )
