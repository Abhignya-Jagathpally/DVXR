"""dvxr.ingest.profile — profile the local data/ tree and propose a mapping into
the 13-column canonical event schema (ARCHITECTURE / MASTER_BRIEF §1.2).

Guardrail: FAIL LOUDLY on files that cannot be mapped rather than silently
coercing. ``profile_data_dir(strict=True)`` raises on unmapped files; the default
collects them into an ``UNMAPPED`` section of the report so a human can decide.
"""
from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from dvxr.schemas import REQUIRED_EVENT_COLUMNS

ALPHA = re.compile(r"[A-Za-z]")

# path fragment -> (canonical modality, source_system, device)
_MODALITY_RULES = [
    ("emotiv", ("eeg", "EmotivPRO", "EPOCX")),
    ("openbci", ("eeg", "OpenBCI_GUI", "Galea_BoardGaleaBeta")),
    ("galea", ("eeg", "OpenBCI_GUI", "Galea_BoardGaleaBeta")),
    ("noneeg", ("wearable_phys", "physionet_noneeg", "wrist_noneeg")),
    ("cgmacros", ("cgm", "cgmacros_physionet", "cgmacros")),  # before "cgm": "cgm" is a substring
    ("shanghai", ("cgm", "shanghai_diabetes", "cgm")),
    ("cgm", ("cgm", "shanghai_diabetes", "cgm")),
    ("mimic", ("ehr", "mimic_iv_demo", "ehr")),
    ("omics", ("omics", "omics_panel", "omics")),
    ("deap", ("eeg", "deap", "deap_lab")),
    ("wesad", ("wearable_phys", "wesad", "wesad_wrist")),
]

_SKIP_NAMES = {".gitkeep", ".gitignore", ".DS_Store"}
_SMALL_BYTES = 20_000


@dataclass
class FileProfile:
    path: str
    size: int
    ext: str
    modality: Optional[str]
    source_system: Optional[str]
    device: Optional[str]
    delimiter: Optional[str]
    n_columns: Optional[int]
    header: str
    mapped: bool
    note: str = ""


@dataclass
class DataProfileReport:
    root: str
    files: List[FileProfile] = field(default_factory=list)
    unmapped: List[FileProfile] = field(default_factory=list)

    def modality_coverage(self) -> Dict[str, int]:
        cov: Dict[str, int] = {}
        for f in self.files:
            if f.modality:
                cov[f.modality] = cov.get(f.modality, 0) + 1
        return cov


def _infer_modality(rel_path: str):
    low = rel_path.lower()
    for frag, triple in _MODALITY_RULES:
        if frag in low:
            return triple
    return (None, None, None)


def _sniff(path: Path) -> tuple[Optional[str], Optional[int], str]:
    """Return (delimiter, n_columns, header_preview) for text-ish files."""
    ext = path.suffix.lower()
    if ext in {".json"}:
        return None, None, "<json metadata>"
    if ext in {".xlsx", ".gz", ".dat", ".pkl", ".atr", ".hea", ".pt", ".npy", ".png", ".html"}:
        return None, None, f"<binary/{ext.lstrip('.')}>"
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            first = fh.readline().rstrip("\n")
    except Exception as exc:  # pragma: no cover
        return None, None, f"<unreadable: {exc}>"
    header = first[:200]
    if "\t" in first:
        return "tab", len(first.split("\t")), header
    if "," in first:
        return "comma", len(first.split(",")), header
    if first.strip():
        return "space", len(first.split()), header
    return None, None, header


def profile_data_dir(
    path: str | Path = "data",
    report_path: str | Path = "outputs/data_schema_report.md",
    strict: bool = False,
) -> DataProfileReport:
    """Walk ``path``, profile every file, propose canonical mapping, write report.

    Raises RuntimeError (with the list of offending files) when ``strict`` and any
    file cannot be assigned a canonical modality.
    """
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"data path does not exist: {root.resolve()}")

    report = DataProfileReport(root=str(root))
    for dirpath, _dirs, files in os.walk(root):
        for name in sorted(files):
            if name in _SKIP_NAMES:
                continue
            # Skip transient fetch scratch (download logs / partial archives named _*),
            # which are not dataset files and would otherwise fail strict profiling.
            if name.endswith(".log") or (name.startswith("_") and name.endswith((".log", ".zip"))):
                continue
            p = Path(dirpath) / name
            rel = str(p.relative_to(root))
            modality, source, device = _infer_modality(rel)
            delim, ncols, header = _sniff(p)
            fp = FileProfile(
                path=rel, size=p.stat().st_size, ext=p.suffix.lower(),
                modality=modality, source_system=source, device=device,
                delimiter=delim, n_columns=ncols, header=header,
                mapped=modality is not None,
                note="" if modality else "no modality rule matched this path",
            )
            report.files.append(fp)
            if not fp.mapped:
                report.unmapped.append(fp)

    if strict and report.unmapped:
        offenders = "\n".join(f"  - {f.path}" for f in report.unmapped)
        raise RuntimeError(
            "profile_data_dir(strict=True): could not map these files into the "
            f"canonical schema — refusing to coerce:\n{offenders}")

    _write_report(report, Path(report_path))
    return report


def _write_report(report: DataProfileReport, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = io.StringIO()
    lines.write("# Local `data/` Schema & Canonical-Mapping Report (auto-generated)\n\n")
    lines.write("Generated by `dvxr.ingest.profile.profile_data_dir`.\n\n")
    lines.write("Canonical target columns:\n\n```\n")
    lines.write(", ".join(REQUIRED_EVENT_COLUMNS) + "\n```\n\n")

    cov = report.modality_coverage()
    lines.write("## Modality coverage\n\n| modality | files |\n|---|---|\n")
    for m in ("eeg", "wearable_phys", "cgm", "ehr", "omics", "behavior"):
        lines.write(f"| {m} | {cov.get(m, 0)} |\n")
    lines.write("\n")

    lines.write("## File inventory\n\n")
    lines.write("| file | bytes | modality | source_system | device | cols | header |\n")
    lines.write("|---|---|---|---|---|---|---|\n")
    for f in report.files:
        hdr = f.header.replace("|", "/")[:60]
        lines.write(
            f"| {f.path} | {f.size:,} | {f.modality or '—'} | "
            f"{f.source_system or '—'} | {f.device or '—'} | {f.n_columns or '—'} | {hdr} |\n")
    lines.write("\n")

    if report.unmapped:
        lines.write("## UNMAPPED — needs a human decision (not coerced)\n\n")
        for f in report.unmapped:
            lines.write(f"- `{f.path}` — {f.note}\n")
        lines.write("\n")
    else:
        lines.write("## UNMAPPED\n\n_None — every file matched a modality rule._\n\n")

    out.write_text(lines.getvalue())
