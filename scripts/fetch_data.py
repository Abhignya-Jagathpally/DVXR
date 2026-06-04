from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REAL_DIR = ROOT / "data" / "real"

# Credential-free public sources (download here and now, no Kaggle account needed).
NONEEG_BASE = "https://physionet.org/files/noneeg/1.0.0"
# AccTempEDA carries the .atr phase annotations; SpO2HR has only signal + header.
NONEEG_RECORD_EXTS = {"AccTempEDA": ["dat", "hea", "atr"], "SpO2HR": ["dat", "hea"]}

MIMIC_BASE = "https://physionet.org/files/mimic-iv-demo/2.2/hosp"
MIMIC_FILES = ["patients.csv.gz", "admissions.csv.gz", "d_labitems.csv.gz", "labevents.csv.gz"]

# Shanghai T1DM/T2DM diabetes dataset (Zhao et al., Scientific Data 2023). Open on figshare;
# the article archive is a zip-in-zip of per-patient Excel CGM files.
SHANGHAI_URL = "https://ndownloader.figshare.com/articles/20444397/versions/3"


def _download(url: str, dest: Path) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as response, dest.open("wb") as handle:
        handle.write(response.read())
    return dest.stat().st_size


def fetch_shanghai_cgm() -> Path:
    """Download the open Shanghai T1DM/T2DM CGM dataset and extract per-patient Excel files."""
    import io
    import zipfile

    out = REAL_DIR / "shanghai_cgm"
    out.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(SHANGHAI_URL, headers={"User-Agent": "Mozilla/5.0"})
    archive = urllib.request.urlopen(req, timeout=180).read()

    outer = zipfile.ZipFile(io.BytesIO(archive))
    inner = zipfile.ZipFile(io.BytesIO(outer.read("data.zip")))
    count = 0
    for name in inner.namelist():
        base = name.rsplit("/", 1)[-1]
        if not base.lower().endswith(".xlsx") or base.startswith("~$") or not base[0].isdigit():
            continue  # keep per-patient files (start with a digit); skip lock/summary files
        cohort = "T1DM" if "T1DM" in name else ("T2DM" if "T2DM" in name else "unknown")
        dest = out / f"{cohort}__{base}"
        dest.write_bytes(inner.read(name))
        count += 1
    print(f"  extracted {count} patient CGM workbooks")
    print(f"Shanghai CGM dataset -> {out}")
    return out


def fetch_noneeg(subjects: int) -> Path:
    """Download the PhysioNet Non-EEG stress dataset (WFDB) for the first N subjects."""
    out = REAL_DIR / "noneeg"
    out.mkdir(parents=True, exist_ok=True)
    fetched = 0
    for subject in range(1, subjects + 1):
        try:
            total = 0
            for rec, exts in NONEEG_RECORD_EXTS.items():
                for ext in exts:
                    name = f"Subject{subject}_{rec}.{ext}"
                    total += _download(f"{NONEEG_BASE}/{name}", out / name)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                print(f"  Subject{subject}: not in dataset, stopping.")
                break
            raise
        fetched += 1
        print(f"  Subject{subject}: {total / 1e3:.0f} KB")
    print(f"Fetched {fetched} subjects.")
    print(f"Non-EEG stress dataset -> {out}")
    return out


def fetch_mimic_demo() -> Path:
    """Download the open MIMIC-IV clinical database demo hosp tables (no credentials)."""
    out = REAL_DIR / "mimic_demo" / "hosp"
    out.mkdir(parents=True, exist_ok=True)
    for name in MIMIC_FILES:
        size = _download(f"{MIMIC_BASE}/{name}?download", out / name)
        print(f"  {name}: {size / 1e6:.2f} MB")
    print(f"MIMIC-IV demo -> {out}")
    return out


def fetch_kaggle(slug: str) -> Path:
    """Download a Kaggle dataset via kagglehub. Requires Kaggle credentials."""
    try:
        import kagglehub
    except ImportError:
        sys.exit("kagglehub is not installed. Run: pip install kagglehub")

    import os

    if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")) and not (
        Path.home() / ".kaggle" / "kaggle.json"
    ).exists():
        sys.exit(
            "Kaggle credentials not found. Set KAGGLE_USERNAME and KAGGLE_KEY, or place a\n"
            "kaggle.json token at ~/.kaggle/kaggle.json (https://www.kaggle.com/settings -> API).\n"
            f"Then re-run. Target dataset: {slug}"
        )
    path = Path(kagglehub.dataset_download(slug))
    print(f"Kaggle {slug} -> {path}")
    return path


KAGGLE_SLUGS = {
    "kaggle-wesad": "orvile/wesad-wearable-stress-affect-detection-dataset",
    "kaggle-deap": "manh123df/deap-dataset",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download real public datasets for the pipeline.")
    parser.add_argument(
        "source",
        choices=["noneeg", "mimic-demo", "shanghai-cgm", "kaggle-wesad", "kaggle-deap", "all-free"],
        help="'all-free' downloads the credential-free sources (noneeg + mimic-demo + shanghai-cgm).",
    )
    parser.add_argument("--subjects", type=int, default=4, help="Number of Non-EEG subjects to fetch.")
    args = parser.parse_args()

    if args.source in ("noneeg", "all-free"):
        fetch_noneeg(args.subjects)
    if args.source in ("mimic-demo", "all-free"):
        fetch_mimic_demo()
    if args.source in ("shanghai-cgm", "all-free"):
        fetch_shanghai_cgm()
    if args.source in KAGGLE_SLUGS:
        fetch_kaggle(KAGGLE_SLUGS[args.source])


if __name__ == "__main__":
    main()
