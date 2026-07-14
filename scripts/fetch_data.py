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

# WESAD wearable stress/affect dataset (Schmidt et al. 2018). Official ~2 GB share hosted on
# the University of Siegen sciebo instance; the ``/download`` endpoint returns WESAD.zip.
WESAD_SIEGEN_URL = "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download"

# CGMacros multimodal CGM + diet + wearable dataset (PhysioNet, open CC-BY-NC-SA 4.0).
CGMACROS_ZIP_URL = "https://physionet.org/content/cgmacros/get-zip/1.0.0/"

# EEG-during-Mental-Arithmetic (eegmat, PhysioNet, Zyma et al. 2019). REAL cognitive-workload
# label: SubjectNN_1.edf = resting baseline (low workload); SubjectNN_2.edf = serial subtraction
# (high workload). 19-ch 10-20 EEG + ECG @ 500 Hz.
EEGMAT_BASE = "https://physionet.org/files/eegmat/1.0.0"

# Mumtaz (2016) MDD-vs-healthy resting EEG (figshare 4244171, CC BY 4.0). REAL depression
# label: 'H S* EC.edf' = healthy control, 'MDD S* EC.edf' = major depressive disorder. We fetch
# only the eyes-closed (EC) resting recordings (the standard MDD resting-state condition).
MUMTAZ_ARTICLE = "4244171"


def _download(url: str, dest: Path) -> int:
    """Stream a URL to disk (urllib uses HTTP/1.1, avoiding the HTTP/2 stream errors some
    large mirrors return). Returns the byte count written."""
    import shutil

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as response, dest.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
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


def fetch_eegmat() -> Path:
    """Download the PhysioNet EEG mental-arithmetic cohort (real cognitive-workload labels)."""
    out = REAL_DIR / "eegmat"
    out.mkdir(parents=True, exist_ok=True)
    records = urllib.request.urlopen(f"{EEGMAT_BASE}/RECORDS", timeout=60).read().decode().split()
    edfs = [r for r in records if r.endswith(".edf")]
    fetched = 0
    for name in edfs:
        dest = out / name
        if dest.exists() and dest.stat().st_size > 1000:  # resume: skip already-present files
            fetched += 1
            continue
        _download(f"{EEGMAT_BASE}/{name}", dest)
        fetched += 1
    try:  # subject-info.csv carries count-quality metadata (good/bad counter)
        _download(f"{EEGMAT_BASE}/subject-info.csv", out / "subject-info.csv")
    except urllib.error.HTTPError:
        pass
    print(f"Fetched {fetched}/{len(edfs)} eegmat recordings -> {out}")
    return out


def fetch_mumtaz_mdd() -> Path:
    """Download the Mumtaz MDD-vs-healthy resting EEG cohort (eyes-closed) from figshare."""
    import json

    out = REAL_DIR / "mumtaz_mdd"
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads(urllib.request.urlopen(
        f"https://api.figshare.com/v2/articles/{MUMTAZ_ARTICLE}", timeout=60).read())
    ec = [f for f in meta["files"] if f["name"].strip().endswith("EC.edf")]
    fetched, skipped = 0, []
    for f in ec:
        name = f["name"].strip().replace("  ", " ").replace(" ", "_")  # normalize spacing
        dest = out / name
        if dest.exists() and dest.stat().st_size > 1000:  # resume
            fetched += 1
            continue
        try:  # one dead/renamed file must not abort the whole cohort fetch
            _download(f["download_url"], dest)
            fetched += 1
        except urllib.error.URLError as err:
            dest.unlink(missing_ok=True)
            skipped.append(f"{name} ({err})")
    if skipped:
        print(f"  skipped {len(skipped)} unavailable file(s): {skipped}")
    print(f"Fetched {fetched}/{len(ec)} Mumtaz MDD eyes-closed recordings -> {out}")
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


def fetch_wesad_siegen() -> Path:
    """Download the official WESAD dataset (~2 GB) from the Siegen sciebo share and extract it.

    The archive extracts to ``<REAL_DIR>/WESAD/S<n>/S<n>.pkl``, the layout expected by
    ``dvxr.loaders.load_wesad_dataset``. Credential-free.
    """
    import zipfile

    REAL_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = REAL_DIR / "_wesad.zip"
    if not (zip_path.exists() and zipfile.is_zipfile(zip_path)):
        print("Downloading WESAD (~2 GB, may take several minutes)...")
        size = _download(WESAD_SIEGEN_URL, zip_path)
        print(f"  downloaded {size / 1e6:.0f} MB")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(REAL_DIR)
    out = REAL_DIR / "WESAD"
    n = len(list(out.glob("S*/S*.pkl")))
    print(f"WESAD -> {out} ({n} subject pickles)")
    return out


def fetch_cgmacros() -> Path:
    """Download the open CGMacros dataset (~628 MB) from PhysioNet and extract it."""
    import zipfile

    REAL_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = REAL_DIR / "_cgmacros.zip"
    if not (zip_path.exists() and zipfile.is_zipfile(zip_path)):
        print("Downloading CGMacros (~628 MB)...")
        size = _download(CGMACROS_ZIP_URL, zip_path)
        print(f"  downloaded {size / 1e6:.0f} MB")
    out = REAL_DIR / "cgmacros"
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(out)
    # The PhysioNet archive is a zip-in-zip: unpack the inner CGMacros_dateshifted365.zip
    # in place so per-subject CGMacros-XXX/CGMacros-XXX.csv + bio.csv become loadable.
    for inner in out.glob("**/CGMacros_dateshifted365.zip"):
        with zipfile.ZipFile(inner) as inner_zip:
            inner_zip.extractall(inner.parent / "data")
    n = len(list(out.glob("**/CGMacros-*.csv")))
    print(f"CGMacros -> {out} ({n} subject CSVs)")
    return out


def fetch_kaggle(slug: str) -> Path:
    """Download a Kaggle dataset via kagglehub. Requires Kaggle credentials.

    Accepts either the classic username+key (KAGGLE_USERNAME/KAGGLE_KEY or
    ~/.kaggle/kaggle.json) or the newer bearer token in KAGGLE_API_TOKEN (KGAT_...).
    """
    try:
        import kagglehub
    except ImportError:
        sys.exit("kagglehub is not installed. Run: pip install kagglehub")

    import os

    has_classic = (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")) or (
        Path.home() / ".kaggle" / "kaggle.json"
    ).exists()
    has_token = bool(os.environ.get("KAGGLE_API_TOKEN"))
    if not (has_classic or has_token):
        sys.exit(
            "Kaggle credentials not found. Set KAGGLE_API_TOKEN=KGAT_... (newer tokens),\n"
            "or KAGGLE_USERNAME + KAGGLE_KEY, or place kaggle.json at ~/.kaggle/kaggle.json\n"
            f"(https://www.kaggle.com/settings -> API). Then re-run. Target dataset: {slug}"
        )
    path = Path(kagglehub.dataset_download(slug))
    print(f"Kaggle {slug} -> {path}")
    return path


def fetch_deap(link: bool = True) -> Path:
    """Download DEAP (raw or preprocessed) via kagglehub and expose it at data/real/deap.

    kagglehub caches under ~/.cache/kagglehub; we symlink the resolved dataset dir into
    data/real/deap so the loaders' default path works. load_deap_dataset auto-detects
    preprocessed .dat vs raw .bdf.
    """
    src = fetch_kaggle(KAGGLE_SLUGS["kaggle-deap"])
    out = REAL_DIR / "deap"
    out.mkdir(parents=True, exist_ok=True)
    target = out / src.name
    if link and not target.exists():
        try:
            target.symlink_to(src)
        except OSError:
            pass
    print(f"DEAP -> {out} (source {src})")
    return out


KAGGLE_SLUGS = {
    "kaggle-wesad": "orvile/wesad-wearable-stress-affect-detection-dataset",
    # DEAP raw data (BioSemi .bdf) as requested for the refactor.
    "kaggle-deap": "sayuksh/deap-datasetraw-data",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download real public datasets for the pipeline.")
    parser.add_argument(
        "source",
        choices=[
            "noneeg",
            "eegmat",
            "mumtaz-mdd",
            "mimic-demo",
            "shanghai-cgm",
            "wesad",
            "cgmacros",
            "kaggle-wesad",
            "kaggle-deap",
            "all-free",
        ],
        help=(
            "'all-free' downloads every credential-free source "
            "(noneeg + eegmat + mumtaz-mdd + mimic-demo + shanghai-cgm + wesad + cgmacros)."
        ),
    )
    parser.add_argument("--subjects", type=int, default=4, help="Number of Non-EEG subjects to fetch.")
    args = parser.parse_args()

    if args.source in ("noneeg", "all-free"):
        fetch_noneeg(args.subjects)
    if args.source in ("eegmat", "all-free"):
        fetch_eegmat()
    if args.source in ("mumtaz-mdd", "all-free"):
        fetch_mumtaz_mdd()
    if args.source in ("mimic-demo", "all-free"):
        fetch_mimic_demo()
    if args.source in ("shanghai-cgm", "all-free"):
        fetch_shanghai_cgm()
    if args.source in ("wesad", "all-free"):
        fetch_wesad_siegen()
    if args.source in ("cgmacros", "all-free"):
        fetch_cgmacros()
    if args.source == "kaggle-deap":
        fetch_deap()
    elif args.source in KAGGLE_SLUGS:
        fetch_kaggle(KAGGLE_SLUGS[args.source])


if __name__ == "__main__":
    main()
