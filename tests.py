import os
import sys
import io
import glob
import json
import time
import hashlib
import subprocess
from pathlib import Path
import requests
import pandas as pd

sys.path.append(str((Path(__file__).parent / "src").resolve()))
from io_utils import fetch_csv_from_url  

# -------------------------------
# Helpers
# -------------------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"

DEFAULT_HBN_PHENO_URL = (
    "http://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/File/_pheno/HBN_R11_Pheno.csv"
)

REQUIRED_COLS_ANY = {"EID", "Identifiers"}
TD_REQUIRED_ANY = {"logk_mean", "ed50_mean"}  
TD_OPTIONAL = {"k_mean", "k_abs_diff"}        


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _latest_csv_in(folder: Path) -> Path | None:
    files = sorted(folder.rglob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


# -------------------------------
# 1) API fetch test (requirement)
# -------------------------------
def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    # strip whitespace + leading BOM
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    return df

ID_CANDIDATES = {"_EID","EID","eid","Identifiers","identifiers",
                 "participant_id","participantid","subject","subjectkey","id"}

def test_api_fetch_hbn_pheno():
    """
    Demonstrate programmatic access: HTTP GET -> pandas.read_csv, simple assertions,
    robust to BOM/alias header names.
    """
    url = os.environ.get("HBN_PUBLIC_CSV_URL", DEFAULT_HBN_PHENO_URL)
    print(f"[API] Fetching: {url}")

    r = requests.get(url, timeout=60)
    r.raise_for_status()

    # Use utf-8-sig to auto-strip BOM if present
    df = pd.read_csv(io.StringIO(r.content.decode("utf-8-sig")), low_memory=False)
    df = _normalize_cols(df)

    print("[API] Loaded shape:", df.shape)
    print("[API] Columns:", df.columns.tolist())

    assert df.shape[0] >= 300, "API table should have at least 300 rows"

    # Accept any reasonable ID alias
    colset = {c for c in df.columns}
    if not (ID_CANDIDATES & {c if isinstance(c,str) else c for c in colset}):
        raise AssertionError(
            "Missing ID column. Expected one of: "
            + ", ".join(sorted(ID_CANDIDATES))
        )

    # Checksum to show it's real data
    checksum = hashlib.md5(r.content[:10000]).hexdigest()
    print("[API] Head checksum (md5 of first 10KB):", checksum)

# -------------------------------
# 2) Pipeline execution test
# -------------------------------
def test_run_pipeline_and_outputs():
    """
    Run exported script once to materialize processed outputs (if needed),
    then verify that processed CSVs exist and contain key TD features.
    """
    script = ROOT / "hbn_data_processing_pipeline.py"
    assert script.exists(), f"Pipeline script not found: {script}"

    # Create folders if missing
    PROCESSED.mkdir(parents=True, exist_ok=True)

    # Try running the pipeline script (idempotent expected)
    # If your script requires args, add them here.
    print(f"[PIPELINE] Running: {script.name}")
    start = time.time()
    try:
        subprocess.run([sys.executable, str(script)], check=True)
    except subprocess.CalledProcessError as e:
        raise AssertionError(f"Pipeline script failed with code {e.returncode}") from e
    print(f"[PIPELINE] Finished in {time.time() - start:.1f}s")

    # Find a processed CSV
    cand = _latest_csv_in(PROCESSED)
    assert cand is not None, "No processed CSV found in data/processed/"
    print("[PIPELINE] Latest processed CSV:", cand.name, f"({cand.stat().st_size/1e6:.2f} MB)")

    dfp = pd.read_csv(cand, low_memory=False)
    print("[PIPELINE] Processed shape:", dfp.shape)

    # Core checks: enough rows + key TD columns
    assert dfp.shape[0] >= 1000, "Processed table should have ample rows"
    missing_td = [c for c in TD_REQUIRED_ANY if c not in dfp.columns]
    assert not missing_td, f"Missing required TD columns in processed output: {missing_td}"

    # Optional columns (warn only)
    warn_missing = [c for c in TD_OPTIONAL if c not in dfp.columns]
    if warn_missing:
        print("[PIPELINE][WARN] Optional TD columns not found:", warn_missing)

    # Missingness check (keys should be mostly present)
    td_miss = dfp[list(TD_REQUIRED_ANY)].isna().mean()
    print("[PIPELINE] Missingness (TD required):")
    print(td_miss)
    assert (td_miss <= 0.10).all(), "Too much missingness in key TD features (>10%)"

    # Save a small provenance file to summarize what was validated
    meta = {
        "processed_csv": cand.name,
        "rows": int(dfp.shape[0]),
        "cols": int(dfp.shape[1]),
        "sha256": _sha256(cand),
        "td_required_present": list(TD_REQUIRED_ANY),
        "td_optional_present": [c for c in TD_OPTIONAL if c in dfp.columns],
        "td_missing_pct": td_miss.to_dict(),
        "script": script.name,
    }
    out_meta = PROCESSED / "validation_metadata.json"
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)
    print("[PIPELINE] Wrote:", out_meta)


# -------------------------------
# Entry point
# -------------------------------
if __name__ == "__main__":
    test_api_fetch_hbn_pheno()
    test_run_pipeline_and_outputs()
    print("ALL TESTS PASSED ✅")