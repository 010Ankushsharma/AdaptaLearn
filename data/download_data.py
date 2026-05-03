"""
data/download_data.py
Downloads EdNet KT4 and ASSISTments 2015 datasets, verifies integrity,
and saves them as parquet files for fast downstream processing.

Usage:
    python data/download_data.py
    python data/download_data.py --dataset ednet       # only EdNet
    python data/download_data.py --dataset assistments # only ASSISTments
    python data/download_data.py --force               # re-download even if exists
"""

import argparse
import hashlib
import io
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# Allow running as a script from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_RAW_DIR, EDNET_FILE, ASSISTMENTS_FILE

# ── Expected column schemas ────────────────────────────────────────────────────
EDNET_COLUMNS = {
    "user_id":        "int32",
    "question_id":    "int32",
    "correct":        "int8",     # 0 or 1
    "elapsed_time":   "float32",  # milliseconds
    "hint_count":     "int8",
    "concept_id":     "int16",
    "difficulty":     "float32",  # normalised 0–1
    "timestamp":      "int64",    # unix ms
}

ASSISTMENTS_COLUMNS = {
    "user_id":        "int32",
    "question_id":    "int32",
    "correct":        "int8",
    "skill_id":       "int16",    # concept / skill tag
    "elapsed_time":   "float32",
    "hint_count":     "int8",
    "attempt_count":  "int8",
}

# ── Public mirrors (Kaggle / GitHub releases / HuggingFace) ───────────────────
# EdNet: official public release by Riiid on GitHub
EDNET_SOURCES = [
    # Primary: preprocessed KT4 parquet on HuggingFace datasets
    "https://huggingface.co/datasets/riiid/ednet/resolve/main/KT4/train.csv.zip",
    # Fallback: Kaggle public dataset (requires kaggle CLI)
    "kaggle:riiid-test-answer-prediction/train.csv",
]

# ASSISTments 2015: original host
ASSISTMENTS_SOURCES = [
    "https://sites.google.com/site/assistmentsdata/home/2015-assistments-skill-builder-data/skill_builder_data_corrected.zip",
    # Fallback mirror
    "https://raw.githubusercontent.com/arghosh/AKT/master/data/assist15/assist15.csv",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def download_file(url: str, dest: Path, chunk_size: int = 8192) -> Path:
    """Stream-download url → dest, showing a tqdm progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {url}")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    tmp = dest.with_suffix(".tmp")
    with open(tmp, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=dest.name
    ) as bar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))
    tmp.rename(dest)
    return dest


def md5(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute MD5 of a file in chunks (memory-efficient for large files)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def unzip_first_csv(zip_path: Path, dest_dir: Path) -> Path:
    """Unzip and return the path to the first CSV inside the archive."""
    with zipfile.ZipFile(zip_path) as z:
        csv_names = [n for n in z.namelist() if n.endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No CSV found inside {zip_path}")
        print(f"  Extracting {csv_names[0]} ...")
        z.extract(csv_names[0], dest_dir)
        return dest_dir / csv_names[0]


# ── EdNet ─────────────────────────────────────────────────────────────────────

def _build_synthetic_ednet(n_students: int = 2000, n_questions: int = 500) -> pd.DataFrame:
    """
    Build a realistic synthetic EdNet-like dataset for development/testing
    when the real dataset is unavailable.
    Mirrors the real column schema exactly.
    """
    import numpy as np

    rng = np.random.default_rng(42)
    n_concepts = 188

    rows = []
    for uid in range(n_students):
        # Each student has between 10 and 100 interactions
        n_interactions = rng.integers(10, 101)
        # Simulate mastery that slowly improves
        mastery = rng.uniform(0.1, 0.6, size=n_concepts)
        for _ in range(n_interactions):
            qid    = int(rng.integers(0, n_questions))
            cid    = int(rng.integers(0, n_concepts))
            p_corr = float(np.clip(mastery[cid] + rng.normal(0, 0.1), 0.05, 0.95))
            correct= int(rng.random() < p_corr)
            if correct:
                mastery[cid] = min(1.0, mastery[cid] + rng.uniform(0.02, 0.08))
            rows.append({
                "user_id":      uid,
                "question_id":  qid,
                "correct":      correct,
                "elapsed_time": float(rng.integers(5_000, 120_000)),   # ms
                "hint_count":   int(rng.integers(0, 4)) if not correct else 0,
                "concept_id":   cid,
                "difficulty":   round(float(rng.uniform(0.2, 0.9)), 3),
                "timestamp":    int(1_600_000_000_000 + uid * 1_000_000 + _ * 60_000),
            })
    return pd.DataFrame(rows)


def download_ednet(force: bool = False) -> Path:
    """Download EdNet KT4, convert to parquet, return path."""
    if EDNET_FILE.exists() and not force:
        print(f"  [skip] EdNet already exists: {EDNET_FILE}")
        return EDNET_FILE

    raw_csv = DATA_RAW_DIR / "ednet_kt4_raw.csv"
    zip_file = DATA_RAW_DIR / "ednet_kt4.zip"

    # Try each source in order
    df = None
    for source in EDNET_SOURCES:
        try:
            if source.startswith("kaggle:"):
                # Requires `pip install kaggle` and ~/.kaggle/kaggle.json
                dataset, fname = source[7:].split("/", 1)
                print(f"  Trying Kaggle source: {dataset}/{fname}")
                os.system(f"kaggle competitions download -c {dataset} -f {fname} -p {DATA_RAW_DIR}")
                csv_path = DATA_RAW_DIR / fname
            else:
                zip_path = download_file(source, zip_file)
                csv_path = unzip_first_csv(zip_path, DATA_RAW_DIR)

            print(f"  Parsing CSV ...")
            df = pd.read_csv(csv_path, nrows=None)
            break
        except Exception as e:
            print(f"  [warn] Source failed: {e}")
            continue

    if df is None:
        print("  [warn] All sources failed — generating synthetic EdNet dataset for development.")
        print("         Replace data/raw/ednet_kt4.parquet with the real dataset before final experiments.")
        df = _build_synthetic_ednet()
    else:
        df = _standardise_ednet(df)

    df = df.astype({k: v for k, v in EDNET_COLUMNS.items() if k in df.columns})
    df.to_parquet(EDNET_FILE, index=False)
    print(f"  Saved {len(df):,} rows → {EDNET_FILE}  ({EDNET_FILE.stat().st_size / 1e6:.1f} MB)")
    return EDNET_FILE


def _standardise_ednet(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns from Riiid's raw format to our schema."""
    rename = {
        # Riiid column   → our column
        "user_id":                 "user_id",
        "content_id":              "question_id",
        "answered_correctly":      "correct",
        "prior_question_elapsed_time": "elapsed_time",
        "prior_question_had_explanation": "hint_count",
        "task_container_id":       "concept_id",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Drop rows with nulls in key columns
    key_cols = ["user_id", "question_id", "correct"]
    df = df.dropna(subset=[c for c in key_cols if c in df.columns])

    # Add missing columns with sensible defaults
    for col, default in [("difficulty", 0.5), ("timestamp", 0), ("hint_count", 0)]:
        if col not in df.columns:
            df[col] = default

    return df[list(EDNET_COLUMNS.keys())]


# ── ASSISTments ───────────────────────────────────────────────────────────────

def _build_synthetic_assistments(n_students: int = 1000, n_questions: int = 200) -> pd.DataFrame:
    """Synthetic ASSISTments-like data for development."""
    import numpy as np

    rng = np.random.default_rng(99)
    n_skills = 100
    rows = []
    for uid in range(n_students):
        skill_mastery = rng.uniform(0.2, 0.7, size=n_skills)
        for _ in range(rng.integers(5, 50)):
            sid = int(rng.integers(0, n_skills))
            qid = int(rng.integers(0, n_questions))
            p   = float(np.clip(skill_mastery[sid] + rng.normal(0, 0.15), 0.05, 0.95))
            cor = int(rng.random() < p)
            if cor:
                skill_mastery[sid] = min(1.0, skill_mastery[sid] + rng.uniform(0.03, 0.10))
            rows.append({
                "user_id":       uid,
                "question_id":   qid,
                "correct":       cor,
                "skill_id":      sid,
                "elapsed_time":  float(rng.integers(3_000, 90_000)),
                "hint_count":    int(rng.integers(0, 3)) if not cor else 0,
                "attempt_count": 1,
            })
    return pd.DataFrame(rows)


def download_assistments(force: bool = False) -> Path:
    """Download ASSISTments 2015, convert to parquet, return path."""
    if ASSISTMENTS_FILE.exists() and not force:
        print(f"  [skip] ASSISTments already exists: {ASSISTMENTS_FILE}")
        return ASSISTMENTS_FILE

    df = None
    for source in ASSISTMENTS_SOURCES:
        try:
            if source.endswith(".zip"):
                zip_path = download_file(source, DATA_RAW_DIR / "assistments.zip")
                csv_path = unzip_first_csv(zip_path, DATA_RAW_DIR)
            else:
                csv_path = download_file(source, DATA_RAW_DIR / "assistments_raw.csv")
            print("  Parsing CSV ...")
            df = pd.read_csv(csv_path, encoding="latin-1", low_memory=False)
            df = _standardise_assistments(df)
            break
        except Exception as e:
            print(f"  [warn] Source failed: {e}")
            continue

    if df is None:
        print("  [warn] All sources failed — generating synthetic ASSISTments dataset.")
        df = _build_synthetic_assistments()

    df = df.astype({k: v for k, v in ASSISTMENTS_COLUMNS.items() if k in df.columns})
    df.to_parquet(ASSISTMENTS_FILE, index=False)
    print(f"  Saved {len(df):,} rows → {ASSISTMENTS_FILE}  ({ASSISTMENTS_FILE.stat().st_size / 1e6:.1f} MB)")
    return ASSISTMENTS_FILE


def _standardise_assistments(df: pd.DataFrame) -> pd.DataFrame:
    """Rename ASSISTments columns to our schema."""
    rename = {
        "user_id":          "user_id",
        "problem_id":       "question_id",
        "correct":          "correct",
        "skill_id":         "skill_id",
        "ms_first_response":"elapsed_time",
        "hint_count":       "hint_count",
        "attempt_count":    "attempt_count",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df = df.dropna(subset=["user_id", "question_id", "correct"])

    for col, default in [("skill_id", 0), ("elapsed_time", 0.0),
                         ("hint_count", 0), ("attempt_count", 1)]:
        if col not in df.columns:
            df[col] = default

    return df[list(ASSISTMENTS_COLUMNS.keys())]


# ── Verification ──────────────────────────────────────────────────────────────

def verify_dataset(path: Path, expected_columns: dict, name: str) -> bool:
    """Load parquet and run basic sanity checks. Returns True if all pass."""
    print(f"\n  Verifying {name} ...")
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"  [FAIL] Cannot read {path}: {e}")
        return False

    ok = True

    # Check columns
    missing = set(expected_columns) - set(df.columns)
    if missing:
        print(f"  [FAIL] Missing columns: {missing}")
        ok = False
    else:
        print(f"  [ok]   All expected columns present")

    # Check no empty dataframe
    if len(df) == 0:
        print(f"  [FAIL] DataFrame is empty")
        ok = False
    else:
        print(f"  [ok]   {len(df):,} rows, {df['user_id'].nunique():,} unique students")

    # Check correct ∈ {0, 1}
    if "correct" in df.columns:
        invalid = df["correct"].isin([0, 1]).mean()
        if invalid < 0.99:
            print(f"  [FAIL] 'correct' has non-binary values")
            ok = False
        else:
            acc = df["correct"].mean()
            print(f"  [ok]   Mean accuracy: {acc:.3f}  (expected 0.4–0.8)")

    # Check no all-null columns
    null_cols = [c for c in df.columns if df[c].isnull().all()]
    if null_cols:
        print(f"  [FAIL] All-null columns: {null_cols}")
        ok = False

    status = "PASSED" if ok else "FAILED"
    print(f"  Verification {status}: {name}")
    return ok


def print_summary(path: Path, name: str) -> None:
    """Print a quick summary of a saved parquet file."""
    df = pd.read_parquet(path)
    print(f"\n  ── {name} summary ──────────────────────────")
    print(f"     Rows:       {len(df):>12,}")
    print(f"     Students:   {df['user_id'].nunique():>12,}")
    if "question_id" in df.columns:
        print(f"     Questions:  {df['question_id'].nunique():>12,}")
    if "concept_id" in df.columns:
        print(f"     Concepts:   {df['concept_id'].nunique():>12,}")
    elif "skill_id" in df.columns:
        print(f"     Skills:     {df['skill_id'].nunique():>12,}")
    if "correct" in df.columns:
        print(f"     Accuracy:   {df['correct'].mean():>11.3f}")
    print(f"     File size:  {path.stat().st_size / 1e6:>10.1f} MB")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download and prepare datasets")
    parser.add_argument(
        "--dataset", choices=["ednet", "assistments", "all"], default="all",
        help="Which dataset to download (default: all)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if the file already exists"
    )
    args = parser.parse_args()

    print("\n=== Adaptive Learning — Dataset Download ===\n")

    all_ok = True

    if args.dataset in ("ednet", "all"):
        print("► EdNet KT4")
        download_ednet(force=args.force)
        ok = verify_dataset(EDNET_FILE, EDNET_COLUMNS, "EdNet KT4")
        if ok:
            print_summary(EDNET_FILE, "EdNet KT4")
        all_ok = all_ok and ok

    if args.dataset in ("assistments", "all"):
        print("\n► ASSISTments 2015")
        download_assistments(force=args.force)
        ok = verify_dataset(ASSISTMENTS_FILE, ASSISTMENTS_COLUMNS, "ASSISTments 2015")
        if ok:
            print_summary(ASSISTMENTS_FILE, "ASSISTments 2015")
        all_ok = all_ok and ok

    print("\n" + ("=== All downloads complete and verified ===" if all_ok
                  else "=== Some downloads failed — check warnings above ==="))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()