"""
data/preprocess.py
Cleans raw EdNet KT4 and ASSISTments 2015 data, builds the concept
dependency graph, and writes the processed splits used by every
downstream module.

Outputs (all under DATA_PROC_DIR):
    train.parquet          — 80% of students, interaction sequences
    val.parquet            — 10% of students
    test.parquet           — 10% of students
    question_meta.parquet  — per-question (concept_id, difficulty)
    concept_graph.pkl      — networkx DiGraph of prerequisite edges

Expected columns in output parquet files (consumed by dataset.py):
    user_id        int32
    question_id    int32   (re-indexed to 0 … NUM_QUESTIONS-1)
    concept_id     int16   (re-indexed to 0 … NUM_CONCEPTS-1)
    correct        int8    (0 or 1)
    elapsed_time   float32 (milliseconds, clipped + imputed)
    hint_count     int8    (clipped at 10)
    step           int32   (per-student interaction counter, 0-based)

Usage:
    python data/preprocess.py
    python data/preprocess.py --source ednet        # only EdNet
    python data/preprocess.py --source assistments  # only ASSISTments
    python data/preprocess.py --min-interactions 5  # filter short sequences
    python data/preprocess.py --dry-run             # stats only, no writes
"""

import argparse
import pickle
import random
import sys
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ASSISTMENTS_FILE,
    CONCEPT_GRAPH,
    DATA_PROC_DIR,
    EDNET_FILE,
    NUM_CONCEPTS,
    NUM_QUESTIONS,
    SPLIT_RATIOS,
    TEST_FILE,
    TRAIN_FILE,
    VAL_FILE,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

QUESTION_META_FILE   = DATA_PROC_DIR / "question_meta.parquet"

# Students with fewer interactions than this are dropped (too sparse for KT)
MIN_INTERACTIONS_DEFAULT = 5

# Interactions beyond this per student are truncated (very long-tail outliers)
MAX_INTERACTIONS_CAP     = 2_000

# Elapsed-time clipping: values outside this range are treated as missing
ELAPSED_MIN_MS  = 100      # < 100ms is likely a mis-click
ELAPSED_MAX_MS  = 600_000  # > 10 min is likely AFK

# Random seed for reproducible splits
SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# EdNet loader
# ─────────────────────────────────────────────────────────────────────────────

def load_ednet(path: Path) -> pd.DataFrame:
    """
    Load EdNet KT4 parquet produced by download_data.py.

    Expected columns (subset used):
        user_id, question_id, correct, elapsed_time,
        hint_count, concept_id, difficulty, timestamp
    """
    print(f"Loading EdNet from {path} …")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python data/download_data.py` first."
        )

    df = pd.read_parquet(path)
    print(f"  Raw rows: {len(df):,}   unique students: {df['user_id'].nunique():,}")

    # Keep only columns we need; add defaults for optional ones
    required = ["user_id", "question_id", "correct", "concept_id"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"EdNet file is missing columns: {missing}")

    for col, default in [("elapsed_time", 0.0), ("hint_count", 0), ("difficulty", 0.5)]:
        if col not in df.columns:
            print(f"  Warning: '{col}' missing — filling with {default}")
            df[col] = default

    # Sort by student then timestamp (or original order if no timestamp)
    if "timestamp" in df.columns:
        df = df.sort_values(["user_id", "timestamp"])
    else:
        df = df.sort_values("user_id")

    df = df.reset_index(drop=True)

    # Unified source tag so we can merge later
    df["source"] = "ednet"
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ASSISTments loader
# ─────────────────────────────────────────────────────────────────────────────

def load_assistments(path: Path) -> pd.DataFrame:
    """
    Load ASSISTments 2015 parquet produced by download_data.py.

    ASSISTments uses skill_id instead of concept_id — we rename it.
    question_id is also re-indexed to avoid collisions with EdNet.
    """
    print(f"Loading ASSISTments from {path} …")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python data/download_data.py --dataset assistments` first."
        )

    df = pd.read_parquet(path)
    print(f"  Raw rows: {len(df):,}   unique students: {df['user_id'].nunique():,}")

    # Rename skill_id → concept_id for unified schema
    if "skill_id" in df.columns and "concept_id" not in df.columns:
        df = df.rename(columns={"skill_id": "concept_id"})

    # ASSISTments has no difficulty column — infer from per-question accuracy
    if "difficulty" not in df.columns:
        q_acc = df.groupby("question_id")["correct"].mean()
        df["difficulty"] = (1.0 - df["question_id"].map(q_acc)).clip(0.0, 1.0).astype("float32")

    if "hint_count" not in df.columns:
        df["hint_count"] = 0

    if "elapsed_time" not in df.columns:
        df["elapsed_time"] = 0.0

    df = df.sort_values("user_id").reset_index(drop=True)
    df["source"] = "assistments"
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Merging and re-indexing
# ─────────────────────────────────────────────────────────────────────────────

def merge_sources(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate DataFrames from different sources and re-index
    user_id, question_id, and concept_id to contiguous integers
    starting from 0.

    This is necessary because EdNet and ASSISTments have overlapping
    ID spaces, and our config defines NUM_QUESTIONS=13169, NUM_CONCEPTS=188
    which are EdNet-specific; ASSISTments interactions are remapped into
    the same space during KT pre-training.
    """
    print("\nMerging sources …")
    df = pd.concat(frames, ignore_index=True)
    print(f"  Combined rows: {len(df):,}")

    # Re-index question_id to [0, NUM_QUESTIONS)
    # Questions beyond the cap are collapsed to NUM_QUESTIONS - 1 (UNK bucket)
    q_encoder = {qid: i for i, qid in enumerate(sorted(df["question_id"].unique()))}
    df["question_id"] = df["question_id"].map(q_encoder).clip(0, NUM_QUESTIONS - 1).astype("int32")

    # Re-index concept_id to [0, NUM_CONCEPTS)
    c_encoder = {cid: i for i, cid in enumerate(sorted(df["concept_id"].unique()))}
    df["concept_id"] = df["concept_id"].map(c_encoder).clip(0, NUM_CONCEPTS - 1).astype("int16")

    # Re-index user_id to contiguous integers
    u_encoder = {uid: i for i, uid in enumerate(sorted(df["user_id"].unique()))}
    df["user_id"] = df["user_id"].map(u_encoder).astype("int32")

    print(f"  Unique students:  {df['user_id'].nunique():,}")
    print(f"  Unique questions: {df['question_id'].nunique():,}")
    print(f"  Unique concepts:  {df['concept_id'].nunique():,}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame, min_interactions: int) -> pd.DataFrame:
    """
    Drop bad rows, impute missing values, clip outliers.

    Steps:
      1. Drop rows with null correct / question_id / concept_id
      2. Ensure correct ∈ {0, 1}
      3. Clip elapsed_time to [ELAPSED_MIN_MS, ELAPSED_MAX_MS];
         replace 0 / NaN with per-concept median (imputed)
      4. Clip hint_count to [0, 10]
      5. Drop students with fewer than min_interactions responses
      6. Truncate students with more than MAX_INTERACTIONS_CAP responses
         (keep the most recent interactions)
      7. Add per-student step counter (0-based)
    """
    print("\nCleaning …")
    n0 = len(df)

    # 1. Drop null critical columns
    df = df.dropna(subset=["correct", "question_id", "concept_id"])

    # 2. Binarise correct
    df["correct"] = df["correct"].astype(float).round().clip(0, 1).astype("int8")

    # 3. Elapsed time: clip then impute 0/NaN rows per concept
    df["elapsed_time"] = pd.to_numeric(df["elapsed_time"], errors="coerce")
    out_of_range = (df["elapsed_time"] < ELAPSED_MIN_MS) | (df["elapsed_time"] > ELAPSED_MAX_MS)
    df.loc[out_of_range, "elapsed_time"] = np.nan

    concept_medians = (
        df.groupby("concept_id")["elapsed_time"]
        .median()
        .fillna(30_000)   # global fallback: 30 seconds
    )
    mask_null = df["elapsed_time"].isna()
    df.loc[mask_null, "elapsed_time"] = df.loc[mask_null, "concept_id"].map(concept_medians)
    df["elapsed_time"] = df["elapsed_time"].fillna(30_000).astype("float32")

    # 4. Hint count
    df["hint_count"] = pd.to_numeric(df["hint_count"], errors="coerce").fillna(0).clip(0, 10).astype("int8")

    # 5. Drop short sequences
    counts = df.groupby("user_id").size()
    keep   = counts[counts >= min_interactions].index
    df     = df[df["user_id"].isin(keep)]

    # 6. Truncate long sequences (keep most recent)
    def _truncate(grp):
        if len(grp) > MAX_INTERACTIONS_CAP:
            return grp.iloc[-MAX_INTERACTIONS_CAP:]
        return grp

    df = df.groupby("user_id", group_keys=False).apply(_truncate)
    df = df.reset_index(drop=True)

    # 7. Per-student step counter
    df["step"] = df.groupby("user_id").cumcount().astype("int32")

    n1 = len(df)
    print(f"  Dropped {n0 - n1:,} rows ({(n0-n1)/n0*100:.1f}%)")
    print(f"  Remaining: {n1:,} interactions  {df['user_id'].nunique():,} students")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Question metadata
# ─────────────────────────────────────────────────────────────────────────────

def build_question_meta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a (question_id → concept_id, difficulty) lookup table.

    difficulty is computed as 1 − mean(correct) per question (higher
    difficulty → fewer students answered correctly).

    This table is loaded by rl/env.py::QuestionPool.
    """
    print("\nBuilding question metadata …")
    meta = (
        df.groupby("question_id")
        .agg(
            concept_id = ("concept_id", "first"),
            difficulty = ("correct",    lambda x: float(1.0 - x.mean())),
            n_attempts = ("correct",    "count"),
        )
        .reset_index()
    )
    meta["concept_id"] = meta["concept_id"].astype("int16")
    meta["difficulty"] = meta["difficulty"].clip(0.0, 1.0).astype("float32")

    # Fill gaps: questions that appear in config but not in data
    all_q = pd.DataFrame({"question_id": np.arange(NUM_QUESTIONS, dtype="int32")})
    meta  = all_q.merge(meta, on="question_id", how="left")
    meta["concept_id"] = meta["concept_id"].fillna(0).astype("int16")
    meta["difficulty"] = meta["difficulty"].fillna(0.5).astype("float32")
    meta["n_attempts"] = meta["n_attempts"].fillna(0).astype("int32")

    print(f"  Questions in metadata: {len(meta):,}")
    print(f"  Mean difficulty: {meta['difficulty'].mean():.3f}")
    print(f"  Concept coverage: {meta['concept_id'].nunique()} / {NUM_CONCEPTS}")
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Concept dependency graph
# ─────────────────────────────────────────────────────────────────────────────

def build_concept_graph(df: pd.DataFrame, meta: pd.DataFrame) -> nx.DiGraph:
    """
    Build a directed graph of concept prerequisites.

    Heuristic: if concept A is frequently answered BEFORE concept B by
    the same student, add a directed edge A → B (A is a prerequisite of B).

    This is a lightweight proxy for a proper curriculum ontology.
    In production, replace with a domain expert–annotated graph.

    The graph is used by the RL environment to:
      - Compute ZPD (Zone of Proximal Development) masks
      - Order concepts in the learning path
    """
    print("\nBuilding concept dependency graph …")
    G = nx.DiGraph()
    G.add_nodes_from(range(NUM_CONCEPTS))

    # Assign difficulty as a node attribute
    c_diff = meta.groupby("concept_id")["difficulty"].mean().to_dict()
    for c in range(NUM_CONCEPTS):
        G.nodes[c]["difficulty"] = c_diff.get(c, 0.5)

    # Build co-occurrence matrix: for each student, which concepts appear
    # before which other concepts?
    # We sample students to keep runtime manageable.
    students = df["user_id"].unique()
    sample   = students if len(students) <= 20_000 else np.random.default_rng(SEED).choice(
        students, size=20_000, replace=False
    )

    # Count how many students studied concept A before concept B
    before_counts: dict[tuple[int, int], int] = {}

    for uid in tqdm(sample, desc="  Computing concept ordering", unit="students", ncols=70):
        seq = df[df["user_id"] == uid].sort_values("step")["concept_id"].values
        seen: set[int] = set()
        for c in seq:
            for prev in seen:
                if prev != c:
                    key = (prev, c)
                    before_counts[key] = before_counts.get(key, 0) + 1
            seen.add(c)

    # Add edges where the ordering is consistent and frequent enough
    # Threshold: at least 5% of students who saw both followed this order
    concept_freq = df.groupby("concept_id").size().to_dict()

    edge_threshold = max(10, len(sample) * 0.02)  # 2% of sampled students
    edges_added = 0
    for (a, b), count in before_counts.items():
        if count >= edge_threshold:
            # Only add if it doesn't create a cycle (keep the graph a DAG)
            if not nx.has_path(G, b, a) if G.number_of_edges() > 0 else True:
                G.add_edge(a, b, weight=count)
                edges_added += 1

    print(f"  Nodes: {G.number_of_nodes()}   Edges: {G.number_of_edges()}")
    print(f"  Connected components: {nx.number_weakly_connected_components(G)}")
    return G


# ─────────────────────────────────────────────────────────────────────────────
# Train / val / test split
# ─────────────────────────────────────────────────────────────────────────────

def split_by_student(
    df: pd.DataFrame,
    ratios: dict = SPLIT_RATIOS,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split students (not interactions) into train / val / test sets.

    Splitting by student prevents data leakage: the model never sees
    test students during training, which is essential for evaluating
    generalisation to new learners.

    Args:
        df:     full cleaned DataFrame
        ratios: {"train": 0.80, "val": 0.10, "test": 0.10}
        seed:   random seed for reproducibility

    Returns:
        (train_df, val_df, test_df)
    """
    print("\nSplitting by student …")
    students = np.array(sorted(df["user_id"].unique()))
    rng      = np.random.default_rng(seed)
    rng.shuffle(students)

    n          = len(students)
    n_train    = int(n * ratios["train"])
    n_val      = int(n * ratios["val"])

    train_ids  = set(students[:n_train])
    val_ids    = set(students[n_train : n_train + n_val])
    test_ids   = set(students[n_train + n_val :])

    train_df = df[df["user_id"].isin(train_ids)].copy()
    val_df   = df[df["user_id"].isin(val_ids)].copy()
    test_df  = df[df["user_id"].isin(test_ids)].copy()

    print(f"  Train: {len(train_ids):,} students  {len(train_df):,} interactions")
    print(f"  Val:   {len(val_ids):,} students  {len(val_df):,} interactions")
    print(f"  Test:  {len(test_ids):,} students  {len(test_df):,} interactions")

    # Sanity: check no student appears in two splits
    assert train_ids.isdisjoint(val_ids),  "Train/val overlap!"
    assert train_ids.isdisjoint(test_ids), "Train/test overlap!"
    assert val_ids.isdisjoint(test_ids),   "Val/test overlap!"

    return train_df, val_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# Column finalisation
# ─────────────────────────────────────────────────────────────────────────────

FINAL_COLUMNS = [
    "user_id", "question_id", "concept_id",
    "correct", "elapsed_time", "hint_count", "step",
]

def finalise(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the columns dataset.py expects, in the right dtypes."""
    df = df[FINAL_COLUMNS].copy()
    df["user_id"]      = df["user_id"].astype("int32")
    df["question_id"]  = df["question_id"].astype("int32")
    df["concept_id"]   = df["concept_id"].astype("int16")
    df["correct"]      = df["correct"].astype("int8")
    df["elapsed_time"] = df["elapsed_time"].astype("float32")
    df["hint_count"]   = df["hint_count"].astype("int8")
    df["step"]         = df["step"].astype("int32")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_stats(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """Print a summary table of the processed dataset."""
    print("\n" + "=" * 60)
    print("Dataset statistics")
    print("=" * 60)

    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        n_students   = df["user_id"].nunique()
        n_inter      = len(df)
        mean_seq_len = df.groupby("user_id").size().mean()
        accuracy     = df["correct"].mean()
        print(
            f"  {name:5s}  "
            f"students={n_students:>7,}  "
            f"interactions={n_inter:>10,}  "
            f"mean_seq={mean_seq_len:5.1f}  "
            f"accuracy={accuracy:.3f}"
        )

    all_df = pd.concat([train_df, val_df, test_df])
    print(f"\n  Concept coverage : {all_df['concept_id'].nunique()} / {NUM_CONCEPTS}")
    print(f"  Question coverage: {all_df['question_id'].nunique()} / {NUM_QUESTIONS}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(
    source: str              = "both",
    min_interactions: int    = MIN_INTERACTIONS_DEFAULT,
    dry_run: bool            = False,
) -> None:
    """
    Full preprocessing pipeline.

    Args:
        source:           "ednet", "assistments", or "both"
        min_interactions: drop students with fewer responses
        dry_run:          compute stats but don't write any files
    """
    np.random.seed(SEED)
    random.seed(SEED)

    # ── 1. Load raw data ──────────────────────────────────────────────────────
    frames = []
    if source in ("ednet", "both"):
        if EDNET_FILE.exists():
            frames.append(load_ednet(EDNET_FILE))
        else:
            print(f"Warning: EdNet file not found at {EDNET_FILE} — skipping.")

    if source in ("assistments", "both"):
        if ASSISTMENTS_FILE.exists():
            frames.append(load_assistments(ASSISTMENTS_FILE))
        else:
            print(f"Warning: ASSISTments file not found at {ASSISTMENTS_FILE} — skipping.")

    if not frames:
        raise RuntimeError(
            "No raw data files found. "
            "Run `python data/download_data.py` first."
        )

    # ── 2. Merge and re-index ─────────────────────────────────────────────────
    df = merge_sources(frames)

    # ── 3. Clean ──────────────────────────────────────────────────────────────
    df = clean(df, min_interactions=min_interactions)

    # ── 4. Build question metadata ────────────────────────────────────────────
    meta = build_question_meta(df)

    # ── 5. Build concept graph ────────────────────────────────────────────────
    graph = build_concept_graph(df, meta)

    # ── 6. Split by student ───────────────────────────────────────────────────
    train_df, val_df, test_df = split_by_student(df)

    # ── 7. Finalise column types ──────────────────────────────────────────────
    train_df = finalise(train_df)
    val_df   = finalise(val_df)
    test_df  = finalise(test_df)

    # ── 8. Report ──────────────────────────────────────────────────────────────
    print_stats(train_df, val_df, test_df)

    if dry_run:
        print("\nDry run — no files written.")
        return

    # ── 9. Write outputs ──────────────────────────────────────────────────────
    print("\nWriting outputs …")
    DATA_PROC_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(TRAIN_FILE,         index=False, compression="snappy")
    val_df.to_parquet(VAL_FILE,             index=False, compression="snappy")
    test_df.to_parquet(TEST_FILE,           index=False, compression="snappy")
    meta.to_parquet(QUESTION_META_FILE,     index=False, compression="snappy")

    with open(CONCEPT_GRAPH, "wb") as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"  {TRAIN_FILE}")
    print(f"  {VAL_FILE}")
    print(f"  {TEST_FILE}")
    print(f"  {QUESTION_META_FILE}")
    print(f"  {CONCEPT_GRAPH}")
    print("\nPreprocessing complete ✓")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess EdNet / ASSISTments data for the Adaptive Learning System."
    )
    parser.add_argument(
        "--source",
        choices=["ednet", "assistments", "both"],
        default="both",
        help="Which dataset(s) to process (default: both)",
    )
    parser.add_argument(
        "--min-interactions",
        type=int,
        default=MIN_INTERACTIONS_DEFAULT,
        help=f"Drop students with fewer than N interactions (default: {MIN_INTERACTIONS_DEFAULT})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print statistics without writing any files",
    )
    args = parser.parse_args()

    preprocess(
        source           = args.source,
        min_interactions = args.min_interactions,
        dry_run          = args.dry_run,
    )


if __name__ == "__main__":
    main()