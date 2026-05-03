"""
data/dataset.py
PyTorch Dataset and DataLoader for knowledge tracing sequences.

Each sample is one student's full interaction history, padded/truncated
to MAX_SEQ_LEN. The DKVMN model consumes these batches during training.

Shapes produced per batch:
  question_ids  : (B, T)        int32  — question index at each step
  concept_ids   : (B, T)        int16  — concept index at each step
  corrects      : (B, T)        float  — 0 or 1
  elapsed_times : (B, T)        float  — normalised ms
  hint_counts   : (B, T)        float  — normalised hint count
  mask          : (B, T)        bool   — True = real step, False = padding
  seq_lens      : (B,)          int    — actual length of each sequence

Usage:
    from data.dataset import make_dataloaders
    train_dl, val_dl, test_dl = make_dataloaders()
    for batch in train_dl:
        ...
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TRAIN_FILE, VAL_FILE, TEST_FILE,
    MAX_SEQ_LEN, PAD_TOKEN, KT,
)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class KTDataset(Dataset):
    """
    One item = one student's interaction sequence.
    Sequences longer than MAX_SEQ_LEN are truncated (most recent kept).
    Sequences shorter are right-padded with PAD_TOKEN / zeros.
    """

    def __init__(self, parquet_path: Path, max_seq_len: int = MAX_SEQ_LEN):
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"{parquet_path} not found.\n"
                "Run `python data/preprocess.py` first."
            )
        df = pd.read_parquet(parquet_path)
        self.max_seq_len = max_seq_len
        self.sequences   = self._build_sequences(df)

    # ------------------------------------------------------------------
    def _build_sequences(self, df: pd.DataFrame) -> list[dict]:
        """Group by student, sort by step, return list of sequence dicts."""
        sequences = []
        for uid, grp in df.groupby("user_id"):
            grp = grp.sort_values("step") if "step" in grp.columns else grp

            # Truncate to most recent MAX_SEQ_LEN interactions
            if len(grp) > self.max_seq_len:
                grp = grp.iloc[-self.max_seq_len:]

            seq_len = len(grp)

            # Core arrays
            q_ids   = grp["question_id"].values.astype(np.int32)
            c_ids   = grp["concept_id"].values.astype(np.int16)
            corr    = grp["correct"].values.astype(np.float32)

            # Optional features (default to zeros if missing)
            elapsed = grp["elapsed_time"].values.astype(np.float32) \
                      if "elapsed_time" in grp.columns else np.zeros(seq_len, np.float32)
            hints   = grp["hint_count"].values.astype(np.float32) \
                      if "hint_count" in grp.columns else np.zeros(seq_len, np.float32)

            sequences.append({
                "user_id":      uid,
                "seq_len":      seq_len,
                "question_ids": q_ids,
                "concept_ids":  c_ids,
                "corrects":     corr,
                "elapsed":      elapsed,
                "hints":        hints,
            })
        return sequences

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s   = self.sequences[idx]
        T   = self.max_seq_len
        L   = s["seq_len"]

        # Pad all arrays to MAX_SEQ_LEN
        def pad(arr: np.ndarray, pad_val: float = 0.0) -> np.ndarray:
            out = np.full(T, pad_val, dtype=arr.dtype)
            out[:L] = arr
            return out

        # Normalise elapsed time: log-scale, then z-score to [0,1] range
        elapsed = s["elapsed"].copy()
        elapsed = np.log1p(elapsed)                     # log(1 + ms)
        elapsed = elapsed / np.log1p(600_000)           # normalise to ~[0,1]

        # Normalise hint count to [0, 1]
        hints   = np.clip(s["hints"] / 10.0, 0.0, 1.0)

        # Build boolean mask: True = real, False = pad
        mask = np.zeros(T, dtype=bool)
        mask[:L] = True

        return {
            "question_ids":  torch.from_numpy(pad(s["question_ids"], 0)),
            "concept_ids":   torch.from_numpy(pad(s["concept_ids"],  0).astype(np.int32)),
            "corrects":      torch.from_numpy(pad(s["corrects"],     0.0)),
            "elapsed":       torch.from_numpy(pad(elapsed,           0.0)),
            "hints":         torch.from_numpy(pad(hints,             0.0)),
            "mask":          torch.from_numpy(mask),
            "seq_len":       torch.tensor(L, dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Collate function
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """
    Stack individual samples into a batch.
    All sequences are already padded to MAX_SEQ_LEN in __getitem__,
    so simple torch.stack() suffices.
    """
    return {
        key: torch.stack([item[key] for item in batch])
        for key in batch[0]
    }


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def make_dataloaders(
    batch_size:  int  = KT["batch_size"],
    num_workers: int  = 2,
    pin_memory:  bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders from preprocessed parquet files.

    Args:
        batch_size:  samples per batch (default from config.KT)
        num_workers: parallel workers for data loading
        pin_memory:  speeds up CPU→GPU transfers when using CUDA

    Returns:
        (train_loader, val_loader, test_loader)
    """
    pin = pin_memory and torch.cuda.is_available()

    train_ds = KTDataset(TRAIN_FILE)
    val_ds   = KTDataset(VAL_FILE)
    test_ds  = KTDataset(TEST_FILE)

    train_dl = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = True,           # shuffle students each epoch
        num_workers = num_workers,
        pin_memory  = pin,
        collate_fn  = collate_fn,
        drop_last   = True,           # avoid tiny last batch destabilising BN
    )
    val_dl = DataLoader(
        val_ds,
        batch_size  = batch_size * 2, # no grad → can use larger batch
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = pin,
        collate_fn  = collate_fn,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size  = batch_size * 2,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = pin,
        collate_fn  = collate_fn,
    )

    print(f"DataLoaders ready:")
    print(f"  Train: {len(train_ds):,} students  {len(train_dl):,} batches")
    print(f"  Val:   {len(val_ds):,} students  {len(val_dl):,} batches")
    print(f"  Test:  {len(test_ds):,} students  {len(test_dl):,} batches")

    return train_dl, val_dl, test_dl


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== KTDataset smoke test ===\n")
    train_dl, val_dl, test_dl = make_dataloaders(batch_size=4, num_workers=0)

    batch = next(iter(train_dl))
    print("Batch keys:", list(batch.keys()))
    for k, v in batch.items():
        print(f"  {k:15s}  shape={str(v.shape):<20} dtype={v.dtype}")

    print("\nFirst sample seq_len:", batch["seq_len"][0].item())
    print("Mask sum (real steps):", batch["mask"][0].sum().item())
    print("Mean accuracy in batch:", batch["corrects"][batch["mask"]].mean().item())
    print("\nSmoke test passed ✓")