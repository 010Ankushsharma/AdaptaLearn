"""
models/train_kt.py
Training loop for the DKVMN knowledge tracing model.

What this script does:
  1. Load train / val DataLoaders from preprocessed parquet files
  2. Build DKVMN from config
  3. Train with BCE loss, masking padding positions
  4. Evaluate AUC on validation set after every epoch
  5. Early stopping on val AUC (patience from config)
  6. Save best checkpoint to config.KT["checkpoint_path"]
  7. Log all metrics to MLflow

Usage:
    python models/train_kt.py
    python models/train_kt.py --epochs 30 --lr 5e-4
    python models/train_kt.py --resume   # continue from last checkpoint
"""

import argparse
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import KT, MLFLOW, NUM_CONCEPTS, NUM_QUESTIONS
from data.dataset import make_dataloaders
from models.dkvmn import DKVMN, build_model, count_params


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def masked_bce_loss(
    logits: torch.Tensor,   # (B, T)
    targets: torch.Tensor,  # (B, T)  float 0/1
    mask: torch.Tensor,     # (B, T)  bool — True = real step
) -> torch.Tensor:
    """
    Binary cross-entropy only over real (non-padded) positions.

    We predict P(correct at step t+1) using the state after step t,
    so we shift: predict logits[:, :-1] against targets[:, 1:].
    This is the standard next-step prediction setup for KT.
    """
    # Shift: predict next step
    logits_  = logits[:, :-1]    # (B, T-1)
    targets_ = targets[:, 1:]    # (B, T-1)
    mask_    = mask[:, 1:]       # (B, T-1)  — mask for target positions

    loss = nn.functional.binary_cross_entropy_with_logits(
        logits_, targets_, reduction="none"
    )                             # (B, T-1)

    # Zero out padding positions, average over real positions
    loss = (loss * mask_.float()).sum() / mask_.float().sum().clamp(min=1)
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# One epoch
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model:      DKVMN,
    loader,
    optimizer:  torch.optim.Optimizer | None,
    device:     torch.device,
    is_train:   bool,
) -> tuple[float, float]:
    """
    Run one full pass over the dataloader.

    Returns:
        (mean_loss, auc) for this epoch
    """
    model.train(is_train)
    context = torch.enable_grad() if is_train else torch.no_grad()

    total_loss   = 0.0
    all_probs    = []
    all_targets  = []
    n_batches    = 0

    with context:
        for batch in loader:
            q_ids   = batch["question_ids"].to(device)   # (B, T)
            c_ids   = batch["concept_ids"].to(device)
            corr    = batch["corrects"].to(device)
            elapsed = batch["elapsed"].to(device)
            hints   = batch["hints"].to(device)
            mask    = batch["mask"].to(device)

            logits, _ = model(q_ids, c_ids, corr, elapsed, hints, mask)

            loss = masked_bce_loss(logits, corr, mask)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Collect predictions for AUC (shift same as loss)
            probs_  = torch.sigmoid(logits[:, :-1])   # (B, T-1)
            targets_= corr[:, 1:]                      # (B, T-1)
            mask_   = mask[:, 1:]                      # (B, T-1)

            all_probs.append(probs_[mask_].detach().cpu().numpy())
            all_targets.append(targets_[mask_].detach().cpu().numpy())

            total_loss += loss.item()
            n_batches  += 1

    mean_loss = total_loss / max(n_batches, 1)

    all_probs   = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)

    # AUC requires both classes present
    if len(np.unique(all_targets)) < 2:
        auc = 0.5
    else:
        auc = roc_auc_score(all_targets, all_probs)

    return mean_loss, auc


# ─────────────────────────────────────────────────────────────────────────────
# Early stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """Stop training when val AUC stops improving for `patience` epochs."""

    def __init__(self, patience: int, checkpoint_path: Path, min_delta: float = 1e-4):
        self.patience        = patience
        self.checkpoint_path = checkpoint_path
        self.min_delta       = min_delta
        self.best_auc        = -float("inf")
        self.counter         = 0
        self.should_stop     = False

    def step(self, val_auc: float, model: DKVMN) -> bool:
        """
        Call after each epoch with the validation AUC.
        Saves checkpoint if improved. Returns True if training should stop.
        """
        if val_auc > self.best_auc + self.min_delta:
            self.best_auc = val_auc
            self.counter  = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "val_auc":     val_auc,
                },
                self.checkpoint_path,
            )
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(
    num_epochs:    int   = KT["num_epochs"],
    learning_rate: float = KT["learning_rate"],
    batch_size:    int   = KT["batch_size"],
    patience:      int   = KT["patience"],
    resume:        bool  = False,
) -> DKVMN:
    """
    Full training loop. Returns the best model loaded from checkpoint.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Training DKVMN Knowledge Tracing ===")
    print(f"  Device:       {device}")
    print(f"  Epochs:       {num_epochs}")
    print(f"  LR:           {learning_rate}")
    print(f"  Batch size:   {batch_size}")
    print(f"  Patience:     {patience}")

    # ── Data ──────────────────────────────────────────────────────────
    train_dl, val_dl, _ = make_dataloaders(batch_size=batch_size, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────
    model = build_model(str(device))
    print(f"  Parameters:   {count_params(model):,}")

    start_epoch = 0
    if resume and KT["checkpoint_path"].exists():
        ckpt = torch.load(KT["checkpoint_path"], map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"  Resumed from checkpoint (best AUC: {ckpt['val_auc']:.4f})")

    # ── Optimiser & scheduler ─────────────────────────────────────────
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True
    )

    # ── Early stopping ─────────────────────────────────────────────────
    stopper = EarlyStopping(
        patience        = patience,
        checkpoint_path = KT["checkpoint_path"],
    )

    # ── MLflow run ────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW["tracking_uri"])
    mlflow.set_experiment(MLFLOW["experiment_name"])

    with mlflow.start_run(run_name="dkvmn_training"):
        mlflow.log_params({
            "model":         "DKVMN",
            "num_epochs":    num_epochs,
            "learning_rate": learning_rate,
            "batch_size":    batch_size,
            "patience":      patience,
            "key_dim":       KT["key_dim"],
            "value_dim":     KT["value_dim"],
            "dropout":       KT["dropout"],
            "num_concepts":  NUM_CONCEPTS,
            "num_questions": NUM_QUESTIONS,
            "n_params":      count_params(model),
        })

        # ── Epoch loop ────────────────────────────────────────────────
        print(f"\n{'Epoch':>6}  {'Train Loss':>11}  {'Train AUC':>10}  "
              f"{'Val Loss':>9}  {'Val AUC':>8}  {'Time':>6}")
        print("─" * 62)

        for epoch in range(start_epoch, num_epochs):
            t0 = time.time()

            train_loss, train_auc = run_epoch(
                model, train_dl, optimizer, device, is_train=True
            )
            val_loss, val_auc = run_epoch(
                model, val_dl, optimizer=None, device=device, is_train=False
            )

            elapsed = time.time() - t0
            scheduler.step(val_auc)

            # Log to MLflow
            mlflow.log_metrics({
                "train_loss": round(train_loss, 5),
                "train_auc":  round(train_auc,  5),
                "val_loss":   round(val_loss,   5),
                "val_auc":    round(val_auc,    5),
                "lr":         optimizer.param_groups[0]["lr"],
            }, step=epoch)

            # Console output
            marker = " ✓" if val_auc > stopper.best_auc else ""
            print(
                f"{epoch+1:>6}  {train_loss:>11.5f}  {train_auc:>10.4f}  "
                f"{val_loss:>9.5f}  {val_auc:>8.4f}  {elapsed:>5.1f}s{marker}"
            )

            # Early stopping check
            if stopper.step(val_auc, model):
                print(f"\n  Early stopping at epoch {epoch+1} "
                      f"(best val AUC: {stopper.best_auc:.4f})")
                break

        print("─" * 62)
        print(f"\n  Best val AUC: {stopper.best_auc:.4f}")
        print(f"  Checkpoint:   {KT['checkpoint_path']}")
        mlflow.log_metric("best_val_auc", stopper.best_auc)

    # ── Load and return best model ─────────────────────────────────────
    if KT["checkpoint_path"].exists():
        ckpt = torch.load(KT["checkpoint_path"], map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print("\n  Best model loaded from checkpoint.")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Test-set evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_test(model: DKVMN, batch_size: int = KT["batch_size"]) -> float:
    """Run the best model on the held-out test set and report AUC."""
    device = next(model.parameters()).device
    _, _, test_dl = make_dataloaders(batch_size=batch_size, num_workers=0)
    test_loss, test_auc = run_epoch(
        model, test_dl, optimizer=None, device=device, is_train=False
    )
    print(f"\n  Test set  →  loss: {test_loss:.5f}   AUC: {test_auc:.4f}")
    return test_auc


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train DKVMN knowledge tracing")
    parser.add_argument("--epochs",  type=int,   default=KT["num_epochs"])
    parser.add_argument("--lr",      type=float, default=KT["learning_rate"])
    parser.add_argument("--batch",   type=int,   default=KT["batch_size"])
    parser.add_argument("--patience",type=int,   default=KT["patience"])
    parser.add_argument("--resume",  action="store_true",
                        help="Resume training from existing checkpoint")
    parser.add_argument("--test-only", action="store_true",
                        help="Skip training — evaluate checkpoint on test set")
    args = parser.parse_args()

    if args.test_only:
        if not KT["checkpoint_path"].exists():
            print("No checkpoint found. Run training first.")
            sys.exit(1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = build_model(str(device))
        ckpt   = torch.load(KT["checkpoint_path"], map_location=device)
        model.load_state_dict(ckpt["model_state"])
        evaluate_test(model, batch_size=args.batch)
    else:
        model = train(
            num_epochs    = args.epochs,
            learning_rate = args.lr,
            batch_size    = args.batch,
            patience      = args.patience,
            resume        = args.resume,
        )
        evaluate_test(model, batch_size=args.batch)


if __name__ == "__main__":
    main()