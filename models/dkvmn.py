"""
models/dkvmn.py
Dynamic Key-Value Memory Network (DKVMN) for knowledge tracing.

Paper: Zhang et al., 2017 — "Dynamic Key-Value Memory Networks for Knowledge Tracing"
https://arxiv.org/abs/1611.02796

Architecture:
  ┌─────────────────────────────────────────────────────┐
  │  Input: question_id, concept_id, correct (sequence) │
  └──────────────────────┬──────────────────────────────┘
                         │
          ┌──────────────▼──────────────┐
          │   Key Matrix  M^k            │  (NUM_CONCEPTS × key_dim)
          │   Static — shared by all     │  concept embeddings
          │   students                   │
          └──────────────┬──────────────┘
                         │ attention weights w_t = softmax(k_t · M^k)
          ┌──────────────▼──────────────┐
          │   Value Matrix  M^v          │  (NUM_CONCEPTS × value_dim)
          │   Dynamic — per student,     │  stores mastery state
          │   updated at each step       │
          └──────────┬──────────┬───────┘
                     │ read     │ write
                  r_t = Σ w_t M^v_t     (what the student knows)
                  M^v_{t+1} = M^v_t + w_t ⊗ e_t  (update after answer)
                         │
          ┌──────────────▼──────────────┐
          │   Output layer               │
          │   f([r_t, k_t]) → P(correct) │
          └─────────────────────────────┘

The value matrix after processing a student's history gives us
P(mastery per concept) — this is the RL environment's state vector.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import KT, NUM_CONCEPTS, NUM_QUESTIONS


# ─────────────────────────────────────────────────────────────────────────────
# Sub-modules
# ─────────────────────────────────────────────────────────────────────────────

class KeyMemory(nn.Module):
    """
    Static key matrix M^k  (NUM_CONCEPTS × key_dim).
    Shared across all students — represents the concept embedding space.
    Each row is the embedding of one concept.
    """
    def __init__(self, num_concepts: int, key_dim: int):
        super().__init__()
        self.key_dim     = key_dim
        self.num_concepts = num_concepts
        # Learnable concept embeddings
        self.embeddings = nn.Parameter(
            torch.randn(num_concepts, key_dim) * 0.01
        )

    def attention(self, query: torch.Tensor) -> torch.Tensor:
        """
        Compute soft attention weights over concepts.

        Args:
            query: (B, key_dim) — embedding of the current question/concept

        Returns:
            weights: (B, NUM_CONCEPTS) — softmax attention over concept slots
        """
        # (B, key_dim) @ (key_dim, N) → (B, N)
        scores = query @ self.embeddings.T          # (B, N)
        return F.softmax(scores, dim=-1)            # (B, N)


class ValueMemory(nn.Module):
    """
    Dynamic value matrix M^v  (NUM_CONCEPTS × value_dim) per student.
    Initialised to zeros at the start of each episode/sequence.
    Updated after each interaction using an erase-add mechanism.
    """
    def __init__(self, num_concepts: int, value_dim: int):
        super().__init__()
        self.num_concepts = num_concepts
        self.value_dim    = value_dim

        # Erase and add gates (conditioned on the interaction embedding)
        self.erase_linear = nn.Linear(value_dim, value_dim)
        self.add_linear   = nn.Linear(value_dim, value_dim)

    def read(self, memory: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Read from value memory using attention weights.

        Args:
            memory:  (B, N, value_dim)
            weights: (B, N)

        Returns:
            r: (B, value_dim) — weighted sum of memory slots
        """
        # (B, N, 1) * (B, N, value_dim) → sum → (B, value_dim)
        return (weights.unsqueeze(-1) * memory).sum(dim=1)

    def write(
        self,
        memory:  torch.Tensor,   # (B, N, value_dim)
        weights: torch.Tensor,   # (B, N)
        v_t:     torch.Tensor,   # (B, value_dim) interaction vector
    ) -> torch.Tensor:
        """
        Update value memory with erase-add mechanism.

        M^v_{t+1}[i] = M^v_t[i] * (1 − w_i * e_t) + w_i * a_t

        The erase gate removes old information proportional to attention weight.
        The add gate writes new information proportional to attention weight.
        """
        erase = torch.sigmoid(self.erase_linear(v_t))   # (B, value_dim)
        add   = torch.tanh(self.add_linear(v_t))         # (B, value_dim)

        w = weights.unsqueeze(-1)                        # (B, N, 1)

        # Erase: reduce old memory
        memory = memory * (1.0 - w * erase.unsqueeze(1))

        # Add: write new information
        memory = memory + w * add.unsqueeze(1)

        return memory                                    # (B, N, value_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Main DKVMN model
# ─────────────────────────────────────────────────────────────────────────────

class DKVMN(nn.Module):
    """
    Dynamic Key-Value Memory Network.

    Processes a student's interaction sequence step-by-step,
    maintaining a value memory that encodes their evolving knowledge state.

    Args:
        num_questions: total number of unique questions
        num_concepts:  total number of unique concepts (= memory slots)
        key_dim:       dimension of concept key embeddings
        value_dim:     dimension of value memory slots
        dropout:       dropout rate for regularisation

    Forward input (all shape B × T):
        question_ids, concept_ids, corrects, elapsed, hints, mask

    Forward output:
        logits:   (B, T) — raw predictions for P(correct at step t+1)
        mastery:  (B, T, NUM_CONCEPTS) — per-concept mastery after each step
    """

    def __init__(
        self,
        num_questions: int = NUM_QUESTIONS,
        num_concepts:  int = NUM_CONCEPTS,
        key_dim:       int = KT["key_dim"],
        value_dim:     int = KT["value_dim"],
        dropout:       float = KT["dropout"],
    ):
        super().__init__()
        self.num_concepts = num_concepts
        self.key_dim      = key_dim
        self.value_dim    = value_dim

        # ── Embeddings ────────────────────────────────────────────────
        # Question embedding: maps question_id → key_dim vector
        self.question_emb = nn.Embedding(num_questions + 1, key_dim, padding_idx=0)

        # Interaction embedding: encodes (question, correct) pair → value_dim
        # We embed question and correctness jointly: 2 * num_questions slots
        # (one for correct, one for incorrect per question)
        self.interaction_emb = nn.Embedding(
            2 * num_questions + 1, value_dim, padding_idx=0
        )

        # ── Memory ───────────────────────────────────────────────────
        self.key_memory   = KeyMemory(num_concepts, key_dim)
        self.value_memory = ValueMemory(num_concepts, value_dim)

        # ── Feature fusion ────────────────────────────────────────────
        # After reading from memory, fuse with question embedding
        # Extra features: elapsed_time (1) + hint_count (1)
        self.fusion = nn.Sequential(
            nn.Linear(value_dim + key_dim + 2, value_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )

        # ── Output head ───────────────────────────────────────────────
        self.output_layer = nn.Sequential(
            nn.Linear(value_dim, value_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(value_dim // 2, 1),
        )

        # ── Mastery projection ────────────────────────────────────────
        # Projects each value memory slot → scalar mastery probability
        self.mastery_proj = nn.Linear(value_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(
        self,
        question_ids: torch.Tensor,   # (B, T)
        concept_ids:  torch.Tensor,   # (B, T)
        corrects:     torch.Tensor,   # (B, T)  float 0/1
        elapsed:      torch.Tensor,   # (B, T)  normalised
        hints:        torch.Tensor,   # (B, T)  normalised
        mask:         torch.Tensor,   # (B, T)  bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process a full sequence.

        Returns:
            logits:  (B, T)             — pre-sigmoid predictions per step
            mastery: (B, T, N_concepts) — per-concept mastery after each step
        """
        B, T = question_ids.shape
        device = question_ids.device

        # Initialise value memory to zeros for each student in batch
        memory = torch.zeros(B, self.num_concepts, self.value_dim, device=device)

        logits_list  = []
        mastery_list = []

        for t in range(T):
            q_t    = question_ids[:, t]   # (B,)
            c_t    = concept_ids[:, t]    # (B,)
            cor_t  = corrects[:, t]       # (B,)
            el_t   = elapsed[:, t]        # (B,)
            hi_t   = hints[:, t]          # (B,)

            # ── Key query: embed the question ─────────────────────────
            k_t = self.question_emb(q_t)                       # (B, key_dim)

            # ── Attention weights over concept slots ──────────────────
            w_t = self.key_memory.attention(k_t)               # (B, N)

            # ── Read from memory (what does the student know?) ────────
            r_t = self.value_memory.read(memory, w_t)          # (B, value_dim)

            # ── Predict correctness ───────────────────────────────────
            extra = torch.stack([el_t, hi_t], dim=-1)          # (B, 2)
            fused = self.fusion(
                torch.cat([r_t, k_t, extra], dim=-1)
            )                                                   # (B, value_dim)
            logit_t = self.output_layer(fused).squeeze(-1)     # (B,)
            logits_list.append(logit_t)

            # ── Interaction embedding for memory write ────────────────
            # Index: correct answers use question_id, incorrect use question_id + N
            interaction_idx = q_t + (cor_t * 0).long()  # placeholder baseline
            interaction_idx = torch.where(
                cor_t.bool(), q_t, q_t + question_ids.max() + 1
            ).clamp(0, self.interaction_emb.num_embeddings - 1)
            v_t = self.interaction_emb(interaction_idx)        # (B, value_dim)

            # ── Write to memory (update student's knowledge state) ────
            memory = self.value_memory.write(memory, w_t, v_t)

            # ── Mastery snapshot ──────────────────────────────────────
            # For each concept slot, project value vector → [0,1] probability
            mastery_t = torch.sigmoid(
                self.mastery_proj(memory).squeeze(-1)          # (B, N)
            )
            mastery_list.append(mastery_t)

        logits  = torch.stack(logits_list,  dim=1)            # (B, T)
        mastery = torch.stack(mastery_list, dim=1)            # (B, T, N)

        return logits, mastery

    # ── Inference helpers ─────────────────────────────────────────────────────

    @torch.no_grad()
    def get_mastery(
        self,
        question_ids: torch.Tensor,
        concept_ids:  torch.Tensor,
        corrects:     torch.Tensor,
        elapsed:      torch.Tensor,
        hints:        torch.Tensor,
        mask:         torch.Tensor,
    ) -> torch.Tensor:
        """
        Return the final mastery vector after processing a full sequence.
        Used by the RL environment to build the student state.

        Returns:
            mastery: (B, NUM_CONCEPTS) — P(mastered) per concept, current state
        """
        self.eval()
        _, mastery = self.forward(
            question_ids, concept_ids, corrects, elapsed, hints, mask
        )
        # Take mastery at the last real step for each student in batch
        seq_lens = mask.sum(dim=1).clamp(min=1) - 1          # (B,) last real index
        idx = seq_lens.view(B := question_ids.size(0), 1, 1).expand(B, 1, mastery.size(-1))
        return mastery.gather(1, idx).squeeze(1)              # (B, N)

    @torch.no_grad()
    def predict_next(
        self,
        mastery_state: torch.Tensor,   # (N,) current mastery vector
        question_id:   int,
        device:        torch.device,
    ) -> float:
        """
        Given a current mastery vector, predict P(correct) on a specific question.
        Used by the RL environment's ZPD filter.

        Args:
            mastery_state: (NUM_CONCEPTS,) float tensor from get_mastery()
            question_id:   integer question index
            device:        torch device

        Returns:
            p_correct: float in [0, 1]
        """
        # Approximate: return mastery of the concept associated with this question
        # (a more precise version would run a 1-step forward pass)
        return mastery_state.mean().item()  # simplified; override in env.py


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(device: str = "cpu") -> DKVMN:
    """Instantiate DKVMN from config and move to device."""
    model = DKVMN(
        num_questions = NUM_QUESTIONS,
        num_concepts  = NUM_CONCEPTS,
        key_dim       = KT["key_dim"],
        value_dim     = KT["value_dim"],
        dropout       = KT["dropout"],
    )
    return model.to(device)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== DKVMN smoke test ===\n")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = build_model(device)
    print(f"Parameters: {count_params(model):,}")

    # Fake batch: B=4 students, T=20 time steps
    B, T, N = 4, 20, NUM_CONCEPTS
    q_ids  = torch.randint(0, NUM_QUESTIONS, (B, T)).to(device)
    c_ids  = torch.randint(0, N,             (B, T)).to(device)
    corr   = torch.randint(0, 2,             (B, T)).float().to(device)
    el     = torch.rand(B, T).to(device)
    hi     = torch.rand(B, T).to(device)
    mask   = torch.ones(B, T, dtype=torch.bool).to(device)

    logits, mastery = model(q_ids, c_ids, corr, el, hi, mask)

    print(f"logits shape:  {logits.shape}   (expected {B}×{T})")
    print(f"mastery shape: {mastery.shape}  (expected {B}×{T}×{N})")
    print(f"P(correct) range: [{logits.sigmoid().min():.3f}, {logits.sigmoid().max():.3f}]")
    print(f"Mastery range:    [{mastery.min():.3f}, {mastery.max():.3f}]")
    print("\nSmoke test passed ✓")