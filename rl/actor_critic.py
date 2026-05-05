"""
rl/actor_critic.py
Actor-Critic network for the Adaptive Learning PPO agent.

Architecture:
                    Student state (NUM_CONCEPTS + 4,)
                            │
                    ┌───────▼────────┐
                    │ Shared Backbone │  256 → 128 → 64  (ReLU + LayerNorm)
                    └───────┬────────┘
                  ┌─────────┴──────────┐
         ┌────────▼───────┐   ┌────────▼────────┐
         │   Actor head   │   │   Critic head   │
         │  Linear(64,A)  │   │  Linear(64, 1)  │
         │  + ZPD mask    │   │                 │
         │  + Softmax     │   │   scalar V(s)   │
         └────────────────┘   └─────────────────┘
                 π(a|s)               V(s)

The ZPD mask is applied BEFORE softmax so masked actions get
probability exactly 0 (not just low probability). This is the
correct way to implement action masking in policy gradient methods.

Integration with Stable-Baselines3 MaskablePPO:
  sb3-contrib's MaskablePPO expects a custom policy class that
  inherits from MaskableActorCriticPolicy. We register our network
  as the mlp_extractor, keeping SB3's standard PPO update loop.

Usage:
    from rl.actor_critic import MaskedActorCriticPolicy, build_policy_kwargs

    model = MaskablePPO(
        MaskedActorCriticPolicy,
        env,
        policy_kwargs = build_policy_kwargs(),
        **PPO_KWARGS,
    )
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ENV, NUM_CONCEPTS, NUM_QUESTIONS, PPO


# ─────────────────────────────────────────────────────────────────────────────
# Shared backbone
# ─────────────────────────────────────────────────────────────────────────────

class SharedBackbone(nn.Module):
    """
    Shared MLP feature extractor used by both actor and critic.

    Input  : (B, state_dim)   — concatenated mastery + session meta
    Output : (B, latent_dim)  — shared latent representation

    Uses LayerNorm instead of BatchNorm because:
      - Episodes have variable length (batch statistics are unstable)
      - LayerNorm normalises per-sample, safe for single-sample inference
        in the Django backend

    Architecture: Linear → LayerNorm → ReLU, stacked N times.
    Skip connections between same-sized layers for gradient flow.
    """

    def __init__(
        self,
        input_dim:  int,
        net_arch:   List[int] = PPO["policy_kwargs"]["net_arch"],
        dropout:    float     = 0.1,
    ):
        super().__init__()
        self.input_dim  = input_dim
        self.output_dim = net_arch[-1]

        layers = []
        prev   = input_dim
        for i, hidden in enumerate(net_arch):
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.LayerNorm(hidden))
            layers.append(nn.ReLU())
            if i < len(net_arch) - 1:          # dropout except last layer
                layers.append(nn.Dropout(dropout))
            prev = hidden

        self.net = nn.Sequential(*layers)

        # Input normalisation — learns mean/std of each state dimension
        # Prevents mastery features (range 0–1) from being dominated by
        # large session metadata features
        self.input_norm = nn.LayerNorm(input_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x : (B, input_dim) float32 state vector

        Returns:
            h : (B, latent_dim) shared feature representation
        """
        x = self.input_norm(x)
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Actor head (masked policy)
# ─────────────────────────────────────────────────────────────────────────────

class MaskedActorHead(nn.Module):
    """
    Policy head π(a|s) with ZPD action masking.

    The mask is applied by setting logits of invalid actions to -1e9
    BEFORE softmax — this gives them exactly 0 probability while
    preserving the gradient flow through valid actions.

    Why not clamp after softmax?
      - Would still assign tiny probabilities to invalid actions
      - Gradient would flow into masked actions during backprop
      - Hard to ensure exactly 0 probability in fp32

    Output: log-probabilities (log_softmax) for numerical stability
            during PPO's policy ratio computation.
    """

    def __init__(self, latent_dim: int, action_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.linear = nn.Linear(latent_dim, action_dim)

        # Small initialisation for the actor head — prevents confident
        # predictions before the backbone has learned useful features
        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.zeros_(self.linear.bias)

    def forward(
        self,
        h:    Tensor,                        # (B, latent_dim)
        mask: Optional[Tensor] = None,       # (B, action_dim) bool or None
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            h    : shared feature vector from backbone
            mask : True = valid action, False = masked (ZPD-filtered)
                   If None, all actions are valid

        Returns:
            log_probs : (B, action_dim)  log π(a|s)  — for loss computation
            probs     : (B, action_dim)  π(a|s)      — for sampling / entropy
        """
        logits = self.linear(h)                              # (B, A)

        if mask is not None:
            # Apply mask: set invalid action logits to large negative
            logits = logits.masked_fill(~mask, -1e9)

        log_probs = F.log_softmax(logits, dim=-1)            # (B, A)
        probs     = log_probs.exp()                          # (B, A)

        return log_probs, probs

    def get_action(
        self,
        h:          Tensor,
        mask:       Optional[Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Sample (or argmax) an action.

        Returns:
            action    : (B,) int64   — chosen action indices
            log_prob  : (B,) float   — log π(chosen action | s)
            entropy   : (B,) float   — policy entropy H[π(·|s)]
        """
        log_probs, probs = self.forward(h, mask)

        if deterministic:
            action = probs.argmax(dim=-1)                    # (B,)
        else:
            # Categorical sampling
            action = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (B,)

        # Log prob of chosen action
        log_prob = log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)  # (B,)

        # Policy entropy: H[π] = −Σ π(a) log π(a)
        # Only over valid (non-masked) actions
        entropy = -(probs * log_probs).sum(dim=-1)           # (B,)
        # Clamp masked-action contributions (they contribute -inf * 0 = NaN)
        entropy = torch.nan_to_num(entropy, nan=0.0)

        return action, log_prob, entropy


# ─────────────────────────────────────────────────────────────────────────────
# Critic head (value function)
# ─────────────────────────────────────────────────────────────────────────────

class CriticHead(nn.Module):
    """
    Value function head V(s).

    Predicts the expected cumulative reward from state s under the
    current policy. Used to compute advantages: Â_t = Q(s,a) − V(s).

    Larger initial gain (1.0) vs actor (0.01) because the critic needs
    to output meaningful value predictions from the start, while the
    actor should start near-uniform to explore properly.
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.linear = nn.Linear(latent_dim, 1)
        nn.init.orthogonal_(self.linear.weight, gain=1.0)
        nn.init.zeros_(self.linear.bias)

    def forward(self, h: Tensor) -> Tensor:
        """
        Args:
            h : (B, latent_dim)

        Returns:
            value : (B,) — estimated V(s) for each state in the batch
        """
        return self.linear(h).squeeze(-1)                    # (B,)


# ─────────────────────────────────────────────────────────────────────────────
# Combined Actor-Critic module
# ─────────────────────────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    """
    Full Actor-Critic network.

    Wraps SharedBackbone + MaskedActorHead + CriticHead into a single
    module for easy parameter management and checkpoint saving.

    This module is used:
      1. Directly in rl/train_ppo.py for custom PPO training
      2. As the feature extractor in MaskedActorCriticPolicy for SB3

    The forward pass returns everything needed for the PPO loss:
      - action      : sampled or argmax action
      - log_prob    : log π(action | state)
      - entropy     : H[π(·|state)]  — for entropy regularisation
      - value       : V(state)       — for advantage estimation
    """

    def __init__(
        self,
        state_dim:  int         = ENV["state_dim"],
        action_dim: int         = NUM_QUESTIONS,
        net_arch:   List[int]   = PPO["policy_kwargs"]["net_arch"],
        dropout:    float       = 0.1,
    ):
        super().__init__()
        self.state_dim  = state_dim
        self.action_dim = action_dim

        self.backbone = SharedBackbone(state_dim, net_arch, dropout)
        self.actor    = MaskedActorHead(self.backbone.output_dim, action_dim)
        self.critic   = CriticHead(self.backbone.output_dim)

    def forward(
        self,
        state:        Tensor,                    # (B, state_dim)
        mask:         Optional[Tensor] = None,   # (B, action_dim) bool
        deterministic: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Full forward pass.

        Returns dict with:
            action    : (B,)
            log_prob  : (B,)
            entropy   : (B,)
            value     : (B,)
            probs     : (B, A)  — full probability distribution
        """
        h = self.backbone(state)

        action, log_prob, entropy = self.actor.get_action(h, mask, deterministic)
        value                     = self.critic(h)

        _, probs = self.actor(h, mask)

        return {
            "action":   action,
            "log_prob": log_prob,
            "entropy":  entropy,
            "value":    value,
            "probs":    probs,
        }

    def evaluate_actions(
        self,
        state:  Tensor,   # (B, state_dim)
        action: Tensor,   # (B,) int64 — actions taken during rollout
        mask:   Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Re-evaluate log_prob, entropy, value for actions taken during rollout.
        Called during the PPO update to compute the probability ratio r_t(θ).

        Returns:
            log_prob : (B,) — log π_θ(action | state) under NEW policy
            entropy  : (B,) — H[π_θ(·|state)]
            value    : (B,) — V_θ(state)
        """
        h = self.backbone(state)

        log_probs, probs = self.actor(h, mask)

        # Log prob of the SPECIFIC action taken
        log_prob = log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)

        # Entropy
        entropy  = -(probs * log_probs).sum(dim=-1)
        entropy  = torch.nan_to_num(entropy, nan=0.0)

        value = self.critic(h)

        return log_prob, entropy, value

    def get_value(self, state: Tensor) -> Tensor:
        """Critic-only forward pass — used during rollout collection."""
        h = self.backbone(state)
        return self.critic(h)


# ─────────────────────────────────────────────────────────────────────────────
# SB3 MaskablePPO policy class
# ─────────────────────────────────────────────────────────────────────────────

def build_policy_kwargs(
    net_arch: List[int] = PPO["policy_kwargs"]["net_arch"],
    dropout:  float     = 0.1,
) -> dict:
    """
    Build policy_kwargs dict for SB3 MaskablePPO.

    Pass to MaskablePPO(policy_kwargs=build_policy_kwargs()).
    SB3 will use net_arch to construct the MLP extractor automatically.

    For full custom control (e.g. adding dropout or LayerNorm), use
    ActorCritic directly in rl/train_ppo.py instead.
    """
    return {
        "net_arch":        dict(pi=net_arch, vf=net_arch),
        "activation_fn":   nn.ReLU,
        "ortho_init":      True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Utility: parameter count and layer summary
# ─────────────────────────────────────────────────────────────────────────────

def count_params(model: nn.Module) -> Dict[str, int]:
    """Return parameter counts for backbone, actor, critic, and total."""
    def n(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
    return {
        "backbone": n(model.backbone),
        "actor":    n(model.actor),
        "critic":   n(model.critic),
        "total":    n(model),
    }


def print_summary(model: ActorCritic) -> None:
    counts = count_params(model)
    print(f"\n{'─'*40}")
    print(f"  ActorCritic summary")
    print(f"{'─'*40}")
    print(f"  state_dim   : {model.state_dim}")
    print(f"  action_dim  : {model.action_dim}")
    print(f"  net_arch    : {PPO['policy_kwargs']['net_arch']}")
    print(f"{'─'*40}")
    print(f"  Backbone    : {counts['backbone']:>8,} params")
    print(f"  Actor head  : {counts['actor']:>8,} params")
    print(f"  Critic head : {counts['critic']:>8,} params")
    print(f"  {'Total':10}: {counts['total']:>8,} params")
    print(f"{'─'*40}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== ActorCritic smoke test ===")

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim  = ENV["state_dim"]
    action_dim = NUM_QUESTIONS
    B          = 8   # batch size

    model = ActorCritic(state_dim=state_dim, action_dim=action_dim).to(device)
    print_summary(model)

    # Fake batch
    state = torch.rand(B, state_dim, device=device)
    mask  = torch.ones(B, action_dim, dtype=torch.bool, device=device)
    # Mask out 80% of actions (simulating ZPD filter — most questions invalid)
    mask[:, action_dim // 5:] = False

    # ── Forward pass ──────────────────────────────────────────────────
    out = model(state, mask, deterministic=False)

    assert out["action"].shape   == (B,),             f"action shape: {out['action'].shape}"
    assert out["log_prob"].shape == (B,),             f"log_prob shape: {out['log_prob'].shape}"
    assert out["entropy"].shape  == (B,),             f"entropy shape: {out['entropy'].shape}"
    assert out["value"].shape    == (B,),             f"value shape: {out['value'].shape}"
    assert out["probs"].shape    == (B, action_dim),  f"probs shape: {out['probs'].shape}"

    # All sampled actions should respect the mask
    valid_actions = out["action"] < (action_dim // 5)
    assert valid_actions.all(), \
        f"Agent chose masked actions: {out['action'][~valid_actions]}"

    print(f"Forward pass OK")
    print(f"  action range  : [{out['action'].min()}, {out['action'].max()}]  "
          f"(valid: 0–{action_dim//5 - 1})")
    print(f"  log_prob range: [{out['log_prob'].min():.3f}, {out['log_prob'].max():.3f}]")
    print(f"  value range   : [{out['value'].min():.3f}, {out['value'].max():.3f}]")
    print(f"  entropy mean  : {out['entropy'].mean():.4f}  "
          f"(max possible: {np.log(action_dim//5):.4f})")

    # ── Masked actions get exactly 0 probability ──────────────────────
    masked_probs = out["probs"][:, action_dim // 5:]
    assert masked_probs.abs().max() < 1e-6, \
        f"Masked actions have non-zero probability: {masked_probs.max():.2e}"
    print(f"  Masked action prob max: {masked_probs.abs().max():.2e}  (should be ~0) ✓")

    # ── Probabilities sum to 1 over valid actions ─────────────────────
    prob_sum = out["probs"].sum(dim=-1)
    assert (prob_sum - 1.0).abs().max() < 1e-5, \
        f"Probs don't sum to 1: {prob_sum}"
    print(f"  Prob sum: {prob_sum.mean():.6f}  (should be 1.0) ✓")

    # ── Deterministic forward ─────────────────────────────────────────
    out_det = model(state, mask, deterministic=True)
    # Run twice — should give same result
    out_det2 = model(state, mask, deterministic=True)
    assert (out_det["action"] == out_det2["action"]).all(), \
        "Deterministic forward not reproducible"
    print(f"  Deterministic forward: reproducible ✓")

    # ── evaluate_actions (PPO update path) ────────────────────────────
    actions   = out["action"]
    log_prob2, entropy2, value2 = model.evaluate_actions(state, actions, mask)

    assert log_prob2.shape == (B,)
    assert entropy2.shape  == (B,)
    assert value2.shape    == (B,)
    assert torch.isfinite(log_prob2).all(), "Non-finite log_probs in evaluate_actions"
    assert torch.isfinite(entropy2).all(),  "Non-finite entropy in evaluate_actions"
    print(f"  evaluate_actions: shapes and finiteness OK ✓")

    # ── get_value (critic-only) ───────────────────────────────────────
    values = model.get_value(state)
    assert values.shape == (B,)
    assert torch.isfinite(values).all()
    print(f"  get_value: shape {values.shape}, finite ✓")

    # ── Gradient flow test ────────────────────────────────────────────
    loss = -log_prob2.mean() + 0.5 * (value2 ** 2).mean() - 0.01 * entropy2.mean()
    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"  [warn] No gradient: {name}")
        elif not torch.isfinite(param.grad).all():
            print(f"  [warn] Non-finite gradient: {name}")

    grad_norms = {
        "backbone": sum(
            p.grad.norm().item() ** 2
            for p in model.backbone.parameters() if p.grad is not None
        ) ** 0.5,
        "actor": sum(
            p.grad.norm().item() ** 2
            for p in model.actor.parameters() if p.grad is not None
        ) ** 0.5,
        "critic": sum(
            p.grad.norm().item() ** 2
            for p in model.critic.parameters() if p.grad is not None
        ) ** 0.5,
    }
    print(f"  Gradient norms — backbone: {grad_norms['backbone']:.4f}  "
          f"actor: {grad_norms['actor']:.4f}  critic: {grad_norms['critic']:.4f} ✓")

    # ── No-mask fallback ──────────────────────────────────────────────
    out_nomask = model(state, mask=None)
    prob_sum_nomask = out_nomask["probs"].sum(dim=-1)
    assert (prob_sum_nomask - 1.0).abs().max() < 1e-5
    print(f"  No-mask forward: probs sum to 1 ✓")

    print("\nAll smoke tests passed ✓")