"""
rl/train_ppo.py
Full PPO training loop for the Adaptive Learning agent.

Training cycle (repeats until total_timesteps reached):
  ┌─────────────────────────────────────────────────┐
  │  ROLLOUT PHASE  (n_steps = 2048)                │
  │  for each step:                                 │
  │    1. Get ZPD action mask from env              │
  │    2. Actor samples action  a_t ~ π(·|s_t)     │
  │    3. Critic estimates      V(s_t)              │
  │    4. Env steps →  s_{t+1}, r_t, done           │
  │    5. Store (s,a,r,done,logp,V) in buffer       │
  └─────────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────────┐
  │  UPDATE PHASE  (n_epochs = 10)                  │
  │  1. Compute GAE advantages from buffer          │
  │  2. Normalise advantages                        │
  │  3. For each mini-batch:                        │
  │     a. Re-evaluate actions under new policy     │
  │     b. Compute ratio r_t(θ) = π_new/π_old      │
  │     c. Compute clipped surrogate loss L_CLIP    │
  │     d. Compute value loss L_VF                  │
  │     e. Compute entropy bonus S[π]               │
  │     f. L = L_CLIP − c1·L_VF + c2·S[π]          │
  │     g. Gradient step + clip grad norm           │
  └─────────────────────────────────────────────────┘

Usage:
    python rl/train_ppo.py
    python rl/train_ppo.py --timesteps 200000 --n-envs 4
    python rl/train_ppo.py --resume checkpoints/ppo_best.zip
"""

import argparse
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ENV, MLFLOW, NUM_CONCEPTS, NUM_QUESTIONS, PPO, EVAL
from rl.actor_critic import ActorCritic, count_params, print_summary
from rl.env import AdaptiveLearningEnv, make_env
from rl.reward import RewardShaper


# ─────────────────────────────────────────────────────────────────────────────
# Rollout buffer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RolloutBuffer:
    """
    Stores one rollout of n_steps transitions for PPO updates.

    All tensors are pre-allocated for efficiency — no list appending.
    Supports multiple parallel environments (n_envs > 1).

    Buffer layout  (n_steps × n_envs):
      states        : (T, E, state_dim)
      actions       : (T, E)
      rewards       : (T, E)
      dones         : (T, E)    1 = episode ended
      log_probs_old : (T, E)    log π_old(a|s) at collection time
      values        : (T, E)    V(s) at collection time
      masks         : (T, E, A) ZPD action masks
    """

    n_steps:   int
    n_envs:    int
    state_dim: int
    action_dim: int
    device:    torch.device

    # Filled during rollout
    states:        torch.Tensor = field(init=False)
    actions:       torch.Tensor = field(init=False)
    rewards:       torch.Tensor = field(init=False)
    dones:         torch.Tensor = field(init=False)
    log_probs_old: torch.Tensor = field(init=False)
    values:        torch.Tensor = field(init=False)
    masks:         torch.Tensor = field(init=False)

    # Computed after rollout
    advantages:    torch.Tensor = field(init=False)
    returns:       torch.Tensor = field(init=False)

    _pos: int = field(init=False, default=0)
    _full: bool = field(init=False, default=False)

    def __post_init__(self):
        T, E, S, A = self.n_steps, self.n_envs, self.state_dim, self.action_dim
        d = self.device
        self.states        = torch.zeros(T, E, S, device=d)
        self.actions       = torch.zeros(T, E, dtype=torch.long, device=d)
        self.rewards       = torch.zeros(T, E, device=d)
        self.dones         = torch.zeros(T, E, device=d)
        self.log_probs_old = torch.zeros(T, E, device=d)
        self.values        = torch.zeros(T, E, device=d)
        self.masks         = torch.ones(T, E, A, dtype=torch.bool, device=d)
        self.advantages    = torch.zeros(T, E, device=d)
        self.returns       = torch.zeros(T, E, device=d)
        self._pos  = 0
        self._full = False

    def add(
        self,
        state:    torch.Tensor,   # (E, S)
        action:   torch.Tensor,   # (E,)
        reward:   torch.Tensor,   # (E,)
        done:     torch.Tensor,   # (E,)
        log_prob: torch.Tensor,   # (E,)
        value:    torch.Tensor,   # (E,)
        mask:     torch.Tensor,   # (E, A)
    ):
        t = self._pos
        self.states[t]        = state
        self.actions[t]       = action
        self.rewards[t]       = reward
        self.dones[t]         = done
        self.log_probs_old[t] = log_prob
        self.values[t]        = value
        self.masks[t]         = mask
        self._pos = (self._pos + 1) % self.n_steps
        self._full = self._full or (self._pos == 0)

    def compute_gae(
        self,
        last_values: torch.Tensor,   # (E,)  V(s_{T}) — bootstrapped
        last_dones:  torch.Tensor,   # (E,)
        gamma:   float = PPO["gamma"],
        lam:     float = PPO["gae_lambda"],
    ):
        """
        Compute Generalised Advantage Estimation (GAE-λ).

        GAE formula:
          δ_t   = r_t + γ·V(s_{t+1})·(1−done_t) − V(s_t)
          Â_t   = δ_t + (γλ)·δ_{t+1} + (γλ)²·δ_{t+2} + ...
                = δ_t + γλ·Â_{t+1}·(1−done_t)

        Iterates backwards from T-1 to 0 in O(T) time.
        Returns are Â_t + V(s_t)  (used as critic targets).
        """
        gae = torch.zeros(self.n_envs, device=self.device)

        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - last_dones.float()
                next_values       = last_values
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_values       = self.values[t + 1]

            # TD error δ_t
            delta = (
                self.rewards[t]
                + gamma * next_values * next_non_terminal
                - self.values[t]
            )

            # GAE recursive update
            gae = delta + gamma * lam * next_non_terminal * gae
            self.advantages[t] = gae

        # Returns = advantages + values  (used as V targets in L_VF)
        self.returns = self.advantages + self.values

    def get_minibatches(self, batch_size: int):
        """
        Yield random mini-batches from the flattened rollout buffer.

        Flattens (T × E) into (T*E,) then shuffles and yields
        chunks of size batch_size.

        Yields dicts with keys:
            states, actions, log_probs_old, advantages, returns, masks
        """
        T, E = self.n_steps, self.n_envs
        N    = T * E

        # Flatten all tensors from (T, E, ...) to (N, ...)
        flat = {
            "states":        self.states.view(N, self.state_dim),
            "actions":       self.actions.view(N),
            "log_probs_old": self.log_probs_old.view(N),
            "advantages":    self.advantages.view(N),
            "returns":       self.returns.view(N),
            "masks":         self.masks.view(N, self.action_dim),
        }

        # Normalise advantages (per-rollout, not running)
        adv = flat["advantages"]
        flat["advantages"] = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Random permutation for unbiased mini-batches
        indices = torch.randperm(N, device=self.device)

        for start in range(0, N, batch_size):
            idx = indices[start : start + batch_size]
            if len(idx) < batch_size // 2:
                continue   # skip tiny last batch
            yield {k: v[idx] for k, v in flat.items()}


# ─────────────────────────────────────────────────────────────────────────────
# PPO loss
# ─────────────────────────────────────────────────────────────────────────────

def ppo_loss(
    model:       ActorCritic,
    batch:       dict,
    clip_range:  float = PPO["clip_range"],
    vf_coef:     float = PPO["vf_coef"],
    ent_coef:    float = PPO["ent_coef"],
) -> tuple[torch.Tensor, dict]:
    """
    Compute the PPO clipped objective.

    L_PPO = L_CLIP − c₁·L_VF + c₂·S[π]

    where:
      L_CLIP = E[min(r_t·Â, clip(r_t, 1−ε, 1+ε)·Â)]
      L_VF   = MSE(V_θ(s), returns)
      S[π]   = E[H[π_θ(·|s)]]  (entropy bonus)

    Returns:
        total_loss : scalar to call .backward() on
        metrics    : dict of component losses for logging
    """
    states        = batch["states"]
    actions       = batch["actions"]
    log_probs_old = batch["log_probs_old"]
    advantages    = batch["advantages"]
    returns       = batch["returns"]
    masks         = batch["masks"]

    # Re-evaluate actions under current policy θ
    log_probs_new, entropy, values = model.evaluate_actions(
        states, actions, masks
    )

    # ── Policy ratio r_t(θ) = π_θ(a|s) / π_θ_old(a|s) ───────────────
    # Computed in log space for numerical stability
    log_ratio = log_probs_new - log_probs_old
    ratio     = log_ratio.exp()

    # ── Clipped surrogate loss L_CLIP ─────────────────────────────────
    # min of unclipped and clipped objective
    # minus sign: we maximise, PyTorch minimises
    loss_unclipped = ratio * advantages
    loss_clipped   = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
    loss_policy    = -torch.min(loss_unclipped, loss_clipped).mean()

    # ── Value function loss L_VF ──────────────────────────────────────
    loss_value = nn.functional.mse_loss(values, returns)

    # ── Entropy bonus S[π] ────────────────────────────────────────────
    # Maximise entropy → discourage premature policy collapse
    loss_entropy = -entropy.mean()   # negative because we maximise

    # ── Total PPO loss ────────────────────────────────────────────────
    total = loss_policy + vf_coef * loss_value + ent_coef * loss_entropy

    # ── Diagnostics ───────────────────────────────────────────────────
    with torch.no_grad():
        # Approximate KL divergence (for early stopping / monitoring)
        approx_kl = ((ratio - 1) - log_ratio).mean().item()
        # Clip fraction: how often the clip was active
        clip_frac  = ((ratio - 1.0).abs() > clip_range).float().mean().item()
        explained_var = _explained_variance(values.detach(), returns)

    metrics = {
        "loss/total":        total.item(),
        "loss/policy":       loss_policy.item(),
        "loss/value":        loss_value.item(),
        "loss/entropy":      (-loss_entropy).item(),   # log as positive entropy
        "ppo/approx_kl":     approx_kl,
        "ppo/clip_fraction": clip_frac,
        "ppo/explained_var": explained_var,
    }

    return total, metrics


def _explained_variance(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Fraction of variance in returns explained by the value function.
    1.0 = perfect, 0.0 = no better than mean, <0 = worse than mean.
    """
    var_y = target.var()
    if var_y < 1e-8:
        return float("nan")
    return float(1 - (target - pred).var() / var_y)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(
    total_timesteps: int   = PPO["total_timesteps"],
    n_steps:         int   = PPO["n_steps"],
    batch_size:      int   = PPO["batch_size"],
    n_epochs:        int   = PPO["n_epochs"],
    learning_rate:   float = PPO["learning_rate"],
    gamma:           float = PPO["gamma"],
    gae_lambda:      float = PPO["gae_lambda"],
    clip_range:      float = PPO["clip_range"],
    ent_coef:        float = PPO["ent_coef"],
    vf_coef:         float = PPO["vf_coef"],
    max_grad_norm:   float = PPO["max_grad_norm"],
    n_envs:          int   = 1,
    resume:          Optional[str] = None,
    seed:            int   = EVAL["random_seed"],
) -> ActorCritic:
    """
    Full PPO training loop. Returns the best trained model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*55}")
    print(f"  PPO Training — Adaptive Learning Agent")
    print(f"{'='*55}")
    print(f"  Device        : {device}")
    print(f"  Timesteps     : {total_timesteps:,}")
    print(f"  n_steps       : {n_steps}  (rollout length per env)")
    print(f"  n_envs        : {n_envs}")
    print(f"  batch_size    : {batch_size}")
    print(f"  n_epochs      : {n_epochs}")
    print(f"  γ / λ         : {gamma} / {gae_lambda}")
    print(f"  clip ε        : {clip_range}")
    print(f"  lr            : {learning_rate}")
    print(f"{'='*55}\n")

    # ── Environments ──────────────────────────────────────────────────
    envs = [AdaptiveLearningEnv() for _ in range(n_envs)]

    state_dim  = ENV["state_dim"]
    action_dim = NUM_QUESTIONS

    # ── Model & optimiser ─────────────────────────────────────────────
    model = ActorCritic(
        state_dim  = state_dim,
        action_dim = action_dim,
    ).to(device)

    if resume and Path(resume).exists():
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"  Resumed from: {resume}\n")

    optimizer = Adam(model.parameters(), lr=learning_rate, eps=1e-5)
    print_summary(model)

    # ── Rollout buffer ────────────────────────────────────────────────
    buffer = RolloutBuffer(
        n_steps    = n_steps,
        n_envs     = n_envs,
        state_dim  = state_dim,
        action_dim = action_dim,
        device     = device,
    )

    # ── Episode tracking ──────────────────────────────────────────────
    ep_rewards        = deque(maxlen=100)   # last 100 episode returns
    ep_mastery_gains  = deque(maxlen=100)
    ep_lengths        = deque(maxlen=100)
    best_mean_reward  = -float("inf")

    # ── Initial reset ─────────────────────────────────────────────────
    obs_list   = []
    mask_list  = []
    info_list  = []
    for env in envs:
        o, info = env.reset(seed=seed)
        obs_list.append(o)
        mask_list.append(info["action_mask"])
        info_list.append(info)

    current_obs  = torch.tensor(np.array(obs_list),  dtype=torch.float32, device=device)
    current_mask = torch.tensor(np.array(mask_list), dtype=torch.bool,    device=device)

    # Per-env episode trackers
    ep_reward_buf = np.zeros(n_envs)
    ep_mastery_start = np.array([
        info["mastery_mean"] for info in info_list
    ])

    # ── MLflow ────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW["tracking_uri"])
    mlflow.set_experiment(MLFLOW["experiment_name"])

    global_step   = 0
    n_updates     = 0
    update_t0     = time.time()

    with mlflow.start_run(run_name="ppo_training"):
        mlflow.log_params({
            "total_timesteps": total_timesteps,
            "n_steps":         n_steps,
            "n_envs":          n_envs,
            "batch_size":      batch_size,
            "n_epochs":        n_epochs,
            "gamma":           gamma,
            "gae_lambda":      gae_lambda,
            "clip_range":      clip_range,
            "ent_coef":        ent_coef,
            "vf_coef":         vf_coef,
            "learning_rate":   learning_rate,
            "state_dim":       state_dim,
            "action_dim":      action_dim,
            "n_params":        count_params(model)["total"],
        })

        # ── Main training loop ────────────────────────────────────────
        while global_step < total_timesteps:

            # ── ROLLOUT PHASE ─────────────────────────────────────────
            model.eval()
            rollout_t0 = time.time()

            for step in range(n_steps):
                with torch.no_grad():
                    out = model(current_obs, current_mask, deterministic=False)

                actions   = out["action"]      # (E,)
                log_probs = out["log_prob"]     # (E,)
                values    = out["value"]        # (E,)

                # Step all environments
                next_obs_list   = []
                next_mask_list  = []
                rewards_list    = []
                dones_list      = []

                for e, env in enumerate(envs):
                    a = int(actions[e].item())
                    obs_next, reward, terminated, truncated, info = env.step(a)
                    done = terminated or truncated

                    rewards_list.append(reward)
                    dones_list.append(float(done))
                    next_obs_list.append(obs_next)
                    ep_reward_buf[e] += reward

                    if done:
                        ep_rewards.append(ep_reward_buf[e])
                        ep_mastery_gains.append(
                            info["mastery_mean"] - ep_mastery_start[e]
                        )
                        ep_lengths.append(info["n_questions"])
                        ep_reward_buf[e] = 0.0

                        # Reset this env
                        obs_next, info = env.reset()
                        ep_mastery_start[e] = info["mastery_mean"]
                        next_mask_list.append(info["action_mask"])
                    else:
                        next_mask_list.append(info["action_mask"])

                # Store transition
                buffer.add(
                    state    = current_obs,
                    action   = actions,
                    reward   = torch.tensor(rewards_list,  device=device),
                    done     = torch.tensor(dones_list,    device=device),
                    log_prob = log_probs,
                    value    = values,
                    mask     = current_mask,
                )

                current_obs  = torch.tensor(
                    np.array(next_obs_list),  dtype=torch.float32, device=device
                )
                current_mask = torch.tensor(
                    np.array(next_mask_list), dtype=torch.bool,    device=device
                )
                global_step += n_envs

            # Bootstrap final value
            with torch.no_grad():
                last_values = model.get_value(current_obs)
            last_dones = torch.zeros(n_envs, device=device)

            # ── GAE ───────────────────────────────────────────────────
            buffer.compute_gae(last_values, last_dones, gamma, gae_lambda)

            # ── UPDATE PHASE ──────────────────────────────────────────
            model.train()
            epoch_metrics = []

            for epoch in range(n_epochs):
                for batch in buffer.get_minibatches(batch_size):
                    loss, metrics = ppo_loss(
                        model, batch, clip_range, vf_coef, ent_coef
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        model.parameters(), max_grad_norm
                    )
                    optimizer.step()
                    epoch_metrics.append(metrics)

            n_updates += 1
            rollout_time = time.time() - rollout_t0

            # ── Aggregate metrics ─────────────────────────────────────
            avg = {
                k: float(np.mean([m[k] for m in epoch_metrics]))
                for k in epoch_metrics[0]
            }

            fps = int(n_steps * n_envs / rollout_time)
            mean_reward = float(np.mean(ep_rewards)) if ep_rewards else 0.0
            mean_mastery_gain = float(np.mean(ep_mastery_gains)) if ep_mastery_gains else 0.0
            mean_ep_len = float(np.mean(ep_lengths)) if ep_lengths else 0.0

            # ── Log to MLflow ─────────────────────────────────────────
            log_dict = {
                "rollout/mean_reward":      mean_reward,
                "rollout/mean_mastery_gain":mean_mastery_gain,
                "rollout/mean_ep_length":   mean_ep_len,
                "rollout/fps":              fps,
                **avg,
            }
            mlflow.log_metrics(log_dict, step=global_step)

            # ── Console output ────────────────────────────────────────
            elapsed = time.time() - update_t0
            print(
                f"  step {global_step:>7,} | "
                f"reward {mean_reward:>7.3f} | "
                f"mastery↑ {mean_mastery_gain:>6.4f} | "
                f"kl {avg['ppo/approx_kl']:>6.4f} | "
                f"clip {avg['ppo/clip_fraction']:>5.3f} | "
                f"ev {avg['ppo/explained_var']:>6.3f} | "
                f"fps {fps:>5}"
            )

            # ── Checkpoint best model ─────────────────────────────────
            if mean_reward > best_mean_reward and len(ep_rewards) >= 10:
                best_mean_reward = mean_reward
                torch.save(
                    {
                        "model_state":     model.state_dict(),
                        "global_step":     global_step,
                        "mean_reward":     mean_reward,
                        "mean_mastery_gain": mean_mastery_gain,
                        "hyperparams": {
                            "gamma": gamma, "gae_lambda": gae_lambda,
                            "clip_range": clip_range, "lr": learning_rate,
                        }
                    },
                    PPO["checkpoint_path"],
                )
                print(f"    ✓ checkpoint saved (reward={mean_reward:.3f})")

            # Early exit for quick experiments
            if global_step >= total_timesteps:
                break

        # ── Final stats ────────────────────────────────────────────────
        total_time = time.time() - update_t0
        print(f"\n{'='*55}")
        print(f"  Training complete")
        print(f"  Total time   : {total_time/60:.1f} min")
        print(f"  Updates      : {n_updates}")
        print(f"  Best reward  : {best_mean_reward:.4f}")
        print(f"  Checkpoint   : {PPO['checkpoint_path']}")
        print(f"{'='*55}\n")

        mlflow.log_metrics({
            "final/best_mean_reward": best_mean_reward,
            "final/n_updates":        n_updates,
            "final/total_time_min":   total_time / 60,
        }, step=global_step)

    # Load and return best checkpoint
    if PPO["checkpoint_path"].exists():
        ckpt = torch.load(PPO["checkpoint_path"], map_location=device)
        model.load_state_dict(ckpt["model_state"])

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train PPO adaptive learning agent")
    parser.add_argument("--timesteps", type=int,   default=PPO["total_timesteps"])
    parser.add_argument("--n-envs",    type=int,   default=1)
    parser.add_argument("--n-steps",   type=int,   default=PPO["n_steps"])
    parser.add_argument("--batch",     type=int,   default=PPO["batch_size"])
    parser.add_argument("--epochs",    type=int,   default=PPO["n_epochs"])
    parser.add_argument("--lr",        type=float, default=PPO["learning_rate"])
    parser.add_argument("--clip",      type=float, default=PPO["clip_range"])
    parser.add_argument("--ent-coef",  type=float, default=PPO["ent_coef"])
    parser.add_argument("--seed",      type=int,   default=EVAL["random_seed"])
    parser.add_argument("--resume",    type=str,   default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    train(
        total_timesteps = args.timesteps,
        n_envs          = args.n_envs,
        n_steps         = args.n_steps,
        batch_size      = args.batch,
        n_epochs        = args.epochs,
        learning_rate   = args.lr,
        clip_range      = args.clip,
        ent_coef        = args.ent_coef,
        seed            = args.seed,
        resume          = args.resume,
    )


if __name__ == "__main__":
    main()