"""
analysis/ablation.py
Ablation study for the Adaptive Learning System's PPO agent.

Answers the thesis question: "which components actually matter?"
by selectively disabling each one and re-running evaluation episodes
with a *fixed, already-trained* PPO policy.

Ablations covered:
  1. No ZPD masking        — zpd_lower=0.0, zpd_upper=1.0 (agent picks
                              any question, not just ones in the
                              student's Zone of Proximal Development)
  2. No mastery reward (α=0) — agent gets no signal for Δmastery
  3. No correctness bonus (β=0)
  4. No hint penalty (γ=0)
  5. No time penalty (δ=0)
  6. No episode-end bonus (λ=0)
  7. Full model (all components active) — baseline for comparison

Design note — why ablate at evaluation time, not retrain from scratch:
  Retraining 7 separate PPO models (500k timesteps each) is extremely
  expensive and conflates "what the policy was trained to optimise"
  with "what we evaluate it against". This script does the standard,
  cheaper ablation: it takes ONE trained policy and asks "how much
  does removing each reward/environment component change behaviour
  and outcomes when this same policy is deployed?" This isolates each
  component's contribution to *evaluation-time pedagogical quality*.

  For a true training-time ablation (more rigorous, much slower), use
  --retrain, which calls rl/train_ppo.py::train() once per ablation
  with a reduced timestep budget (--retrain-timesteps).

Output files (in logs/ablation/):
  ablation_results.json    — raw per-ablation episode data
  ablation_table.csv       — summary table (thesis-ready)
  ablation_comparison.png  — bar chart: reward & gain per ablation
  component_contribution.png — % drop in performance per component

Usage:
  # Fast ablation using existing trained checkpoint (recommended):
  python analysis/ablation.py

  python analysis/ablation.py --episodes 100
  python analysis/ablation.py --checkpoint checkpoints/ppo_best.zip

  # Slow, rigorous ablation — retrain a small model per condition:
  python analysis/ablation.py --retrain --retrain-timesteps 50000
"""

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ENV, EVAL, LOG_DIR, NUM_QUESTIONS, PPO, REWARD
from rl.actor_critic import ActorCritic
from rl.env import AdaptiveLearningEnv
from rl.reward import RewardShaper
from rl.student_simulator import SimulatorEnsemble

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

ABLATION_DIR = LOG_DIR / "ablation"
ABLATION_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = EVAL["alpha"]

COLORS = {
    "Full model":         "#7F77DD",
    "No ZPD mask":        "#EF9F27",
    "No mastery reward":  "#F0997B",
    "No correctness bonus": "#85B7EB",
    "No hint penalty":    "#5DCAA5",
    "No time penalty":    "#C792EA",
    "No episode bonus":   "#FFB6A3",
}


# ─────────────────────────────────────────────────────────────────────────────
# Ablation environment wrapper
# ─────────────────────────────────────────────────────────────────────────────

class AblatedEnv(AdaptiveLearningEnv):
    """
    Subclass of AdaptiveLearningEnv that lets us override the
    RewardShaper instance after construction.

    AdaptiveLearningEnv.__init__ hardcodes `self.reward_shaper =
    RewardShaper()` using config defaults, with no constructor
    parameter to inject a custom shaper. Rather than edit env.py,
    we re-assign self.reward_shaper immediately after super().__init__()
    runs — this is safe because reward_shaper is only read inside
    .step(), which always happens after construction completes.
    """

    def __init__(self, reward_shaper: Optional[RewardShaper] = None, **kwargs):
        super().__init__(**kwargs)
        if reward_shaper is not None:
            self.reward_shaper = reward_shaper


# ─────────────────────────────────────────────────────────────────────────────
# Ablation condition definitions
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AblationCondition:
    """
    One ablation condition: a name plus a factory function that builds
    the environment for that condition.
    """
    name:        str
    description: str
    env_factory: Callable[[SimulatorEnsemble], AdaptiveLearningEnv]


def _full_model_env(ensemble: SimulatorEnsemble) -> AdaptiveLearningEnv:
    """Baseline: every component active, matches production config."""
    return AblatedEnv(
        ensemble          = ensemble,
        use_fresh_student = True,
        reward_shaper     = RewardShaper(),  # default weights
    )


def _no_zpd_env(ensemble: SimulatorEnsemble) -> AdaptiveLearningEnv:
    """
    Disable ZPD masking entirely by widening the band to [0, 1] —
    every question becomes "valid", so the agent can pick anything
    regardless of whether it matches the student's current ability.
    """
    return AblatedEnv(
        ensemble          = ensemble,
        use_fresh_student = True,
        zpd_lower         = 0.0,
        zpd_upper         = 1.0,
        reward_shaper     = RewardShaper(),
    )


def _no_mastery_reward_env(ensemble: SimulatorEnsemble) -> AdaptiveLearningEnv:
    """Zero out α — agent gets no signal for Δmastery (primary reward)."""
    return AblatedEnv(
        ensemble          = ensemble,
        use_fresh_student = True,
        reward_shaper     = RewardShaper(alpha=0.0),
    )


def _no_correctness_env(ensemble: SimulatorEnsemble) -> AdaptiveLearningEnv:
    """Zero out β — agent gets no bonus for correct answers."""
    return AblatedEnv(
        ensemble          = ensemble,
        use_fresh_student = True,
        reward_shaper     = RewardShaper(beta=0.0),
    )


def _no_hint_penalty_env(ensemble: SimulatorEnsemble) -> AdaptiveLearningEnv:
    """Zero out γ — no penalty for hint usage."""
    return AblatedEnv(
        ensemble          = ensemble,
        use_fresh_student = True,
        reward_shaper     = RewardShaper(gamma=0.0),
    )


def _no_time_penalty_env(ensemble: SimulatorEnsemble) -> AdaptiveLearningEnv:
    """Zero out δ — no penalty for taking too long."""
    return AblatedEnv(
        ensemble          = ensemble,
        use_fresh_student = True,
        reward_shaper     = RewardShaper(delta=0.0),
    )


def _no_episode_bonus_env(ensemble: SimulatorEnsemble) -> AdaptiveLearningEnv:
    """Zero out λ — no end-of-episode mastery-gain bonus."""
    return AblatedEnv(
        ensemble          = ensemble,
        use_fresh_student = True,
        reward_shaper     = RewardShaper(lam=0.0),
    )


CONDITIONS: list[AblationCondition] = [
    AblationCondition("Full model",           "All components active (baseline)",        _full_model_env),
    AblationCondition("No ZPD mask",          "Action space not restricted to ZPD",      _no_zpd_env),
    AblationCondition("No mastery reward",    "α=0 — no Δmastery signal",                 _no_mastery_reward_env),
    AblationCondition("No correctness bonus", "β=0 — no per-step correctness reward",     _no_correctness_env),
    AblationCondition("No hint penalty",      "γ=0 — hints are free",                     _no_hint_penalty_env),
    AblationCondition("No time penalty",      "δ=0 — no penalty for slow responses",      _no_time_penalty_env),
    AblationCondition("No episode bonus",     "λ=0 — no end-of-session bonus",            _no_episode_bonus_env),
]


# ─────────────────────────────────────────────────────────────────────────────
# Policy loading
# ─────────────────────────────────────────────────────────────────────────────

def load_policy(checkpoint: str, device: torch.device) -> ActorCritic:
    """Load a trained ActorCritic model from a PPO checkpoint."""
    model = ActorCritic(
        state_dim  = ENV["state_dim"],
        action_dim = NUM_QUESTIONS,
    ).to(device)

    ckpt_path = Path(checkpoint)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"  Loaded checkpoint: {checkpoint}")
    else:
        print(f"  [warn] Checkpoint not found at {checkpoint}.")
        print("  Using a freshly initialised (untrained) policy instead —")
        print("  ablation results will reflect random-policy behaviour.")

    model.eval()
    return model


def make_policy_fn(model: ActorCritic, deterministic: bool = True) -> Callable:
    """Wrap an ActorCritic model as an env -> action policy function."""
    def policy(env: AdaptiveLearningEnv, obs: np.ndarray) -> int:
        with torch.no_grad():
            state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            mask  = torch.tensor(env.action_masks(), dtype=torch.bool).unsqueeze(0)
            out   = model(state, mask, deterministic=deterministic)
            return int(out["action"].item())
    return policy


# ─────────────────────────────────────────────────────────────────────────────
# Episode runner (mirrors rl/evaluate.py::run_episode)
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(env: AdaptiveLearningEnv, policy: Callable, seed: Optional[int] = None) -> dict:
    """
    Run one episode and collect the same metrics as evaluate.py, so
    ablation results are directly comparable to baseline-comparison
    results.
    """
    obs, info = env.reset(seed=seed)

    mastery_before = info["mastery_mean"]
    total_reward   = 0.0
    zpd_steps      = 0
    done           = False

    while not done:
        action = policy(env, obs)
        mask   = env.action_masks()
        if mask[action]:
            zpd_steps += 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    mastery_after = info["mastery_mean"]
    n_q           = info["n_questions"]
    potential     = 1.0 - mastery_before
    norm_gain     = float(np.clip(
        (mastery_after - mastery_before) / max(potential, 1e-6), 0.0, 1.0
    ))

    return {
        "total_reward":   total_reward,
        "mastery_before": mastery_before,
        "mastery_after":  mastery_after,
        "learning_gain":  norm_gain,
        "n_questions":    n_q,
        "zpd_adherence":  zpd_steps / max(n_q, 1),
    }


def run_condition(
    condition:  AblationCondition,
    policy_fn:  Callable,
    n_episodes: int,
    seed:       int,
) -> dict:
    """Run n_episodes for one ablation condition."""
    print(f"  Running [{condition.name}] ({condition.description}) "
          f"— {n_episodes} episodes …")

    ensemble = SimulatorEnsemble(seed=seed)
    env      = condition.env_factory(ensemble)
    rng      = np.random.default_rng(seed)

    episodes = []
    for _ in range(n_episodes):
        ep_seed = int(rng.integers(0, 100_000))
        episodes.append(run_episode(env, policy_fn, seed=ep_seed))

    rewards = [e["total_reward"]  for e in episodes]
    gains   = [e["learning_gain"] for e in episodes]
    zpd     = [e["zpd_adherence"] for e in episodes]
    n_q     = [e["n_questions"]   for e in episodes]

    def ci95(data):
        a  = np.array(data)
        se = stats.sem(a)
        t  = stats.t.ppf(0.975, df=len(a) - 1)
        return float(a.mean()), float(t * se)

    mean_r, ci_r = ci95(rewards)
    mean_g, ci_g = ci95(gains)
    mean_z, ci_z = ci95(zpd)
    mean_q, ci_q = ci95(n_q)

    print(f"    reward={mean_r:+.3f}±{ci_r:.3f}  gain={mean_g:.4f}±{ci_g:.4f}  "
          f"zpd={mean_z:.3f}±{ci_z:.3f}")

    return {
        "name":        condition.name,
        "description": condition.description,
        "episodes":    episodes,
        "stats": {
            "mean_reward": mean_r, "ci_reward": ci_r,
            "mean_gain":   mean_g, "ci_gain":   ci_g,
            "mean_zpd":    mean_z, "ci_zpd":    ci_z,
            "mean_n_questions": mean_q, "ci_n_questions": ci_q,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Statistical comparison vs full model
# ─────────────────────────────────────────────────────────────────────────────

def compare_to_baseline(results: dict[str, dict]) -> dict:
    """
    Welch's t-test + Cohen's d for each ablation vs "Full model",
    on both reward and learning gain.
    """
    baseline = results["Full model"]
    base_r   = [e["total_reward"]  for e in baseline["episodes"]]
    base_g   = [e["learning_gain"] for e in baseline["episodes"]]

    def cohens_d(a, b):
        a, b = np.array(a), np.array(b)
        pooled = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
        return float((a.mean() - b.mean()) / max(pooled, 1e-9))

    comparisons = {}
    for name, data in results.items():
        if name == "Full model":
            continue
        r = [e["total_reward"]  for e in data["episodes"]]
        g = [e["learning_gain"] for e in data["episodes"]]

        t_r, p_r = stats.ttest_ind(base_r, r, equal_var=False)
        t_g, p_g = stats.ttest_ind(base_g, g, equal_var=False)
        d_r      = cohens_d(base_r, r)
        d_g      = cohens_d(base_g, g)

        pct_drop_r = (np.mean(base_r) - np.mean(r)) / max(abs(np.mean(base_r)), 1e-6) * 100
        pct_drop_g = (np.mean(base_g) - np.mean(g)) / max(np.mean(base_g), 1e-6) * 100

        comparisons[name] = {
            "reward": {"t": t_r, "p": p_r, "cohens_d": d_r,
                       "significant": p_r < ALPHA, "pct_drop": pct_drop_r},
            "gain":   {"t": t_g, "p": p_g, "cohens_d": d_g,
                       "significant": p_g < ALPHA, "pct_drop": pct_drop_g},
        }

    return comparisons


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def save_table(results: dict, comparisons: dict) -> Path:
    """Write a thesis-ready CSV summary table."""
    path = ABLATION_DIR / "ablation_table.csv"
    rows = []

    for name, data in results.items():
        s = data["stats"]
        row = {
            "Condition":        name,
            "Description":      data["description"],
            "Mean Reward":      f"{s['mean_reward']:+.3f}",
            "CI Reward":        f"±{s['ci_reward']:.3f}",
            "Mean Gain":        f"{s['mean_gain']:.4f}",
            "CI Gain":          f"±{s['ci_gain']:.4f}",
            "ZPD %":            f"{s['mean_zpd']*100:.1f}%",
            "Ep Length":        f"{s['mean_n_questions']:.1f}",
            "Reward Δ% vs Full": "—",
            "Gain Δ% vs Full":   "—",
            "p (reward)":        "—",
            "Significant":       "—",
        }
        if name in comparisons:
            c = comparisons[name]
            row["Reward Δ% vs Full"] = f"{c['reward']['pct_drop']:+.1f}%"
            row["Gain Δ% vs Full"]   = f"{c['gain']['pct_drop']:+.1f}%"
            row["p (reward)"]        = f"{c['reward']['p']:.4f}"
            row["Significant"]       = "Yes" if c["reward"]["significant"] else "No"
        rows.append(row)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Saved: {path}")
    return path


def print_table(results: dict, comparisons: dict):
    """Print formatted ablation table to console."""
    print(f"\n{'─'*92}")
    print(f"  {'Condition':<22} {'Reward':>12} {'Gain':>12} {'ZPD%':>7} "
          f"{'Δ% Reward':>10} {'p-val':>8} {'Sig':>5}")
    print(f"{'─'*92}")
    for name, data in results.items():
        s = data["stats"]
        c = comparisons.get(name)
        drop = f"{c['reward']['pct_drop']:+.1f}%" if c else "  —  "
        pval = f"{c['reward']['p']:.4f}"          if c else "  —   "
        sig  = ("Yes" if c["reward"]["significant"] else "No") if c else " — "
        print(
            f"  {name:<22} "
            f"{s['mean_reward']:>+7.3f}±{s['ci_reward']:.3f} "
            f"{s['mean_gain']:>7.4f}±{s['ci_gain']:.4f} "
            f"{s['mean_zpd']*100:>5.1f}% "
            f"{drop:>10} {pval:>8} {sig:>5}"
        )
    print(f"{'─'*92}")


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_ablation_comparison(results: dict, save: bool = True):
    """Grouped bar chart: mean reward and mean gain per condition."""
    names  = list(results.keys())
    colors = [COLORS.get(n, "#AAAAAA") for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    rewards = [results[n]["stats"]["mean_reward"] for n in names]
    r_cis   = [results[n]["stats"]["ci_reward"]    for n in names]
    ax1.bar(names, rewards, color=colors, alpha=0.85, yerr=r_cis, capsize=4)
    ax1.set_ylabel("Mean episode reward", fontsize=11)
    ax1.set_title("Reward per Ablation Condition", fontsize=12)
    ax1.tick_params(axis="x", rotation=35)
    ax1.grid(axis="y", alpha=0.3)

    gains  = [results[n]["stats"]["mean_gain"] for n in names]
    g_cis  = [results[n]["stats"]["ci_gain"]   for n in names]
    ax2.bar(names, gains, color=colors, alpha=0.85, yerr=g_cis, capsize=4)
    ax2.set_ylabel("Mean normalised learning gain", fontsize=11)
    ax2.set_title("Learning Gain per Ablation Condition", fontsize=12)
    ax2.tick_params(axis="x", rotation=35)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    if save:
        path = ABLATION_DIR / "ablation_comparison.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def plot_component_contribution(comparisons: dict, save: bool = True):
    """
    Bar chart of % performance drop when each component is removed —
    directly answers "how much does each component matter?"
    """
    names = list(comparisons.keys())
    drops = [comparisons[n]["reward"]["pct_drop"] for n in names]
    colors = [COLORS.get(n, "#AAAAAA") for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(names, drops, color=colors, alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("% drop in reward vs full model", fontsize=11)
    ax.set_title("Component Contribution to Performance", fontsize=12)
    ax.grid(axis="x", alpha=0.3)

    for bar, drop in zip(bars, drops):
        ax.text(
            bar.get_width() + (1 if drop >= 0 else -1),
            bar.get_y() + bar.get_height() / 2,
            f"{drop:+.1f}%", va="center",
            ha="left" if drop >= 0 else "right", fontsize=9,
        )

    fig.tight_layout()
    if save:
        path = ABLATION_DIR / "component_contribution.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Optional: training-time ablation (slow, more rigorous)
# ─────────────────────────────────────────────────────────────────────────────

def retrain_ablation(condition: AblationCondition, timesteps: int, seed: int) -> ActorCritic:
    """
    Train a fresh policy from scratch under one ablation condition.

    This is the methodologically stronger ablation (the policy actually
    learns under the modified incentives, rather than being evaluated
    out-of-distribution), but is far more expensive — only enabled via
    --retrain since a full run is 7x the cost of normal training.
    """
    from rl.train_ppo import train as train_ppo

    print(f"\n  [retrain] Training under condition: {condition.name} "
          f"({timesteps:,} timesteps) …")

    # train_ppo.train() builds its own AdaptiveLearningEnv() internally
    # with hardcoded defaults, so for retraining ablations we monkeypatch
    # the env factory it uses. This is the only condition-injection point
    # available without modifying rl/train_ppo.py.
    import rl.train_ppo as train_ppo_module

    original_env_cls = train_ppo_module.AdaptiveLearningEnv

    def _patched_env_cls(*args, **kwargs):
        ensemble = SimulatorEnsemble(seed=seed)
        return condition.env_factory(ensemble)

    train_ppo_module.AdaptiveLearningEnv = _patched_env_cls
    try:
        model = train_ppo(total_timesteps=timesteps, seed=seed)
    finally:
        train_ppo_module.AdaptiveLearningEnv = original_env_cls

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(
    checkpoint:         str,
    n_episodes:         int,
    seed:               int,
    deterministic:      bool,
    retrain:            bool,
    retrain_timesteps:  int,
    skip_plots:         bool,
) -> dict:
    print(f"\n{'='*60}")
    print("  Ablation Study — Adaptive Learning PPO Agent")
    print(f"{'='*60}")
    print(f"  Conditions  : {len(CONDITIONS)}")
    print(f"  Episodes    : {n_episodes}")
    print(f"  Mode        : {'retrain (slow)' if retrain else 'fixed-policy eval (fast)'}")
    print(f"  Seed        : {seed}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results: dict[str, dict] = {}

    if retrain:
        # Train a separate small model per condition, then evaluate each
        for condition in CONDITIONS:
            model     = retrain_ablation(condition, retrain_timesteps, seed)
            policy_fn = make_policy_fn(model, deterministic=deterministic)
            results[condition.name] = run_condition(condition, policy_fn, n_episodes, seed)
    else:
        # Fast path: one trained checkpoint, evaluated under each condition
        model     = load_policy(checkpoint, device)
        policy_fn = make_policy_fn(model, deterministic=deterministic)

        print("  Running ablation conditions:\n")
        for condition in CONDITIONS:
            results[condition.name] = run_condition(condition, policy_fn, n_episodes, seed)

    # ── Statistical comparison ────────────────────────────────────────────
    print("\n  Comparing each condition to Full model …")
    comparisons = compare_to_baseline(results)

    # ── Report ─────────────────────────────────────────────────────────────
    print_table(results, comparisons)
    save_table(results, comparisons)

    raw_path = ABLATION_DIR / "ablation_results.json"
    serialisable = {
        name: {
            "description": data["description"],
            "stats":       data["stats"],
            "episodes":    data["episodes"],
        }
        for name, data in results.items()
    }
    with open(raw_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"\n  Raw results saved: {raw_path}")

    # ── Plots ──────────────────────────────────────────────────────────────
    if not skip_plots:
        print("\n  Generating plots …")
        plot_ablation_comparison(results)
        plot_component_contribution(comparisons)

    print("\nAblation study complete ✓")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run an ablation study on the PPO adaptive learning agent."
    )
    parser.add_argument("--checkpoint", type=str, default=str(PPO["checkpoint_path"]),
                         help="Trained PPO checkpoint to evaluate (ignored with --retrain)")
    parser.add_argument("--episodes", type=int, default=EVAL["n_eval_episodes"],
                         help="Episodes per ablation condition")
    parser.add_argument("--seed", type=int, default=EVAL["random_seed"])
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--stochastic", action="store_false", dest="deterministic")
    parser.add_argument("--retrain", action="store_true",
                         help="Train a fresh model per condition instead of reusing one checkpoint")
    parser.add_argument("--retrain-timesteps", type=int, default=50_000,
                         help="Timesteps per condition when --retrain is set")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    run_ablation(
        checkpoint        = args.checkpoint,
        n_episodes        = args.episodes,
        seed              = args.seed,
        deterministic     = args.deterministic,
        retrain           = args.retrain,
        retrain_timesteps = args.retrain_timesteps,
        skip_plots        = args.no_plots,
    )


if __name__ == "__main__":
    main()