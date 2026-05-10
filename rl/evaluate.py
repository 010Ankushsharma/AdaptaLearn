"""
rl/evaluate.py
Evaluation suite for the trained PPO agent.

Compares PPO against four baselines over N evaluation episodes:

  1. Random          — picks any ZPD-valid action uniformly at random
  2. Fixed-linear    — always picks the next concept in sequential order
  3. Greedy-KT       — always picks the concept with lowest P(mastery)
                       (myopic, no long-term planning)
  4. DQN-equivalent  — greedy argmax policy (no exploration), same network
  5. PPO (ours)      — stochastic trained policy / deterministic eval

Metrics reported per agent:
  - Mean episode reward        (primary RL metric)
  - Mean normalised learning gain  (ΔMastery / potential gain)
  - Mean questions to mastery  (efficiency)
  - ZPD adherence %            (pedagogical quality)
  - Mean episode length
  - 95% confidence intervals   (for statistical significance)

Output files (in logs/eval/):
  results.json          — raw per-episode data for all agents
  summary_table.csv     — means + CIs ready for thesis table
  reward_curves.png     — reward distribution comparison plot
  mastery_curves.png    — mean mastery over steps per agent
  zpd_adherence.png     — ZPD adherence per agent bar chart

Usage:
    python rl/evaluate.py
    python rl/evaluate.py --episodes 200 --deterministic
    python rl/evaluate.py --checkpoint checkpoints/ppo_best.zip
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ENV, EVAL, LOG_DIR, NUM_CONCEPTS, NUM_QUESTIONS, PPO
from rl.env import AdaptiveLearningEnv
from rl.student_simulator import SimulatorEnsemble

# Output directory for evaluation artefacts
EVAL_DIR = LOG_DIR / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# Plot style
AGENT_COLORS = {
    "PPO (ours)":   "#7F77DD",
    "Greedy-KT":    "#5DCAA5",
    "Fixed-linear": "#EF9F27",
    "Random":       "#F0997B",
    "DQN-greedy":   "#85B7EB",
}


# ─────────────────────────────────────────────────────────────────────────────
# Policy interfaces (all return int action given env state)
# ─────────────────────────────────────────────────────────────────────────────

def random_policy(env: AdaptiveLearningEnv, obs: np.ndarray) -> int:
    """Uniform random over ZPD-valid actions."""
    mask  = env.action_masks()
    valid = np.where(mask)[0]
    return int(np.random.choice(valid))


def fixed_linear_policy(env: AdaptiveLearningEnv, obs: np.ndarray) -> int:
    """
    Sequential curriculum: cycle through concepts in order 0,1,2,...
    For each concept, pick the first valid question that tests it.
    Falls back to random if no valid question for this concept.
    """
    mask = env.action_masks()
    # Determine current concept from step count
    n_q      = env._n_questions
    concept  = n_q % NUM_CONCEPTS
    # Find questions for this concept that are in ZPD
    for q_id in env.pool.questions_for_concept(concept):
        if mask[q_id]:
            return int(q_id)
    # Fallback: any valid action
    valid = np.where(mask)[0]
    return int(np.random.choice(valid)) if len(valid) > 0 else 0


def greedy_kt_policy(env: AdaptiveLearningEnv, obs: np.ndarray) -> int:
    """
    Greedy knowledge tracing: always target the concept with lowest mastery.
    Myopic — maximises immediate mastery gain with no lookahead.
    """
    mask    = env.action_masks()
    mastery = env.get_mastery_vector()             # (N,)

    # Find concept with lowest mastery that has a valid ZPD question
    concept_order = np.argsort(mastery)            # ascending — lowest first
    for concept in concept_order:
        questions = env.pool.questions_for_concept(concept)
        valid_qs  = [q for q in questions if mask[q]]
        if valid_qs:
            # Among valid questions for this concept, pick lowest difficulty
            diffs   = [env.pool.get_difficulty(q) for q in valid_qs]
            best_q  = valid_qs[int(np.argmin(diffs))]
            return int(best_q)

    # Fallback
    valid = np.where(mask)[0]
    return int(np.random.choice(valid)) if len(valid) > 0 else 0


def make_ppo_policy(model, deterministic: bool = True) -> Callable:
    """
    Return a policy function from a trained ActorCritic model.

    Args:
        model:        trained ActorCritic
        deterministic: True = argmax (eval mode), False = sample (stochastic)
    """
    import torch

    def policy(env: AdaptiveLearningEnv, obs: np.ndarray) -> int:
        model.eval()
        with torch.no_grad():
            state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            mask  = torch.tensor(
                env.action_masks(), dtype=torch.bool
            ).unsqueeze(0)
            out   = model(state, mask, deterministic=deterministic)
            return int(out["action"].item())

    return policy


def make_dqn_greedy_policy(model) -> Callable:
    """
    Greedy argmax of actor logits — same network as PPO but no exploration.
    Shows the value of stochastic vs deterministic evaluation.
    """
    return make_ppo_policy(model, deterministic=True)


# ─────────────────────────────────────────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(
    env:    AdaptiveLearningEnv,
    policy: Callable,
    seed:   Optional[int] = None,
) -> dict:
    """
    Run one full episode with a given policy.

    Returns dict with:
        total_reward        : float
        mastery_before      : float  — mean mastery at episode start
        mastery_after       : float  — mean mastery at episode end
        learning_gain       : float  — normalised (after - before) / (1 - before)
        n_questions         : int    — episode length
        zpd_adherence       : float  — fraction of steps within ZPD
        mastery_trajectory  : list   — mean mastery after each step
        rewards             : list   — reward at each step
    """
    obs, info = env.reset(seed=seed)

    mastery_before  = info["mastery_mean"]
    total_reward    = 0.0
    zpd_steps       = 0
    mastery_traj    = [mastery_before]
    rewards_traj    = []
    done            = False

    while not done:
        action = policy(env, obs)

        # Check ZPD adherence before stepping
        mask = env.action_masks()
        if mask[action]:
            zpd_steps += 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        mastery_traj.append(info["mastery_mean"])
        rewards_traj.append(reward)
        done = terminated or truncated

    mastery_after = info["mastery_mean"]
    n_q           = info["n_questions"]

    # Normalised learning gain: (after - before) / (1 - before)
    # Accounts for ceiling effects — a student at 0.9 can gain less than one at 0.1
    potential = 1.0 - mastery_before
    norm_gain = (mastery_after - mastery_before) / max(potential, 1e-6)
    norm_gain = float(np.clip(norm_gain, 0.0, 1.0))

    return {
        "total_reward":       total_reward,
        "mastery_before":     mastery_before,
        "mastery_after":      mastery_after,
        "learning_gain":      norm_gain,
        "n_questions":        n_q,
        "zpd_adherence":      zpd_steps / max(n_q, 1),
        "mastery_trajectory": mastery_traj,
        "rewards":            rewards_traj,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Multi-episode evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_agent(
    name:       str,
    policy:     Callable,
    n_episodes: int = EVAL["n_eval_episodes"],
    seed:       int = EVAL["random_seed"],
) -> dict:
    """
    Run n_episodes with the given policy and collect metrics.
    Uses a fresh SimulatorEnsemble so all agents see the same students.

    Returns dict with per-episode lists and aggregate stats.
    """
    print(f"  Evaluating [{name}] over {n_episodes} episodes ...")

    ensemble = SimulatorEnsemble(seed=seed)
    env      = AdaptiveLearningEnv(
        ensemble          = ensemble,
        use_fresh_student = True,
    )
    rng = np.random.default_rng(seed)

    episodes = []
    for ep in range(n_episodes):
        ep_seed = int(rng.integers(0, 100_000))
        result  = run_episode(env, policy, seed=ep_seed)
        episodes.append(result)

    # Aggregate
    rewards      = [e["total_reward"]   for e in episodes]
    gains        = [e["learning_gain"]  for e in episodes]
    n_questions  = [e["n_questions"]    for e in episodes]
    zpd          = [e["zpd_adherence"]  for e in episodes]

    def ci95(data):
        """95% confidence interval using t-distribution."""
        a = np.array(data)
        n = len(a)
        se = stats.sem(a)
        t  = stats.t.ppf(0.975, df=n - 1)
        return float(a.mean()), float(t * se)

    mean_r, ci_r   = ci95(rewards)
    mean_g, ci_g   = ci95(gains)
    mean_q, ci_q   = ci95(n_questions)
    mean_z, ci_z   = ci95(zpd)

    print(f"    reward:        {mean_r:+.3f} ± {ci_r:.3f}")
    print(f"    learning gain: {mean_g:.4f} ± {ci_g:.4f}")
    print(f"    ep length:     {mean_q:.1f} ± {ci_q:.1f}")
    print(f"    ZPD adherence: {mean_z:.3f} ± {ci_z:.3f}")

    return {
        "name":     name,
        "episodes": episodes,
        "stats": {
            "mean_reward":        mean_r,  "ci_reward":        ci_r,
            "mean_learning_gain": mean_g,  "ci_learning_gain": ci_g,
            "mean_n_questions":   mean_q,  "ci_n_questions":   ci_q,
            "mean_zpd":           mean_z,  "ci_zpd":           ci_z,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Statistical tests
# ─────────────────────────────────────────────────────────────────────────────

def run_statistical_tests(results: dict[str, dict], alpha: float = EVAL["alpha"]) -> dict:
    """
    Run two-sample Welch t-tests comparing PPO against each baseline.
    Reports t-statistic, p-value, Cohen's d effect size, and significance.

    Cohen's d interpretation:
      0.2 = small, 0.5 = medium, 0.8 = large
    """
    print("\n  Statistical tests (PPO vs baselines):")
    print(f"  α = {alpha}\n")

    ppo_rewards = [e["total_reward"] for e in results["PPO (ours)"]["episodes"]]
    ppo_gains   = [e["learning_gain"] for e in results["PPO (ours)"]["episodes"]]

    tests = {}
    for name, data in results.items():
        if name == "PPO (ours)":
            continue

        base_rewards = [e["total_reward"] for e in data["episodes"]]
        base_gains   = [e["learning_gain"] for e in data["episodes"]]

        # Welch t-test (does not assume equal variances)
        t_r, p_r = stats.ttest_ind(ppo_rewards, base_rewards, equal_var=False)
        t_g, p_g = stats.ttest_ind(ppo_gains,   base_gains,   equal_var=False)

        # Cohen's d = (mean1 - mean2) / pooled_std
        def cohens_d(a, b):
            a, b = np.array(a), np.array(b)
            pooled_std = np.sqrt((a.std()**2 + b.std()**2) / 2)
            return float((a.mean() - b.mean()) / max(pooled_std, 1e-9))

        d_r = cohens_d(ppo_rewards, base_rewards)
        d_g = cohens_d(ppo_gains,   base_gains)

        sig_r = "✓ significant" if p_r < alpha else "✗ not significant"
        sig_g = "✓ significant" if p_g < alpha else "✗ not significant"

        tests[name] = {
            "reward": {"t": t_r, "p": p_r, "cohens_d": d_r, "significant": p_r < alpha},
            "gain":   {"t": t_g, "p": p_g, "cohens_d": d_g, "significant": p_g < alpha},
        }

        print(f"  PPO vs {name}:")
        print(f"    Reward : t={t_r:.3f}  p={p_r:.4f}  d={d_r:.3f}  {sig_r}")
        print(f"    Gain   : t={t_g:.3f}  p={p_g:.4f}  d={d_g:.3f}  {sig_g}")

    return tests


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_reward_distributions(results: dict, save: bool = True):
    """Box + strip plot of episode rewards per agent."""
    fig, ax = plt.subplots(figsize=(10, 5))

    names   = list(results.keys())
    data    = [[e["total_reward"] for e in results[n]["episodes"]] for n in names]
    colors  = [AGENT_COLORS.get(n, "#AAAAAA") for n in names]

    bp = ax.boxplot(data, patch_artist=True, notch=True,
                    medianprops=dict(color="white", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    # Overlay individual points
    for i, (d, color) in enumerate(zip(data, colors), start=1):
        jitter = np.random.default_rng(i).uniform(-0.15, 0.15, len(d))
        ax.scatter([i + j for j in jitter], d,
                   alpha=0.3, s=12, color=color, zorder=3)

    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Episode Total Reward", fontsize=11)
    ax.set_title("PPO vs Baselines — Episode Reward Distribution", fontsize=12)
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save:
        path = EVAL_DIR / "reward_curves.png"
        fig.savefig(path, dpi=150)
        print(f"  Saved: {path}")
    plt.close(fig)


def plot_mastery_curves(results: dict, save: bool = True):
    """Mean mastery trajectory over episode steps, per agent."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for name, data in results.items():
        trajs = [e["mastery_trajectory"] for e in data["episodes"]]
        # Pad to same length
        max_len = max(len(t) for t in trajs)
        padded  = np.array([
            t + [t[-1]] * (max_len - len(t)) for t in trajs
        ])
        mean = padded.mean(axis=0)
        se   = padded.std(axis=0) / np.sqrt(len(trajs))

        xs = range(len(mean))
        color = AGENT_COLORS.get(name, "#AAAAAA")
        ax.plot(xs, mean, label=name, color=color, linewidth=2)
        ax.fill_between(xs, mean - se, mean + se, alpha=0.15, color=color)

    ax.axhline(ENV["mastery_threshold"], color="red", linestyle="--",
               linewidth=1, alpha=0.7, label=f"Mastery threshold ({ENV['mastery_threshold']})")
    ax.set_xlabel("Question step", fontsize=11)
    ax.set_ylabel("Mean mastery P(mastered)", fontsize=11)
    ax.set_title("Mean Mastery Trajectory per Agent", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save:
        path = EVAL_DIR / "mastery_curves.png"
        fig.savefig(path, dpi=150)
        print(f"  Saved: {path}")
    plt.close(fig)


def plot_zpd_adherence(results: dict, save: bool = True):
    """Bar chart of mean ZPD adherence per agent."""
    fig, ax = plt.subplots(figsize=(8, 4))

    names  = list(results.keys())
    means  = [results[n]["stats"]["mean_zpd"]   for n in names]
    cis    = [results[n]["stats"]["ci_zpd"]      for n in names]
    colors = [AGENT_COLORS.get(n, "#AAAAAA")    for n in names]

    bars = ax.bar(names, means, color=colors, alpha=0.85,
                  yerr=cis, capsize=5, error_kw=dict(linewidth=1.5))
    ax.set_ylabel("ZPD Adherence (fraction of steps)", fontsize=11)
    ax.set_title("Zone of Proximal Development Adherence per Agent", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.75, color="green", linestyle="--", linewidth=1,
               alpha=0.6, label="Target: 75%")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.02,
                f"{mean:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()

    if save:
        path = EVAL_DIR / "zpd_adherence.png"
        fig.savefig(path, dpi=150)
        print(f"  Saved: {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def save_summary_table(results: dict, tests: dict) -> Path:
    """
    Write a CSV summary table — paste directly into thesis.

    Columns:
        Agent | Mean Reward ± CI | Learning Gain ± CI |
        Ep Length ± CI | ZPD % ± CI | vs PPO p-value | Cohen's d
    """
    import csv

    path = EVAL_DIR / "summary_table.csv"
    rows = []

    for name, data in results.items():
        s = data["stats"]
        row = {
            "Agent":           name,
            "Mean Reward":     f"{s['mean_reward']:+.3f}",
            "CI Reward":       f"±{s['ci_reward']:.3f}",
            "Learning Gain":   f"{s['mean_learning_gain']:.4f}",
            "CI Gain":         f"±{s['ci_learning_gain']:.4f}",
            "Ep Length":       f"{s['mean_n_questions']:.1f}",
            "CI Ep Length":    f"±{s['ci_n_questions']:.1f}",
            "ZPD %":           f"{s['mean_zpd']*100:.1f}%",
            "CI ZPD":          f"±{s['ci_zpd']*100:.1f}%",
            "vs PPO p (reward)": "—",
            "Cohen's d":         "—",
        }
        if name in tests:
            row["vs PPO p (reward)"] = f"{tests[name]['reward']['p']:.4f}"
            row["Cohen's d"]         = f"{tests[name]['reward']['cohens_d']:.3f}"
        rows.append(row)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Saved summary table: {path}")
    return path


def print_summary_table(results: dict, tests: dict):
    """Print formatted summary table to console."""
    print(f"\n{'─'*85}")
    print(f"  {'Agent':<18} {'Reward':>10} {'Gain':>10} "
          f"{'EpLen':>7} {'ZPD%':>7} {'p-val':>8} {'d':>6}")
    print(f"{'─'*85}")
    for name, data in results.items():
        s    = data["stats"]
        pval = tests.get(name, {}).get("reward", {}).get("p", float("nan"))
        d    = tests.get(name, {}).get("reward", {}).get("cohens_d", float("nan"))
        pstr = f"{pval:.4f}" if not np.isnan(pval) else "  —   "
        dstr = f"{d:.3f}"   if not np.isnan(d)    else "  —  "
        print(
            f"  {name:<18} "
            f"{s['mean_reward']:>+7.3f}±{s['ci_reward']:.3f}  "
            f"{s['mean_learning_gain']:>6.4f}±{s['ci_learning_gain']:.4f}  "
            f"{s['mean_n_questions']:>5.1f}  "
            f"{s['mean_zpd']*100:>5.1f}%  "
            f"{pstr:>8}  {dstr:>6}"
        )
    print(f"{'─'*85}")


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation runner
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    checkpoint:   Optional[str] = None,
    n_episodes:   int           = EVAL["n_eval_episodes"],
    deterministic: bool         = True,
    seed:          int          = EVAL["random_seed"],
    skip_plots:    bool         = False,
) -> dict:
    """
    Run full evaluation suite. Returns results dict.
    """
    print(f"\n{'='*55}")
    print(f"  Evaluation Suite — Adaptive Learning Agent")
    print(f"{'='*55}")
    print(f"  Episodes per agent : {n_episodes}")
    print(f"  Deterministic PPO  : {deterministic}")
    print(f"  Seed               : {seed}\n")

    # ── Load PPO model (if checkpoint exists) ─────────────────────────
    ppo_policy = None
    if checkpoint and Path(checkpoint).exists():
        import torch
        from rl.actor_critic import ActorCritic
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = ActorCritic(
            state_dim  = ENV["state_dim"],
            action_dim = NUM_QUESTIONS,
        ).to(device)
        ckpt = torch.load(checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"  Loaded PPO checkpoint: {checkpoint}")
        print(f"  (checkpoint reward: {ckpt.get('mean_reward', 'N/A')})\n")
        ppo_policy     = make_ppo_policy(model, deterministic=deterministic)
        dqn_policy     = make_dqn_greedy_policy(model)
    else:
        print("  [warn] No checkpoint — using random policy as PPO stand-in")
        ppo_policy = random_policy
        dqn_policy = random_policy

    # ── Define agents ─────────────────────────────────────────────────
    agents = {
        "PPO (ours)":   ppo_policy,
        "Greedy-KT":    greedy_kt_policy,
        "Fixed-linear": fixed_linear_policy,
        "Random":       random_policy,
        "DQN-greedy":   dqn_policy,
    }

    # ── Run evaluation ────────────────────────────────────────────────
    print("  Running evaluations:\n")
    results = {}
    for name, policy in agents.items():
        results[name] = evaluate_agent(name, policy, n_episodes, seed)
        print()

    # ── Statistical tests ─────────────────────────────────────────────
    tests = run_statistical_tests(results, alpha=EVAL["alpha"])

    # ── Summary table ─────────────────────────────────────────────────
    print_summary_table(results, tests)
    save_summary_table(results, tests)

    # ── Save raw results ──────────────────────────────────────────────
    raw_path = EVAL_DIR / "results.json"
    serialisable = {
        name: {
            "stats":    data["stats"],
            "episodes": [
                {k: v for k, v in ep.items() if k != "mastery_trajectory"}
                for ep in data["episodes"]
            ],
        }
        for name, data in results.items()
    }
    with open(raw_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"\n  Raw results saved: {raw_path}")

    # ── Plots ─────────────────────────────────────────────────────────
    if not skip_plots:
        print("\n  Generating plots ...")
        plot_reward_distributions(results)
        plot_mastery_curves(results)
        plot_zpd_adherence(results)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate PPO adaptive learning agent")
    parser.add_argument("--checkpoint",   type=str,  default=str(PPO["checkpoint_path"]))
    parser.add_argument("--episodes",     type=int,  default=EVAL["n_eval_episodes"])
    parser.add_argument("--seed",         type=int,  default=EVAL["random_seed"])
    parser.add_argument("--deterministic",action="store_true", default=True)
    parser.add_argument("--stochastic",   action="store_false", dest="deterministic")
    parser.add_argument("--no-plots",     action="store_true")
    args = parser.parse_args()

    evaluate(
        checkpoint    = args.checkpoint,
        n_episodes    = args.episodes,
        deterministic = args.deterministic,
        seed          = args.seed,
        skip_plots    = args.no_plots,
    )


if __name__ == "__main__":
    main()