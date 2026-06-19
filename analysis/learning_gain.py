"""
analysis/learning_gain.py
Statistical analysis of learning gains for the Adaptive Learning System.

Bridges two data sources:
  1. Real student data  — from the Django DB (PrePostTest + Session models)
  2. Simulated data     — from rl/evaluate.py episode results (results.json)

Metrics computed:
  - Normalised Learning Gain (NLG, Hake 1998):  (post − pre) / (1 − pre)
  - Raw learning gain:                           post − pre
  - Cohen's d effect size                        (vs zero-gain null)
  - Paired t-test: pre vs post scores
  - Wilcoxon signed-rank test (non-parametric)
  - Per-concept mastery gain breakdown
  - Gain vs initial mastery correlation
  - Agent comparison: PPO vs baselines (from evaluate.py results.json)

Output files (all under logs/analysis/):
  learning_gain_report.txt  — human-readable summary (thesis-ready)
  gain_distribution.png     — NLG histogram + KDE
  gain_by_concept.png       — per-concept bar chart
  pre_post_scatter.png      — pre vs post score scatter
  gain_vs_pretest.png       — NLG vs pre-test score (ceiling-effect check)

Usage:
  # Analyse real DB data (requires Django to be set up):
  python analysis/learning_gain.py --source db

  # Analyse simulation results from evaluate.py:
  python analysis/learning_gain.py --source sim

  # Both sources together (recommended for thesis):
  python analysis/learning_gain.py --source both

  # Dry run — compute and print stats, no files written:
  python analysis/learning_gain.py --dry-run
"""

import argparse
import json
import os
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from scipy import stats

# Project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import EVAL, LOG_DIR, NUM_CONCEPTS

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

ANALYSIS_DIR = LOG_DIR / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

SIM_RESULTS_PATH = LOG_DIR / "eval" / "results.json"

# Hake (1998) NLG classification thresholds
NLG_HIGH   = 0.70   # g ≥ 0.70 → high gain
NLG_MEDIUM = 0.30   # 0.30 ≤ g < 0.70 → medium gain
                    # g < 0.30 → low gain

# Significance level
ALPHA = EVAL["alpha"]  # 0.05

# Plot palette (matches evaluate.py AGENT_COLORS)
COLORS = {
    "PPO (ours)":   "#7F77DD",
    "Greedy-KT":    "#5DCAA5",
    "Fixed-linear": "#EF9F27",
    "Random":       "#F0997B",
    "DQN-greedy":   "#85B7EB",
    "db":           "#7F77DD",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GainResult:
    """
    Holds learning gain statistics for one group of students / episodes.

    Fields:
        name         — label (e.g. "PPO (ours)", "Real students")
        n            — sample size
        pre_scores   — array of pre-test / mastery-before values
        post_scores  — array of post-test / mastery-after values
        nlg          — array of normalised learning gains
        raw_gain     — array of raw (post − pre) gains
    """
    name:       str
    pre_scores: np.ndarray
    post_scores: np.ndarray
    nlg:        np.ndarray = field(init=False)
    raw_gain:   np.ndarray = field(init=False)

    def __post_init__(self):
        self.pre_scores  = np.asarray(self.pre_scores,  dtype=float)
        self.post_scores = np.asarray(self.post_scores, dtype=float)
        potential        = 1.0 - self.pre_scores
        # Avoid division by zero for students at ceiling
        self.nlg      = np.where(
            potential > 1e-6,
            (self.post_scores - self.pre_scores) / potential,
            1.0,
        ).clip(0.0, 1.0)
        self.raw_gain = self.post_scores - self.pre_scores

    @property
    def n(self) -> int:
        return len(self.pre_scores)

    @property
    def mean_nlg(self) -> float:
        return float(self.nlg.mean())

    @property
    def mean_raw(self) -> float:
        return float(self.raw_gain.mean())


# ─────────────────────────────────────────────────────────────────────────────
# Statistical helpers
# ─────────────────────────────────────────────────────────────────────────────

def ci95(data: np.ndarray) -> tuple[float, float]:
    """
    Return (mean, half-width of 95% CI) using the t-distribution.
    Works correctly for small samples.
    """
    a  = np.asarray(data, dtype=float)
    n  = len(a)
    if n < 2:
        return float(a.mean()), float("nan")
    se = stats.sem(a)
    t  = stats.t.ppf(0.975, df=n - 1)
    return float(a.mean()), float(t * se)


def cohens_d_one_sample(data: np.ndarray, mu0: float = 0.0) -> float:
    """
    Cohen's d for a one-sample test vs mu0 (default: zero gain).
    d = (mean − mu0) / std
    """
    a = np.asarray(data, dtype=float)
    return float((a.mean() - mu0) / max(a.std(ddof=1), 1e-9))


def cohens_d_two_sample(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cohen's d for independent two-sample comparison.
    Uses pooled standard deviation.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    pooled = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    return float((a.mean() - b.mean()) / max(pooled, 1e-9))


def interpret_d(d: float) -> str:
    """Textual interpretation of Cohen's d magnitude."""
    ad = abs(d)
    if ad < 0.2:   return "negligible"
    if ad < 0.5:   return "small"
    if ad < 0.8:   return "medium"
    return "large"


def hake_category(nlg: float) -> str:
    """Classify mean NLG per Hake (1998)."""
    if nlg >= NLG_HIGH:   return "high gain (g ≥ 0.70)"
    if nlg >= NLG_MEDIUM: return "medium gain (0.30 ≤ g < 0.70)"
    return "low gain (g < 0.30)"


def paired_tests(pre: np.ndarray, post: np.ndarray) -> dict:
    """
    Run paired t-test and Wilcoxon signed-rank test on pre / post scores.

    Returns dict with keys: t, p_ttest, W, p_wilcoxon, significant_ttest,
    significant_wilcoxon.
    """
    t_stat, p_t = stats.ttest_rel(post, pre)
    W_stat, p_w = stats.wilcoxon(post - pre, alternative="greater",
                                  zero_method="wilcox")
    return {
        "t":                    float(t_stat),
        "p_ttest":              float(p_t),
        "W":                    float(W_stat),
        "p_wilcoxon":           float(p_w),
        "significant_ttest":    p_t < ALPHA,
        "significant_wilcoxon": p_w < ALPHA,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_from_db() -> list[GainResult]:
    """
    Load real student pre/post-test data from the Django database.

    Returns a list containing one GainResult for DB students.
    Sets up Django if it hasn't been configured yet.
    """
    # Configure Django if running standalone
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
    try:
        import django
        django.setup()
    except RuntimeError:
        pass  # already configured

    from django.db.models import Q
    from backend.models import PrePostTest

    # Fetch completed pre/post test pairs
    records = PrePostTest.objects.filter(
        post_score__isnull=False
    ).values("pre_score", "post_score", "pre_concept_scores", "post_concept_scores")

    if not records.exists():
        print("  [warn] No completed pre/post test records found in DB.")
        return []

    pre_scores  = np.array([r["pre_score"]  for r in records], dtype=float)
    post_scores = np.array([r["post_score"] for r in records], dtype=float)

    print(f"  Loaded {len(pre_scores)} real student records from DB.")
    return [GainResult(name="Real students", pre_scores=pre_scores, post_scores=post_scores)]


def load_from_sim(path: Path = SIM_RESULTS_PATH) -> list[GainResult]:
    """
    Load episode data from evaluate.py's results.json.

    Each agent becomes one GainResult — mastery_before maps to pre_score,
    mastery_after to post_score.
    """
    if not path.exists():
        print(f"  [warn] Simulation results not found at {path}.")
        print("  Run `python rl/evaluate.py` first.")
        return []

    with open(path) as f:
        raw = json.load(f)

    results = []
    for name, data in raw.items():
        episodes    = data.get("episodes", [])
        pre_scores  = np.array([e["mastery_before"] for e in episodes], dtype=float)
        post_scores = np.array([e["mastery_after"]  for e in episodes], dtype=float)
        results.append(GainResult(name=name, pre_scores=pre_scores, post_scores=post_scores))
        print(f"  Loaded {len(episodes)} episodes for agent '{name}'.")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Per-concept breakdown
# ─────────────────────────────────────────────────────────────────────────────

def compute_concept_gains_db() -> Optional[np.ndarray]:
    """
    Compute per-concept mean gain from DB concept-score dicts.

    Returns array of shape (NUM_CONCEPTS,) or None if data unavailable.
    """
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
    try:
        import django; django.setup()
    except RuntimeError:
        pass

    from backend.models import PrePostTest

    records = PrePostTest.objects.filter(
        post_score__isnull=False,
        post_concept_scores__isnull=False,
    ).values("pre_concept_scores", "post_concept_scores")

    if not records.exists():
        return None

    gains = np.zeros(NUM_CONCEPTS)
    counts = np.zeros(NUM_CONCEPTS)

    for r in records:
        pre  = r["pre_concept_scores"]
        post = r["post_concept_scores"]
        if not (pre and post):
            continue
        for c_id_str, post_val in post.items():
            try:
                c_id = int(c_id_str)
                if 0 <= c_id < NUM_CONCEPTS:
                    pre_val = float(pre.get(c_id_str, 0.0))
                    gains[c_id]  += float(post_val) - pre_val
                    counts[c_id] += 1
            except (ValueError, TypeError):
                continue

    mask = counts > 0
    gains[mask] /= counts[mask]
    return gains


# ─────────────────────────────────────────────────────────────────────────────
# Statistical analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse(result: GainResult) -> dict:
    """
    Compute full statistics for one GainResult.

    Returns dict with:
        n, mean_nlg, ci_nlg, mean_raw, ci_raw,
        cohens_d, d_interpretation, hake_category,
        paired_tests (if pre/post available),
        gain_vs_pretest_r, gain_vs_pretest_p
    """
    mean_nlg, ci_nlg = ci95(result.nlg)
    mean_raw, ci_raw = ci95(result.raw_gain)
    d = cohens_d_one_sample(result.nlg, mu0=0.0)

    # Correlation: NLG vs pre-test (tests for ceiling/floor effects)
    r, p_corr = stats.pearsonr(result.pre_scores, result.nlg) if result.n > 2 else (float("nan"), float("nan"))

    # NLG category counts
    high   = int((result.nlg >= NLG_HIGH).sum())
    medium = int(((result.nlg >= NLG_MEDIUM) & (result.nlg < NLG_HIGH)).sum())
    low    = int((result.nlg < NLG_MEDIUM).sum())

    stats_dict = {
        "n":               result.n,
        "mean_nlg":        mean_nlg,
        "ci_nlg":          ci_nlg,
        "mean_raw_gain":   mean_raw,
        "ci_raw_gain":     ci_raw,
        "std_nlg":         float(result.nlg.std(ddof=1)),
        "median_nlg":      float(np.median(result.nlg)),
        "cohens_d":        d,
        "d_interpretation":interpret_d(d),
        "hake_category":   hake_category(mean_nlg),
        "high_gain_pct":   high   / result.n * 100,
        "medium_gain_pct": medium / result.n * 100,
        "low_gain_pct":    low    / result.n * 100,
        "gain_vs_pretest_r": float(r),
        "gain_vs_pretest_p": float(p_corr),
    }

    # Paired hypothesis tests (pre vs post)
    if result.n >= 5:
        tests = paired_tests(result.pre_scores, result.post_scores)
        stats_dict["paired_tests"] = tests

    return stats_dict


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────

def format_report(results: list[GainResult], stats_by_name: dict) -> str:
    """
    Generate a formatted plain-text report suitable for a thesis appendix.
    """
    lines = []

    lines.append("=" * 70)
    lines.append("  LEARNING GAIN ANALYSIS REPORT")
    lines.append("  Adaptive Learning System — PPO Agent")
    lines.append("=" * 70)
    lines.append("")

    for result in results:
        s = stats_by_name[result.name]
        lines.append(f"── {result.name.upper()} ({'n=' + str(s['n'])})")
        lines.append("")
        lines.append(f"  Normalised Learning Gain (NLG, Hake 1998):")
        lines.append(f"    Mean NLG:    {s['mean_nlg']:.4f}  ± {s['ci_nlg']:.4f}  (95% CI)")
        lines.append(f"    Median NLG:  {s['median_nlg']:.4f}")
        lines.append(f"    Std dev:     {s['std_nlg']:.4f}")
        lines.append(f"    Category:    {s['hake_category']}")
        lines.append("")
        lines.append(f"  Raw learning gain (post − pre):")
        lines.append(f"    Mean:        {s['mean_raw_gain']:+.4f}  ± {s['ci_raw_gain']:.4f}  (95% CI)")
        lines.append("")
        lines.append(f"  Effect size (Cohen's d vs zero gain):")
        lines.append(f"    d = {s['cohens_d']:.4f}  ({s['d_interpretation']} effect)")
        lines.append("")
        lines.append(f"  Gain category breakdown:")
        lines.append(f"    High   (g ≥ 0.70): {s['high_gain_pct']:5.1f}%")
        lines.append(f"    Medium (g ≥ 0.30): {s['medium_gain_pct']:5.1f}%")
        lines.append(f"    Low    (g < 0.30): {s['low_gain_pct']:5.1f}%")
        lines.append("")

        if "paired_tests" in s:
            t = s["paired_tests"]
            sig_t = "✓" if t["significant_ttest"]    else "✗"
            sig_w = "✓" if t["significant_wilcoxon"] else "✗"
            lines.append(f"  Hypothesis test (H₀: no learning gain):")
            lines.append(f"    Paired t-test:           t={t['t']:+.3f}  p={t['p_ttest']:.4f}  {sig_t}")
            lines.append(f"    Wilcoxon signed-rank:    W={t['W']:.1f}   p={t['p_wilcoxon']:.4f}  {sig_w}")
            lines.append(f"    (α = {ALPHA})")
            lines.append("")

        r, p_r = s["gain_vs_pretest_r"], s["gain_vs_pretest_p"]
        lines.append(f"  Ceiling/floor check (NLG vs pre-test score):")
        lines.append(f"    Pearson r = {r:.4f}   p = {p_r:.4f}")
        if not np.isnan(r):
            if r < -0.30 and p_r < ALPHA:
                lines.append("    → Significant negative correlation — ceiling effect present.")
            elif r > 0.30 and p_r < ALPHA:
                lines.append("    → Significant positive correlation — floor effect present.")
            else:
                lines.append("    → No significant ceiling/floor effect detected.")
        lines.append("")

    # ── Cross-agent comparison (if multiple results) ─────────────────────
    if len(results) > 1:
        ppo_result = next((r for r in results if "PPO" in r.name), None)
        if ppo_result:
            lines.append("─" * 70)
            lines.append("  CROSS-AGENT COMPARISON (PPO vs baselines)")
            lines.append("")
            for other in results:
                if other.name == ppo_result.name:
                    continue
                d = cohens_d_two_sample(ppo_result.nlg, other.nlg)
                t_stat, p_val = stats.ttest_ind(
                    ppo_result.nlg, other.nlg, equal_var=False
                )
                sig = "✓ significant" if p_val < ALPHA else "✗ not significant"
                lines.append(f"  PPO vs {other.name}:")
                lines.append(
                    f"    Δ mean NLG = {ppo_result.mean_nlg - other.mean_nlg:+.4f}"
                    f"   t={t_stat:.3f}  p={p_val:.4f}  d={d:.3f}  {sig}"
                )
                lines.append(f"    ({interpret_d(d)} effect, {'PPO better' if d > 0 else 'baseline better'})")
                lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_gain_distribution(results: list[GainResult], save: bool = True):
    """Histogram + KDE of NLG for each agent/group."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, result in zip(axes, results):
        color = COLORS.get(result.name, "#7F77DD")
        data  = result.nlg

        # Histogram
        ax.hist(data, bins=20, color=color, alpha=0.6,
                edgecolor="white", linewidth=0.5, density=True)

        # KDE overlay
        if len(data) > 5:
            kde_x = np.linspace(-0.05, 1.05, 200)
            kde   = stats.gaussian_kde(data)
            ax.plot(kde_x, kde(kde_x), color=color, linewidth=2)

        # Threshold lines
        ax.axvline(result.mean_nlg, color="black", linestyle="--",
                   linewidth=1.5, label=f"Mean = {result.mean_nlg:.3f}")
        ax.axvline(NLG_MEDIUM, color="#999", linestyle=":", linewidth=1)
        ax.axvline(NLG_HIGH,   color="#666", linestyle=":", linewidth=1)

        ax.set_xlabel("Normalised Learning Gain (g)", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(result.name, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_xlim(-0.05, 1.05)
        ax.grid(alpha=0.3)

    fig.suptitle("NLG Distribution per Agent / Group", fontsize=13, y=1.02)
    fig.tight_layout()

    if save:
        path = ANALYSIS_DIR / "gain_distribution.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def plot_pre_post_scatter(results: list[GainResult], save: bool = True):
    """Pre vs post score scatter with identity line."""
    fig, ax = plt.subplots(figsize=(6, 6))

    for result in results:
        color = COLORS.get(result.name, "#7F77DD")
        ax.scatter(
            result.pre_scores, result.post_scores,
            alpha=0.35, s=15, color=color, label=result.name
        )

    # Identity line (y = x, no gain)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="No gain (y=x)")
    ax.set_xlabel("Pre-test score (mastery before)", fontsize=11)
    ax.set_ylabel("Post-test score (mastery after)", fontsize=11)
    ax.set_title("Pre vs Post Score", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save:
        path = ANALYSIS_DIR / "pre_post_scatter.png"
        fig.savefig(path, dpi=150)
        print(f"  Saved: {path}")
    plt.close(fig)


def plot_gain_vs_pretest(results: list[GainResult], save: bool = True):
    """NLG vs pre-test score — checks for ceiling/floor effects."""
    fig, ax = plt.subplots(figsize=(7, 4))

    for result in results:
        color = COLORS.get(result.name, "#7F77DD")
        ax.scatter(result.pre_scores, result.nlg,
                   alpha=0.3, s=12, color=color, label=result.name)

        # Regression line
        if result.n > 5:
            m, b, _, _, _ = stats.linregress(result.pre_scores, result.nlg)
            xs = np.linspace(0, 1, 100)
            ax.plot(xs, m * xs + b, color=color, linewidth=1.5, alpha=0.8)

    ax.axhline(NLG_MEDIUM, color="#999", linestyle=":", linewidth=1, label="Low/medium threshold")
    ax.axhline(NLG_HIGH,   color="#666", linestyle=":", linewidth=1, label="Medium/high threshold")
    ax.set_xlabel("Pre-test score", fontsize=11)
    ax.set_ylabel("Normalised Learning Gain (g)", fontsize=11)
    ax.set_title("Gain vs Pre-test Score (Ceiling/Floor Check)", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save:
        path = ANALYSIS_DIR / "gain_vs_pretest.png"
        fig.savefig(path, dpi=150)
        print(f"  Saved: {path}")
    plt.close(fig)


def plot_gain_by_concept(concept_gains: np.ndarray, save: bool = True):
    """
    Horizontal bar chart of per-concept mean gain.
    Only shown when concept-level data is available from the DB.
    """
    if concept_gains is None:
        return

    # Show top-20 and bottom-20 concepts by gain
    top_n = 20
    sorted_idx = np.argsort(concept_gains)
    top_idx    = sorted_idx[-top_n:][::-1]
    bottom_idx = sorted_idx[:top_n]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    for ax, idx, title in [
        (ax1, top_idx,    f"Top {top_n} concepts by gain"),
        (ax2, bottom_idx, f"Bottom {top_n} concepts by gain"),
    ]:
        gains  = concept_gains[idx]
        colors = ["#7F77DD" if g >= 0 else "#F0997B" for g in gains]
        labels = [f"Concept {i}" for i in idx]

        ax.barh(range(len(idx)), gains, color=colors, alpha=0.85)
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Mean gain (post − pre mastery)", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Per-Concept Learning Gain", fontsize=13)
    fig.tight_layout()

    if save:
        path = ANALYSIS_DIR / "gain_by_concept.png"
        fig.savefig(path, dpi=150)
        print(f"  Saved: {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(
    source:  str  = "both",
    dry_run: bool = False,
) -> None:
    """
    Full analysis pipeline.

    Args:
        source:  "db", "sim", or "both"
        dry_run: compute + print, but write no files
    """
    print("\n" + "=" * 60)
    print("  Learning Gain Analysis")
    print("=" * 60)

    results: list[GainResult] = []

    if source in ("db", "both"):
        print("\n[DB] Loading real student data …")
        results.extend(load_from_db())

    if source in ("sim", "both"):
        print("\n[SIM] Loading simulation episode data …")
        results.extend(load_from_sim())

    if not results:
        print("\nNo data available. Exiting.")
        return

    # Compute statistics for each result
    stats_by_name: dict[str, dict] = {}
    for result in results:
        print(f"\nAnalysing '{result.name}' (n={result.n}) …")
        stats_by_name[result.name] = analyse(result)

    # Generate report
    report = format_report(results, stats_by_name)
    print("\n" + report)

    if dry_run:
        print("Dry run — no files written.")
        return

    # Write report
    report_path = ANALYSIS_DIR / "learning_gain_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved: {report_path}")

    # Plots
    print("\nGenerating plots …")
    plot_gain_distribution(results)
    plot_pre_post_scatter(results)
    plot_gain_vs_pretest(results)

    # Per-concept plot (DB only)
    if source in ("db", "both"):
        print("  Computing per-concept gains …")
        concept_gains = compute_concept_gains_db()
        plot_gain_by_concept(concept_gains)

    print("\nAnalysis complete ✓")
    print(f"Outputs written to: {ANALYSIS_DIR}/")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute learning gain statistics for the Adaptive Learning System."
    )
    parser.add_argument(
        "--source",
        choices=["db", "sim", "both"],
        default="sim",
        help=(
            "Data source: 'db' = Django DB (real students), "
            "'sim' = evaluate.py results.json, "
            "'both' = both sources (default: sim)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print stats but write no output files",
    )
    args = parser.parse_args()
    run_analysis(source=args.source, dry_run=args.dry_run)


if __name__ == "__main__":
    main()