"""
rl/reward.py
Reward shaping for the Adaptive Learning RL environment.

Reward function design philosophy:
  - Primary signal   : Δmastery (long-term knowledge gain)
  - Secondary signal : step correctness (dense feedback)
  - Penalties        : hint usage, time overrun (efficiency)
  - Episode bonus    : normalised post-session mastery gain

This decomposition ensures the agent cares primarily about learning
rather than just collecting correct answers (which would bias it toward
showing easy questions). The hint penalty stops the agent from giving
away answers. The time penalty encourages efficiency.

Reward at step t:
  r_t = α·ΔMastery_t + β·correct_t − γ·hints_t − δ·time_overrun_t

Episode-end bonus (added to final step):
  R_T += λ · (mean_mastery_after − mean_mastery_before)

All weights are configurable from config.REWARD.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import REWARD


class RewardShaper:
    """
    Computes shaped rewards for every step and the episode-end bonus.

    All weights loaded from config.REWARD:
      alpha (α) : mastery gain weight         (primary — keep highest)
      beta  (β) : correctness bonus weight    (secondary)
      gamma (γ) : hint penalty weight
      delta (δ) : time overrun penalty weight
      lam   (λ) : episode-end bonus weight
      time_limit: seconds per question before overrun kicks in

    Usage:
        shaper = RewardShaper()

        # Each step
        r = shaper.step_reward(
            mastery_before, mastery_after, correct, hint_count, elapsed_time
        )

        # End of episode
        r += shaper.episode_bonus(mastery_gain)
    """

    def __init__(
        self,
        alpha:      float = REWARD["alpha"],
        beta:       float = REWARD["beta"],
        gamma:      float = REWARD["gamma"],
        delta:      float = REWARD["delta"],
        lam:        float = REWARD["lam"],
        time_limit: float = REWARD["time_limit_per_question"],
    ):
        self.alpha      = alpha
        self.beta       = beta
        self.gamma      = gamma
        self.delta      = delta
        self.lam        = lam
        self.time_limit = time_limit

        # Running stats for reward normalisation (Welford online algorithm)
        self._n     = 0
        self._mean  = 0.0
        self._M2    = 0.0

    # ── Step reward ───────────────────────────────────────────────────────────

    def step_reward(
        self,
        mastery_before: np.ndarray,   # (N,) float32 — mastery before this step
        mastery_after:  np.ndarray,   # (N,) float32 — mastery after this step
        correct:        int,          # 0 or 1
        hint_count:     int,          # number of hints used (0 or 1)
        elapsed_time:   float,        # seconds taken to answer
    ) -> float:
        """
        Compute the shaped reward for one interaction step.

        Components:
          mastery_gain  : mean Δ across all concepts  (α-weighted)
          correct_bonus : 1 if correct, 0 if wrong    (β-weighted)
          hint_penalty  : −hint_count                 (γ-weighted)
          time_penalty  : −max(0, elapsed − limit)/60 (δ-weighted)

        Returns:
            r_t : float reward for this step
        """
        # ── Mastery gain ──────────────────────────────────────────────
        # Mean mastery gain across all concepts
        # Clamp to [-0.5, 0.5] to prevent outliers destabilising training
        delta_mastery = float(np.mean(mastery_after - mastery_before))
        delta_mastery = np.clip(delta_mastery, -0.5, 0.5)

        mastery_reward = self.alpha * delta_mastery

        # ── Correctness bonus ─────────────────────────────────────────
        correct_reward = self.beta * float(correct)

        # ── Hint penalty ──────────────────────────────────────────────
        hint_penalty = self.gamma * float(hint_count)

        # ── Time overrun penalty ──────────────────────────────────────
        # Penalty for every 10 seconds over the time limit
        # Normalised to same scale as other components
        overrun_s    = max(0.0, elapsed_time - self.time_limit)
        time_penalty = self.delta * (overrun_s / 10.0)

        # ── Total step reward ─────────────────────────────────────────
        r = mastery_reward + correct_reward - hint_penalty - time_penalty

        # Update running stats (for logging / analysis)
        self._update_stats(r)

        return float(r)

    # ── Episode bonus ─────────────────────────────────────────────────────────

    def episode_bonus(self, mastery_gain: float) -> float:
        """
        Bonus added at the end of the episode.

        mastery_gain = mean_mastery_after_session − mean_mastery_before_session

        Clipped to [0, 1] — we only reward positive gains, not penalise
        negative (which shouldn't happen but protects against edge cases).

        Returns:
            bonus : float  (λ · mastery_gain)
        """
        gain  = float(np.clip(mastery_gain, 0.0, 1.0))
        bonus = self.lam * gain
        self._update_stats(bonus)
        return bonus

    # ── Normalisation ─────────────────────────────────────────────────────────

    def normalise(self, r: float, eps: float = 1e-8) -> float:
        """
        Normalise reward using running mean and std (Welford's algorithm).
        Call this in the training loop if reward scales vary across
        simulator configurations.
        """
        if self._n < 2:
            return r
        std = max(np.sqrt(self._M2 / (self._n - 1)), eps)
        return (r - self._mean) / std

    def _update_stats(self, r: float) -> None:
        """Welford online mean/variance update."""
        self._n    += 1
        delta       = r - self._mean
        self._mean += delta / self._n
        delta2      = r - self._mean
        self._M2   += delta * delta2

    @property
    def running_mean(self) -> float:
        return self._mean

    @property
    def running_std(self) -> float:
        if self._n < 2:
            return 1.0
        return float(np.sqrt(self._M2 / (self._n - 1)))

    # ── Decomposition (for logging) ────────────────────────────────────────────

    def decompose(
        self,
        mastery_before: np.ndarray,
        mastery_after:  np.ndarray,
        correct:        int,
        hint_count:     int,
        elapsed_time:   float,
    ) -> dict:
        """
        Return each reward component separately — useful for reward analysis
        and verifying the agent isn't gaming one component.
        """
        delta_mastery = float(np.clip(
            np.mean(mastery_after - mastery_before), -0.5, 0.5
        ))
        overrun_s = max(0.0, elapsed_time - self.time_limit)

        return {
            "mastery_component":    self.alpha  * delta_mastery,
            "correct_component":    self.beta   * float(correct),
            "hint_penalty":        -self.gamma  * float(hint_count),
            "time_penalty":        -self.delta  * (overrun_s / 10.0),
            "delta_mastery":        delta_mastery,
            "total":               (self.alpha * delta_mastery
                                    + self.beta * float(correct)
                                    - self.gamma * float(hint_count)
                                    - self.delta * (overrun_s / 10.0)),
        }

    def __repr__(self) -> str:
        return (
            f"RewardShaper(α={self.alpha}, β={self.beta}, "
            f"γ={self.gamma}, δ={self.delta}, λ={self.lam}, "
            f"n_steps={self._n}, mean={self._mean:.4f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== RewardShaper smoke test ===\n")

    import numpy as np

    N = 10   # num concepts
    shaper = RewardShaper()

    print(f"Config: α={shaper.alpha}, β={shaper.beta}, "
          f"γ={shaper.gamma}, δ={shaper.delta}, λ={shaper.lam}")
    print(f"Time limit: {shaper.time_limit}s\n")

    # ── Test 1: correct answer, mastery gained, no hints, on time ─────
    mb = np.full(N, 0.40, dtype=np.float32)
    ma = np.full(N, 0.45, dtype=np.float32)   # +0.05 mastery gain
    r1 = shaper.step_reward(mb, ma, correct=1, hint_count=0, elapsed_time=30.0)
    expected_mastery = shaper.alpha * 0.05
    expected_correct = shaper.beta  * 1.0
    expected_total   = expected_mastery + expected_correct
    print(f"Test 1 — correct, no hints, on time:")
    print(f"  reward={r1:.4f}  expected≈{expected_total:.4f}")
    assert abs(r1 - expected_total) < 1e-4, f"Mismatch: {r1} vs {expected_total}"
    print("  ✓\n")

    # ── Test 2: wrong answer, no mastery gain, hint used ─────────────
    mb = np.full(N, 0.40, dtype=np.float32)
    ma = np.full(N, 0.40, dtype=np.float32)   # no mastery change
    r2 = shaper.step_reward(mb, ma, correct=0, hint_count=1, elapsed_time=40.0)
    expected = -shaper.gamma * 1.0             # only hint penalty
    print(f"Test 2 — wrong, hint used, on time:")
    print(f"  reward={r2:.4f}  expected≈{expected:.4f}")
    assert abs(r2 - expected) < 1e-4, f"Mismatch: {r2} vs {expected}"
    print("  ✓\n")

    # ── Test 3: time overrun penalty ──────────────────────────────────
    mb = np.full(N, 0.50, dtype=np.float32)
    ma = np.full(N, 0.50, dtype=np.float32)
    overrun = 30.0   # 30s over limit → penalty = δ * 3.0
    r3 = shaper.step_reward(mb, ma, correct=0, hint_count=0,
                             elapsed_time=shaper.time_limit + overrun)
    expected = -shaper.delta * (overrun / 10.0)
    print(f"Test 3 — time overrun ({overrun}s):")
    print(f"  reward={r3:.4f}  expected≈{expected:.4f}")
    assert abs(r3 - expected) < 1e-4, f"Mismatch: {r3} vs {expected}"
    print("  ✓\n")

    # ── Test 4: episode bonus ─────────────────────────────────────────
    b1 = shaper.episode_bonus(mastery_gain=0.30)
    b2 = shaper.episode_bonus(mastery_gain=-0.05)  # negative → clipped to 0
    b3 = shaper.episode_bonus(mastery_gain=1.50)   # > 1 → clipped to 1
    assert b1 == shaper.lam * 0.30,    f"Bonus mismatch: {b1}"
    assert b2 == 0.0,                  f"Negative gain should give 0 bonus: {b2}"
    assert b3 == shaper.lam * 1.0,     f"Clamped bonus mismatch: {b3}"
    print(f"Test 4 — episode bonus:")
    print(f"  gain=0.30 → {b1:.3f}  gain=-0.05 → {b2:.3f}  gain=1.5 → {b3:.3f}")
    print("  ✓\n")

    # ── Test 5: α > β ensures large mastery gain dominates correctness ──
    # A mastery gain of 0.40 gives α·0.40 = 0.40 > β·1.0 = 0.30
    # (This is intentional: single-step gains are usually small; the config
    #  ensures sustained mastery growth is rewarded more than lucky guesses)
    mb_high = np.full(N, 0.20, dtype=np.float32)
    ma_high = np.full(N, 0.60, dtype=np.float32)   # +0.40 mastery gain per concept
    r_mastery = shaper.step_reward(mb_high, ma_high, correct=0, hint_count=0,
                                    elapsed_time=20.0)
    r_correct  = shaper.step_reward(
        np.full(N, 0.50), np.full(N, 0.50), correct=1, hint_count=0, elapsed_time=20.0
    )
    print(f"Test 5 — large mastery gain (0.40) dominates correctness bonus (α > β):")
    print(f"  mastery_gain_r={r_mastery:.4f}  correct_only_r={r_correct:.4f}")
    assert r_mastery > r_correct, (
        f"α·ΔMastery ({r_mastery:.4f}) should exceed β·correct ({r_correct:.4f}) "
        f"for gains > β/α = {shaper.beta/shaper.alpha:.2f}"
    )
    print("  ✓\n")

    # ── Test 6: decompose() ────────────────────────────────────────────
    mb = np.full(N, 0.40, np.float32)
    ma = np.full(N, 0.46, np.float32)
    parts = shaper.decompose(mb, ma, correct=1, hint_count=1, elapsed_time=150.0)
    recon = parts["total"]
    direct = shaper.step_reward(mb, ma, correct=1, hint_count=1, elapsed_time=150.0)
    assert abs(recon - direct) < 1e-5, f"decompose total mismatch: {recon} vs {direct}"
    print(f"Test 6 — decompose() matches step_reward():")
    for k, v in parts.items():
        print(f"  {k:25s}: {v:+.4f}")
    print("  ✓\n")

    # ── Test 7: running stats ─────────────────────────────────────────
    shaper2 = RewardShaper()
    rewards = []
    for _ in range(100):
        mb = np.random.rand(N).astype(np.float32)
        ma = np.clip(mb + np.random.rand(N) * 0.1, 0, 1).astype(np.float32)
        r  = shaper2.step_reward(mb, ma, correct=np.random.randint(2),
                                 hint_count=0, elapsed_time=30.0)
        rewards.append(r)

    assert abs(shaper2.running_mean - np.mean(rewards)) < 1e-3, \
        "Running mean inaccurate"
    print(f"Test 7 — running stats (100 steps):")
    print(f"  running_mean={shaper2.running_mean:.4f}  "
          f"numpy_mean={np.mean(rewards):.4f}")
    print(f"  running_std ={shaper2.running_std:.4f}")
    print("  ✓\n")

    print(shaper2)
    print("\nAll reward tests passed ✓")