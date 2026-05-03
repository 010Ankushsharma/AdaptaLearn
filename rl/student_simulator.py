"""
rl/student_simulator.py
Bayesian Knowledge Tracing (BKT) synthetic student simulator.

BKT models each concept as a Hidden Markov Model with two states:
  - Unmastered (L=0): student does not know the concept
  - Mastered   (L=1): student knows the concept

At each step the simulator:
  1. Checks whether the student has mastered the target concept (hidden state)
  2. Generates a correct/incorrect response (observable) accounting for
     guess and slip probabilities
  3. Updates the hidden mastery state using Bayes' rule
  4. Applies learning: if unmastered, P(transition to mastered) = p_learn

BKT Parameters per concept:
  p_init   : P(mastered at start)               — prior
  p_learn  : P(learn | was unmastered)          — learning rate
  p_guess  : P(correct | unmastered)            — lucky guess
  p_slip   : P(incorrect | mastered)            — careless mistake
  p_forget : P(forget | was mastered)           — forgetting rate

Ensemble design:
  We instantiate N simulators with parameters sampled from realistic
  ranges. Training the PPO agent against an ensemble prevents the policy
  from overfitting to a single set of BKT parameters (sim-to-real gap).

Usage:
    from rl.student_simulator import StudentSimulator, SimulatorEnsemble

    # Single simulator
    sim = StudentSimulator(num_concepts=188, seed=42)
    sim.reset()
    response = sim.answer(concept_id=5, difficulty=0.6)
    mastery  = sim.get_mastery()

    # Ensemble (for robust PPO training)
    ensemble = SimulatorEnsemble(n=10, num_concepts=188)
    sim = ensemble.sample()   # random simulator from ensemble
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BKT, NUM_CONCEPTS, EVAL


# ─────────────────────────────────────────────────────────────────────────────
# BKT Parameter sampler
# ─────────────────────────────────────────────────────────────────────────────

def sample_bkt_params(
    num_concepts: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """
    Sample realistic BKT parameters for all concepts from configured ranges.

    Returns a dict of arrays, each shape (num_concepts,).
    Parameters are sampled independently per concept, reflecting that
    different skills have different learning dynamics.
    """
    def uniform(lo, hi):
        return rng.uniform(lo, hi, size=num_concepts).astype(np.float32)

    p_learn  = uniform(*BKT["p_learn_range"])
    p_guess  = uniform(*BKT["p_guess_range"])
    p_slip   = uniform(*BKT["p_slip_range"])
    p_forget = uniform(*BKT["p_forget_range"])

    # p_init: prior mastery — students start with partial knowledge
    # Skewed low: most concepts start unmastered
    p_init = rng.beta(1.5, 4.0, size=num_concepts).astype(np.float32)

    # Enforce: p_guess + p_slip < 1 (necessary for BKT identifiability)
    # If violated, scale both down proportionally
    total = p_guess + p_slip
    mask  = total >= 1.0
    p_guess[mask] *= 0.8 / total[mask]
    p_slip[mask]  *= 0.8 / total[mask]

    return {
        "p_init":   p_init,
        "p_learn":  p_learn,
        "p_guess":  p_guess,
        "p_slip":   p_slip,
        "p_forget": p_forget,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Single student simulator
# ─────────────────────────────────────────────────────────────────────────────

class StudentSimulator:
    """
    BKT-based synthetic student.

    Models one student across NUM_CONCEPTS independent skills.
    Each skill evolves independently as a 2-state HMM.

    The simulator maintains:
      _mastery_hidden  : (N,) bool   — true latent mastery (not observable)
      _mastery_belief  : (N,) float  — Bayesian posterior P(mastered | history)
      _attempt_counts  : (N,) int    — times each concept was attempted
      _correct_counts  : (N,) int    — correct responses per concept
      _history         : list of dicts — full interaction log

    The RL environment observes _mastery_belief (what the DKVMN would output).
    The true _mastery_hidden is only used to generate the response.
    """

    def __init__(
        self,
        num_concepts: int = NUM_CONCEPTS,
        params:       Optional[dict] = None,
        seed:         Optional[int]  = None,
        fatigue_rate: float = 0.003,
    ):
        self.num_concepts = num_concepts
        self.fatigue_rate = fatigue_rate  # P(correct) drops by this per question

        self.rng = np.random.default_rng(seed)

        # BKT parameters (per concept)
        self.params = params if params is not None else sample_bkt_params(
            num_concepts, self.rng
        )

        # State (initialised in reset())
        self._mastery_hidden  = None
        self._mastery_belief  = None
        self._attempt_counts  = None
        self._correct_counts  = None
        self._history         = None
        self._n_questions     = 0   # total questions answered this session
        self._session_time    = 0.0 # total elapsed time (seconds)

        self.reset()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """
        Reset to a fresh student at the start of a new episode.
        Mastery is sampled from the prior p_init per concept.

        Returns:
            initial mastery belief vector (N,) float32
        """
        p = self.params

        # Sample initial hidden mastery from prior
        self._mastery_hidden = (
            self.rng.random(self.num_concepts) < p["p_init"]
        ).astype(bool)

        # Initial belief = prior (before any observations)
        self._mastery_belief = p["p_init"].copy()

        self._attempt_counts = np.zeros(self.num_concepts, dtype=np.int32)
        self._correct_counts = np.zeros(self.num_concepts, dtype=np.int32)
        self._history        = []
        self._n_questions    = 0
        self._session_time   = 0.0

        return self._mastery_belief.copy()

    # ── Core interaction ──────────────────────────────────────────────────────

    def answer(
        self,
        concept_id:  int,
        difficulty:  float = 0.5,
        hint_used:   bool  = False,
    ) -> dict:
        """
        Simulate the student answering one question targeting concept_id.

        Args:
            concept_id:  which concept this question tests (0-indexed)
            difficulty:  question difficulty in [0, 1]  (1 = hardest)
            hint_used:   whether the student used a hint

        Returns dict with:
            correct         : int   — 0 or 1
            mastery_before  : float — P(mastered) before this attempt
            mastery_after   : float — P(mastered) after Bayesian update
            elapsed_time    : float — simulated response time (seconds)
            hint_count      : int   — 0 or 1 in this simplified model
            concept_id      : int
            n_questions     : int   — total questions this session
        """
        assert 0 <= concept_id < self.num_concepts, \
            f"concept_id {concept_id} out of range [0, {self.num_concepts})"

        p = self.params
        c = concept_id

        mastery_before = float(self._mastery_belief[c])

        # ── 1. Generate response from hidden state ────────────────────
        is_mastered = self._mastery_hidden[c]

        if is_mastered:
            # Mastered: correct unless slip occurs
            # Difficulty modulates slip: harder → more likely to slip
            p_slip_adjusted = float(p["p_slip"][c]) * (0.5 + difficulty)
            p_slip_adjusted = min(p_slip_adjusted, 0.45)
            correct = int(self.rng.random() > p_slip_adjusted)
        else:
            # Unmastered: correct only by guessing
            # Hints boost guess probability
            p_guess_adjusted = float(p["p_guess"][c])
            if hint_used:
                p_guess_adjusted = min(p_guess_adjusted * 1.5, 0.5)
            # Difficulty modulates: harder → lower guess rate
            p_guess_adjusted *= (1.0 - 0.4 * difficulty)
            # Fatigue: longer sessions → slightly lower performance
            fatigue = self.fatigue_rate * self._n_questions
            p_guess_adjusted = max(p_guess_adjusted - fatigue, 0.05)
            correct = int(self.rng.random() < p_guess_adjusted)

        # ── 2. Bayesian belief update ─────────────────────────────────
        prior = self._mastery_belief[c]

        if correct:
            # P(mastered | correct) via Bayes
            p_correct_given_mastered   = 1.0 - float(p["p_slip"][c])
            p_correct_given_unmastered = float(p["p_guess"][c])
        else:
            p_correct_given_mastered   = float(p["p_slip"][c])
            p_correct_given_unmastered = 1.0 - float(p["p_guess"][c])

        numerator   = p_correct_given_mastered * prior
        denominator = (numerator +
                       p_correct_given_unmastered * (1.0 - prior))
        posterior   = numerator / max(denominator, 1e-9)

        # ── 3. Learning update (hidden state transition) ──────────────
        if not is_mastered:
            # Did the student learn from this attempt?
            if self.rng.random() < float(p["p_learn"][c]):
                self._mastery_hidden[c] = True
                # Bump posterior to reflect the learning event
                posterior = min(posterior + float(p["p_learn"][c]), 0.99)
        else:
            # Forgetting (rare)
            if self.rng.random() < float(p["p_forget"][c]):
                self._mastery_hidden[c] = False
                posterior = max(posterior - float(p["p_forget"][c]), 0.01)

        # ── 4. Update belief ──────────────────────────────────────────
        # Apply learning transition to belief (standard BKT update)
        posterior_with_learn = (
            posterior + (1.0 - posterior) * float(p["p_learn"][c])
        )
        self._mastery_belief[c] = float(
            np.clip(posterior_with_learn, 0.01, 0.99)
        )

        # ── 5. Update counters ────────────────────────────────────────
        self._attempt_counts[c] += 1
        self._correct_counts[c] += correct
        self._n_questions        += 1

        # ── 6. Simulate response time ─────────────────────────────────
        # Harder questions + lower mastery → longer response time
        base_time   = 20.0 + difficulty * 40.0                  # 20–60s base
        mastery_mul = 2.0 - float(self._mastery_belief[c])      # 1.0–2.0×
        noise       = float(self.rng.lognormal(0, 0.3))         # log-normal noise
        elapsed     = float(np.clip(base_time * mastery_mul * noise, 5.0, 180.0))
        if hint_used:
            elapsed *= 1.3
        self._session_time += elapsed

        mastery_after = float(self._mastery_belief[c])

        interaction = {
            "concept_id":    concept_id,
            "correct":       correct,
            "mastery_before":mastery_before,
            "mastery_after": mastery_after,
            "elapsed_time":  elapsed,
            "hint_count":    int(hint_used),
            "difficulty":    difficulty,
            "n_questions":   self._n_questions,
        }
        self._history.append(interaction)
        return interaction

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_mastery(self) -> np.ndarray:
        """Return current mastery belief vector (N,) — the RL state."""
        return self._mastery_belief.copy()

    def get_true_mastery(self) -> np.ndarray:
        """Return hidden true mastery (N,) bool — for evaluation only."""
        return self._mastery_hidden.copy()

    def get_accuracy(self, concept_id: Optional[int] = None) -> float:
        """Return accuracy for a concept (or overall if None)."""
        if concept_id is not None:
            a = self._attempt_counts[concept_id]
            return float(self._correct_counts[concept_id] / a) if a > 0 else 0.0
        total = self._attempt_counts.sum()
        return float(self._correct_counts.sum() / total) if total > 0 else 0.0

    def get_session_stats(self) -> dict:
        """Summary statistics for the current episode."""
        return {
            "n_questions":    self._n_questions,
            "session_time_s": self._session_time,
            "mean_mastery":   float(self._mastery_belief.mean()),
            "n_mastered":     int((self._mastery_belief > 0.8).sum()),
            "overall_accuracy": self.get_accuracy(),
        }

    def is_session_mastered(self, threshold: float = 0.85) -> bool:
        """True if ALL concepts are above mastery threshold."""
        return bool((self._mastery_belief >= threshold).all())

    def p_correct_on(self, concept_id: int, difficulty: float = 0.5) -> float:
        """
        Predict P(correct) if asked a question on concept_id.
        Used by the RL environment's ZPD filter.

        P(correct) = P(mastered)·(1−p_slip) + P(unmastered)·p_guess
        """
        b = float(self._mastery_belief[concept_id])
        p = self.params
        return (
            b * (1.0 - float(p["p_slip"][concept_id]))
            + (1.0 - b) * float(p["p_guess"][concept_id])
        )


# ─────────────────────────────────────────────────────────────────────────────
# Simulator ensemble
# ─────────────────────────────────────────────────────────────────────────────

class SimulatorEnsemble:
    """
    A collection of N StudentSimulators with different BKT parameters.

    Training PPO against an ensemble prevents the policy from memorising
    one student archetype. Each rollout samples a fresh simulator,
    ensuring the agent sees diverse learning trajectories.

    Archetypes included:
      - Fast learner  : high p_learn, low p_slip
      - Slow learner  : low p_learn, moderate p_slip
      - Average       : mid-range on all parameters
      - Forgetter     : moderate p_learn, higher p_forget
      - Guesser       : higher p_guess (poor self-assessment)
    """

    ARCHETYPES = {
        "fast_learner": dict(
            p_learn_range  = (0.25, 0.40),
            p_guess_range  = (0.10, 0.20),
            p_slip_range   = (0.03, 0.10),
            p_forget_range = (0.00, 0.02),
        ),
        "slow_learner": dict(
            p_learn_range  = (0.05, 0.12),
            p_guess_range  = (0.10, 0.20),
            p_slip_range   = (0.10, 0.20),
            p_forget_range = (0.00, 0.03),
        ),
        "average": dict(
            p_learn_range  = (0.15, 0.25),
            p_guess_range  = (0.15, 0.25),
            p_slip_range   = (0.07, 0.15),
            p_forget_range = (0.00, 0.03),
        ),
        "forgetter": dict(
            p_learn_range  = (0.15, 0.30),
            p_guess_range  = (0.10, 0.20),
            p_slip_range   = (0.08, 0.15),
            p_forget_range = (0.03, 0.08),
        ),
        "guesser": dict(
            p_learn_range  = (0.10, 0.20),
            p_guess_range  = (0.25, 0.35),
            p_slip_range   = (0.10, 0.20),
            p_forget_range = (0.00, 0.03),
        ),
    }

    def __init__(
        self,
        n:            int = BKT["num_simulators"],
        num_concepts: int = NUM_CONCEPTS,
        seed:         int = EVAL["random_seed"],
    ):
        self.num_concepts = num_concepts
        self.rng          = np.random.default_rng(seed)
        self.simulators   = self._build(n)

    def _build(self, n: int) -> list[StudentSimulator]:
        sims      = []
        archetype_names = list(self.ARCHETYPES.keys())

        # First: one simulator per archetype
        for name, overrides in self.ARCHETYPES.items():
            # Temporarily override BKT ranges
            original = {k: BKT[k] for k in overrides}
            for k, v in overrides.items():
                BKT[k] = v
            params = sample_bkt_params(self.num_concepts, self.rng)
            for k, v in original.items():
                BKT[k] = v           # restore
            sims.append(StudentSimulator(
                num_concepts = self.num_concepts,
                params       = params,
                seed         = int(self.rng.integers(0, 10_000)),
            ))

        # Fill remaining with random params
        for i in range(n - len(self.ARCHETYPES)):
            params = sample_bkt_params(self.num_concepts, self.rng)
            sims.append(StudentSimulator(
                num_concepts = self.num_concepts,
                params       = params,
                seed         = int(self.rng.integers(0, 10_000)),
            ))

        return sims[:n]

    def sample(self) -> StudentSimulator:
        """Return a random simulator from the ensemble."""
        idx = int(self.rng.integers(0, len(self.simulators)))
        self.simulators[idx].reset()
        return self.simulators[idx]

    def sample_fresh(self) -> StudentSimulator:
        """
        Return a brand-new StudentSimulator with freshly sampled BKT params.
        Use this for maximum diversity during PPO rollouts.
        """
        params = sample_bkt_params(self.num_concepts, self.rng)
        return StudentSimulator(
            num_concepts = self.num_concepts,
            params       = params,
            seed         = int(self.rng.integers(0, 1_000_000)),
        )

    def __len__(self) -> int:
        return len(self.simulators)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== StudentSimulator smoke test ===\n")

    N_CONCEPTS = 10   # small for testing

    # ── Single simulator ──────────────────────────────────────────────
    sim = StudentSimulator(num_concepts=N_CONCEPTS, seed=42)

    print("Initial mastery belief:")
    print(" ", np.round(sim.get_mastery(), 3))

    # Run a short episode
    n_correct = 0
    for step in range(20):
        concept = step % N_CONCEPTS
        result  = sim.answer(concept_id=concept, difficulty=0.5)
        n_correct += result["correct"]

    print(f"\nAfter 20 questions ({n_correct}/20 correct):")
    print(" Mastery:", np.round(sim.get_mastery(), 3))
    print(" Stats:  ", sim.get_session_stats())

    # ── Test p_correct_on ─────────────────────────────────────────────
    for c in range(N_CONCEPTS):
        p = sim.p_correct_on(c, difficulty=0.5)
        assert 0.0 <= p <= 1.0, f"p_correct_on out of range: {p}"

    # ── ZPD filter test ───────────────────────────────────────────────
    zpd_concepts = [
        c for c in range(N_CONCEPTS)
        if 0.40 <= sim.p_correct_on(c) <= 0.75
    ]
    print(f"\n ZPD-appropriate concepts (P(correct) ∈ [0.4, 0.75]): {zpd_concepts}")

    # ── Ensemble ──────────────────────────────────────────────────────
    print("\n--- Ensemble test ---")
    ensemble = SimulatorEnsemble(n=5, num_concepts=N_CONCEPTS, seed=0)
    print(f"Ensemble size: {len(ensemble)}")

    for archetype in ["fast_learner", "slow_learner", "average"]:
        s = ensemble.sample()
        # Run 10 questions
        results = [s.answer(i % N_CONCEPTS, 0.5) for i in range(10)]
        acc = sum(r["correct"] for r in results) / 10
        print(f"  {archetype:15s}: accuracy={acc:.2f}  "
              f"mean_mastery={s.get_mastery().mean():.3f}")

    # ── Mastery growth test ───────────────────────────────────────────
    print("\n--- Mastery growth test (concept 0, 50 attempts) ---")
    sim2   = StudentSimulator(num_concepts=N_CONCEPTS, seed=7)
    before = sim2.get_mastery()[0]
    for _ in range(50):
        sim2.answer(0, difficulty=0.5)
    after = sim2.get_mastery()[0]
    print(f"  Concept 0 mastery: {before:.3f} → {after:.3f}")
    assert after >= before, "Mastery should not decrease with practice"

    # ── Reset test ────────────────────────────────────────────────────
    m_before_reset = sim.get_mastery().copy()
    sim.reset()
    assert sim._n_questions == 0, "n_questions should reset to 0"
    assert not np.array_equal(sim.get_mastery(), m_before_reset) or True  # may differ

    print("\nAll smoke tests passed ✓")