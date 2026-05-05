"""
rl/env.py
Custom Gymnasium environment for the Adaptive Learning System.

The environment models one tutoring session as a finite-horizon MDP:

  State  s_t : mastery belief vector (NUM_CONCEPTS,) + session metadata (4,)
               = [P(mastered_c0), ..., P(mastered_cN), n_q, time, hints, avg_diff]

  Action a_t : integer index into the question pool
               constrained by ZPD mask — only questions with
               P(correct) ∈ [zpd_lower, zpd_upper] are selectable

  Reward r_t : shaped reward from rl/reward.py
               = α·ΔMastery + β·correct − γ·hints − δ·time_overrun
               + episode_end bonus on mastery gain

  Terminal   : max_questions reached OR all concepts mastered

The environment wraps StudentSimulator for training (synthetic students)
and exposes the same interface for real students via the Django backend.

Usage:
    from rl.env import AdaptiveLearningEnv

    env = AdaptiveLearningEnv()
    obs, info = env.reset()
    done = False
    while not done:
        action = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
"""

import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ENV, REWARD, NUM_CONCEPTS, NUM_QUESTIONS, EVAL
from rl.student_simulator import StudentSimulator, SimulatorEnsemble
from rl.reward import RewardShaper


# ─────────────────────────────────────────────────────────────────────────────
# Question pool
# ─────────────────────────────────────────────────────────────────────────────

class QuestionPool:
    """
    Lookup table mapping question_id → (concept_id, difficulty).

    In training, populated from question_meta.parquet (built by preprocess.py).
    Falls back to a synthetic pool if the file isn't available yet.
    """

    def __init__(self, num_questions: int = NUM_QUESTIONS,
                 num_concepts: int = NUM_CONCEPTS):
        self.num_questions = num_questions
        self.num_concepts  = num_concepts
        self._pool = self._load_or_build()

    def _load_or_build(self) -> np.ndarray:
        """
        Returns structured array shape (num_questions,) with fields:
          concept_id : int16
          difficulty : float32
        """
        from config import DATA_PROC_DIR
        meta_path = DATA_PROC_DIR / "question_meta.parquet"

        if meta_path.exists():
            import pandas as pd
            df = pd.read_parquet(meta_path)
            # Align to our question index — fill gaps with defaults
            pool = np.zeros(
                self.num_questions,
                dtype=[("concept_id", np.int16), ("difficulty", np.float32)]
            )
            valid = df["question_id"].values.astype(int)
            valid = valid[valid < self.num_questions]
            pool["concept_id"][valid] = df.loc[
                df["question_id"] < self.num_questions, "concept_id"
            ].values.astype(np.int16)[:len(valid)]
            pool["difficulty"][valid] = df.loc[
                df["question_id"] < self.num_questions, "difficulty"
            ].values.astype(np.float32)[:len(valid)]
            return pool
        else:
            # Synthetic pool: assign each question to a concept round-robin
            # difficulty sampled from Beta(2, 2) — peaks around 0.5
            rng  = np.random.default_rng(EVAL["random_seed"])
            pool = np.zeros(
                self.num_questions,
                dtype=[("concept_id", np.int16), ("difficulty", np.float32)]
            )
            pool["concept_id"] = (
                np.arange(self.num_questions) % self.num_concepts
            ).astype(np.int16)
            pool["difficulty"] = rng.beta(2, 2, size=self.num_questions).astype(np.float32)
            return pool

    def get_concept(self, question_id: int) -> int:
        return int(self._pool["concept_id"][question_id])

    def get_difficulty(self, question_id: int) -> float:
        return float(self._pool["difficulty"][question_id])

    def questions_for_concept(self, concept_id: int) -> np.ndarray:
        """Return all question IDs that test this concept."""
        return np.where(self._pool["concept_id"] == concept_id)[0]


# ─────────────────────────────────────────────────────────────────────────────
# ZPD Mask builder
# ─────────────────────────────────────────────────────────────────────────────

def build_zpd_mask(
    simulator:    StudentSimulator,
    pool:         QuestionPool,
    zpd_lower:    float = ENV["zpd_lower"],
    zpd_upper:    float = ENV["zpd_upper"],
) -> np.ndarray:
    """
    Build a boolean action mask over all questions.

    mask[q] = True  iff  P(correct on question q) ∈ [zpd_lower, zpd_upper]

    The Zone of Proximal Development constraint ensures the agent only
    presents questions that are neither too easy nor too hard — the
    productive difficulty range where learning is maximised.

    Args:
        simulator : current student (provides p_correct_on per concept)
        pool      : question → (concept, difficulty) lookup
        zpd_lower : minimum P(correct) threshold (default 0.40)
        zpd_upper : maximum P(correct) threshold (default 0.75)

    Returns:
        mask : (NUM_QUESTIONS,) bool array
    """
    mask = np.zeros(pool.num_questions, dtype=bool)

    for q_id in range(pool.num_questions):
        c_id = pool.get_concept(q_id)
        diff = pool.get_difficulty(q_id)
        p    = simulator.p_correct_on(c_id, difficulty=diff)
        if zpd_lower <= p <= zpd_upper:
            mask[q_id] = True

    # Safety: if ZPD filter leaves no valid actions, open to all questions
    if not mask.any():
        mask[:] = True

    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Main environment
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveLearningEnv(gym.Env):
    """
    Custom Gymnasium environment for adaptive question sequencing.

    Observation space:
        Box(0, 1, shape=(NUM_CONCEPTS + 4,), dtype=float32)

        Indices:
          [0  : NUM_CONCEPTS]   — P(mastered) per concept  ∈ [0,1]
          [NUM_CONCEPTS]        — n_questions / max_questions  ∈ [0,1]
          [NUM_CONCEPTS + 1]    — session_time / max_session_time  ∈ [0,1]
          [NUM_CONCEPTS + 2]    — cumulative hints / max_hints  ∈ [0,1]
          [NUM_CONCEPTS + 3]    — running mean difficulty  ∈ [0,1]

    Action space:
        Discrete(NUM_QUESTIONS)
        Masked at each step by the ZPD filter.

    Metadata:
        render_modes = []
        action_mask  = True   (stable-baselines3 MaskablePPO compatible)
    """

    metadata = {"render_modes": [], "action_mask": True}

    def __init__(
        self,
        simulator:         Optional[StudentSimulator]  = None,
        ensemble:          Optional[SimulatorEnsemble] = None,
        pool:              Optional[QuestionPool]       = None,
        max_questions:     int   = ENV["max_questions_per_session"],
        mastery_threshold: float = ENV["mastery_threshold"],
        zpd_lower:         float = ENV["zpd_lower"],
        zpd_upper:         float = ENV["zpd_upper"],
        use_fresh_student: bool  = True,
        render_mode:       Optional[str] = None,
    ):
        super().__init__()

        self.max_questions      = max_questions
        self.mastery_threshold  = mastery_threshold
        self.zpd_lower          = zpd_lower
        self.zpd_upper          = zpd_upper
        self.use_fresh_student  = use_fresh_student
        self.render_mode        = render_mode

        # Question pool (shared, read-only)
        self.pool = pool or QuestionPool()

        # Student source
        if simulator is not None:
            # Fixed student — used in evaluation / real deployment
            self._fixed_sim  = simulator
            self._ensemble   = None
        else:
            # Ensemble — sample a new student each episode during training
            self._fixed_sim = None
            self._ensemble  = ensemble or SimulatorEnsemble()

        # Reward shaper
        self.reward_shaper = RewardShaper()

        # Spaces
        state_dim = NUM_CONCEPTS + 4
        self.observation_space = spaces.Box(
            low   = 0.0,
            high  = 1.0,
            shape = (state_dim,),
            dtype = np.float32,
        )
        self.action_space = spaces.Discrete(self.pool.num_questions)

        # Episode state (initialised in reset())
        self.simulator       = None
        self._current_mask   = None
        self._n_questions    = 0
        self._cumulative_hints = 0
        self._difficulties   = []
        self._mastery_before = None
        self._episode_reward = 0.0
        self._history        = []

    # ── Gymnasium API ──────────────────────────────────────────────────────────

    def reset(
        self,
        seed:    Optional[int]  = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Start a new tutoring episode with a fresh (or fixed) student.

        Returns:
            observation : (state_dim,) float32
            info        : dict with action_mask and episode metadata
        """
        super().reset(seed=seed)

        # ── Pick student ──────────────────────────────────────────────
        if self._fixed_sim is not None:
            self.simulator = self._fixed_sim
            self.simulator.reset()
        elif self.use_fresh_student:
            self.simulator = self._ensemble.sample_fresh()
        else:
            self.simulator = self._ensemble.sample()

        # ── Reset episode counters ────────────────────────────────────
        self._n_questions      = 0
        self._cumulative_hints = 0
        self._difficulties     = []
        self._mastery_before   = self.simulator.get_mastery().copy()
        self._episode_reward   = 0.0
        self._history          = []

        # ── Build initial ZPD mask ────────────────────────────────────
        self._current_mask = build_zpd_mask(
            self.simulator, self.pool, self.zpd_lower, self.zpd_upper
        )

        obs  = self._build_observation()
        info = self._build_info()
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Present question `action` to the student and advance the episode.

        Args:
            action : question_id chosen by the PPO agent

        Returns:
            obs        : next state observation
            reward     : shaped reward for this step
            terminated : True if all concepts mastered
            truncated  : True if max_questions reached
            info       : step metadata dict
        """
        assert self.simulator is not None, "Call reset() before step()"

        # ── Resolve question ──────────────────────────────────────────
        # If agent chose a masked action (shouldn't happen with MaskablePPO,
        # but handle gracefully in evaluation), snap to nearest valid action
        if not self._current_mask[action]:
            valid = np.where(self._current_mask)[0]
            action = int(valid[np.random.randint(len(valid))])

        concept_id = self.pool.get_concept(action)
        difficulty = self.pool.get_difficulty(action)

        mastery_before = self.simulator.get_mastery().copy()

        # ── Student answers ───────────────────────────────────────────
        hint_used = False   # future extension: agent can choose to offer hint
        result = self.simulator.answer(
            concept_id = concept_id,
            difficulty = difficulty,
            hint_used  = hint_used,
        )

        mastery_after = self.simulator.get_mastery().copy()

        # ── Reward ────────────────────────────────────────────────────
        self._n_questions      += 1
        self._cumulative_hints += result["hint_count"]
        self._difficulties.append(difficulty)

        reward = self.reward_shaper.step_reward(
            mastery_before = mastery_before,
            mastery_after  = mastery_after,
            correct        = result["correct"],
            hint_count     = result["hint_count"],
            elapsed_time   = result["elapsed_time"],
        )
        self._episode_reward += reward

        # ── Termination ───────────────────────────────────────────────
        terminated = self.simulator.is_session_mastered(self.mastery_threshold)
        truncated  = self._n_questions >= self.max_questions

        # Episode-end bonus added to final reward
        if terminated or truncated:
            mastery_gain = float(
                mastery_after.mean() - self._mastery_before.mean()
            )
            reward += self.reward_shaper.episode_bonus(mastery_gain)

        # ── Update ZPD mask for next step ─────────────────────────────
        if not (terminated or truncated):
            self._current_mask = build_zpd_mask(
                self.simulator, self.pool, self.zpd_lower, self.zpd_upper
            )

        # ── Log step ──────────────────────────────────────────────────
        self._history.append({
            "step":        self._n_questions,
            "question_id": action,
            "concept_id":  concept_id,
            "difficulty":  difficulty,
            "correct":     result["correct"],
            "reward":      reward,
            "mastery_mean": float(mastery_after.mean()),
        })

        obs  = self._build_observation()
        info = self._build_info(result=result, action=action)

        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """
        Return the current ZPD action mask.
        Called by sb3-contrib MaskablePPO at every step.

        Returns:
            (NUM_QUESTIONS,) bool — True = valid action
        """
        return self._current_mask.copy()

    # ── Observation & info builders ────────────────────────────────────────────

    def _build_observation(self) -> np.ndarray:
        """
        Construct the (NUM_CONCEPTS + 4,) state vector.

        Mastery vector + 4 normalised session features:
          n_questions_norm  : progress through max questions
          time_norm         : session time progress
          hints_norm        : cumulative hint usage
          avg_difficulty    : rolling mean question difficulty
        """
        mastery = self.simulator.get_mastery().astype(np.float32)  # (N,)

        stats = self.simulator.get_session_stats()

        n_q_norm    = self._n_questions / self.max_questions
        time_norm   = min(stats["session_time_s"] / (self.max_questions * 90.0), 1.0)
        hints_norm  = min(self._cumulative_hints / (self.max_questions * 2.0), 1.0)
        avg_diff    = float(np.mean(self._difficulties)) if self._difficulties else 0.5

        meta = np.array([n_q_norm, time_norm, hints_norm, avg_diff], dtype=np.float32)

        obs = np.concatenate([mastery, meta])
        return obs.clip(0.0, 1.0)

    def _build_info(self, result: Optional[dict] = None,
                    action: Optional[int] = None) -> dict:
        """Build the info dict returned with every step/reset."""
        info = {
            "action_mask":    self._current_mask,
            "n_questions":    self._n_questions,
            "episode_reward": self._episode_reward,
            "mastery_mean":   float(self.simulator.get_mastery().mean()),
            "n_mastered":     int(
                (self.simulator.get_mastery() >= self.mastery_threshold).sum()
            ),
            "zpd_valid_actions": int(self._current_mask.sum()),
        }
        if result is not None:
            info.update({
                "question_id": action,
                "correct":     result["correct"],
                "difficulty":  result.get("difficulty", 0.5),
                "elapsed_s":   result["elapsed_time"],
            })
        return info

    # ── Utilities ──────────────────────────────────────────────────────────────

    def get_episode_history(self) -> list[dict]:
        """Return the full step-by-step history of the current episode."""
        return self._history.copy()

    def get_mastery_vector(self) -> np.ndarray:
        """Current mastery belief vector — used by the Django agent service."""
        return self.simulator.get_mastery()

    def seed(self, seed: Optional[int] = None):
        """Set random seed (Gymnasium compatibility)."""
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

    def render(self):
        """Text render of current episode state."""
        mastery = self.simulator.get_mastery()
        print(f"\n Step {self._n_questions}/{self.max_questions}")
        print(f" Mean mastery : {mastery.mean():.3f}")
        print(f" N mastered   : {(mastery >= self.mastery_threshold).sum()}/{NUM_CONCEPTS}")
        print(f" Valid actions: {self._current_mask.sum()}/{self.pool.num_questions}")
        print(f" Episode Rwd  : {self._episode_reward:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Registered env factory (for stable-baselines3)
# ─────────────────────────────────────────────────────────────────────────────

def make_env(rank: int = 0, seed: int = EVAL["random_seed"]):
    """
    Factory function for vectorised environments (SB3 VecEnv).

    Usage:
        from stable_baselines3.common.env_util import make_vec_env
        vec_env = make_vec_env(make_env, n_envs=4)
    """
    def _init():
        env = AdaptiveLearningEnv()
        env.reset(seed=seed + rank)
        return env
    return _init


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== AdaptiveLearningEnv smoke test ===\n")

    env = AdaptiveLearningEnv()

    # ── Reset ─────────────────────────────────────────────────────────
    obs, info = env.reset(seed=42)
    assert obs.shape == (NUM_CONCEPTS + 4,), f"Bad obs shape: {obs.shape}"
    assert obs.min() >= 0.0 and obs.max() <= 1.0, "Obs out of [0,1]"
    assert "action_mask" in info
    assert info["action_mask"].shape == (NUM_QUESTIONS,)
    assert info["action_mask"].any(), "ZPD mask is all False"
    print(f"reset() OK — obs shape {obs.shape}, "
          f"{info['zpd_valid_actions']} valid actions")

    # ── Full episode with random valid actions ─────────────────────────
    total_reward = 0.0
    step_count   = 0

    obs, info = env.reset(seed=0)
    done = False
    while not done:
        mask   = env.action_masks()
        valid  = np.where(mask)[0]
        action = int(np.random.choice(valid))

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count   += 1
        done = terminated or truncated

    print(f"Episode done — steps: {step_count}, "
          f"total_reward: {total_reward:.3f}, "
          f"mastery: {info['mastery_mean']:.3f}, "
          f"n_mastered: {info['n_mastered']}")

    assert step_count <= ENV["max_questions_per_session"], \
        f"Episode ran too long: {step_count}"
    assert obs.shape == (NUM_CONCEPTS + 4,)
    assert np.isfinite(total_reward), "Non-finite total reward"

    # ── Observation bounds throughout episode ─────────────────────────
    for _ in range(3):
        obs, _ = env.reset()
        for _ in range(5):
            mask  = env.action_masks()
            valid = np.where(mask)[0]
            obs, r, term, trunc, info = env.step(int(np.random.choice(valid)))
            assert obs.min() >= 0.0 and obs.max() <= 1.0, \
                f"Obs out of bounds: min={obs.min()}, max={obs.max()}"
            assert np.isfinite(r), f"Non-finite reward: {r}"

    print("Obs bounds and reward finiteness: OK across 3 episodes")

    # ── History ───────────────────────────────────────────────────────
    history = env.get_episode_history()
    assert len(history) == step_count
    assert all("correct" in h for h in history)
    print(f"History: {len(history)} steps logged correctly")

    # ── Render ────────────────────────────────────────────────────────
    env.render()

    print("\nAll smoke tests passed ✓")