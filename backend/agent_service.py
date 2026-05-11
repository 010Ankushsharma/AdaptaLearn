"""
backend/agent_service.py
PPO policy inference service.

Loads the trained ActorCritic checkpoint once at Django startup
(via AppConfig.ready()) and exposes a single public function:

    next_question(session, student) -> QuestionRecommendation

The service:
  1. Rebuilds the student's mastery vector from their Interaction history
  2. Constructs the full state vector (mastery + session metadata)
  3. Builds the ZPD action mask from current mastery beliefs
  4. Runs the actor forward pass (deterministic argmax for deployment)
  5. Returns the recommended question with metadata

Design principles:
  - Stateless: all state reconstructed from DB on each call
    (safe for multi-process Django deployments)
  - Thread-safe: model weights are read-only after loading
  - Fast: full inference < 10ms on CPU (small MLP, no batching needed)
  - Graceful degradation: falls back to greedy-KT if checkpoint missing

Usage (called from backend/views.py):
    from backend.agent_service import AgentService
    svc = AgentService.get_instance()
    rec = svc.next_question(session, hint_used=False)
"""

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Return type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QuestionRecommendation:
    """
    Result returned by AgentService.next_question().

    Fields:
        question_id     : int    — DB primary key of recommended question
        concept_id      : int    — concept this question tests
        difficulty      : float  — normalised difficulty [0,1]
        p_correct       : float  — predicted P(student gets this right)
        log_prob        : float  — log π(action|state) for offline analysis
        mastery_vector  : list   — current per-concept mastery beliefs
        n_valid_actions : int    — size of ZPD-filtered action space
        inference_ms    : float  — wall-clock inference time
        source          : str    — "ppo" | "greedy_kt" | "random" | "fixed"
    """
    question_id:     int
    concept_id:      int
    difficulty:      float
    p_correct:       float
    log_prob:        float
    mastery_vector:  list
    n_valid_actions: int
    inference_ms:    float
    source:          str = "ppo"


# ─────────────────────────────────────────────────────────────────────────────
# State builder
# ─────────────────────────────────────────────────────────────────────────────

class StateBuilder:
    """
    Reconstructs the RL state vector from a live Django Session object.

    State vector layout (matches rl/env.py exactly):
      [0  : NUM_CONCEPTS]   — P(mastered) per concept
      [NUM_CONCEPTS]        — n_questions / max_questions
      [NUM_CONCEPTS + 1]    — session_time / max_session_time
      [NUM_CONCEPTS + 2]    — cumulative hints / max_hints
      [NUM_CONCEPTS + 3]    — running mean difficulty
    """

    def __init__(self, num_concepts: int, max_questions: int,
                 max_session_time: float = 1800.0):
        self.num_concepts     = num_concepts
        self.max_questions    = max_questions
        self.max_session_time = max_session_time

    def build(self, session, interactions: list) -> np.ndarray:
        """
        Build state vector from Session + ordered Interaction list.

        Args:
            session      : backend.models.Session instance
            interactions : list of Interaction ordered by step

        Returns:
            state : (num_concepts + 4,) float32 array
        """
        # ── Mastery vector ────────────────────────────────────────────
        # Use latest MasterySnapshot if available, else prior
        mastery = self._get_mastery(session, interactions)

        # ── Session metadata ──────────────────────────────────────────
        n_q   = len(interactions)
        n_q_norm = n_q / self.max_questions

        # Estimate session time from interaction timestamps
        if interactions:
            from django.utils import timezone
            elapsed = (timezone.now() - session.started_at).total_seconds()
            time_norm = min(elapsed / self.max_session_time, 1.0)
        else:
            time_norm = 0.0

        # Cumulative hints
        total_hints = sum(i.hint_used for i in interactions)
        hints_norm  = min(total_hints / max(self.max_questions * 2, 1), 1.0)

        # Running mean difficulty
        if interactions:
            avg_diff = float(np.mean([i.difficulty for i in interactions]))
        else:
            avg_diff = 0.5

        meta  = np.array([n_q_norm, time_norm, hints_norm, avg_diff],
                         dtype=np.float32)
        state = np.concatenate([mastery, meta]).clip(0.0, 1.0)
        return state.astype(np.float32)

    def _get_mastery(self, session, interactions: list) -> np.ndarray:
        """
        Return the most recent mastery vector as (num_concepts,) float32.

        Priority:
          1. MasterySnapshot attached to last Interaction
          2. session.mastery_before (set at session start from DKVMN)
          3. Uniform prior (0.5 per concept)
        """
        if interactions:
            last = interactions[-1]
            try:
                snap = last.mastery_snapshot
                vec  = np.array(snap.mastery_vector, dtype=np.float32)
                if len(vec) == self.num_concepts:
                    return vec
            except Exception:
                pass

        if session.mastery_before and len(session.mastery_before) == self.num_concepts:
            return np.array(session.mastery_before, dtype=np.float32)

        # Fallback: uninformed prior
        logger.warning("Using uniform mastery prior for session %s", session.id)
        return np.full(self.num_concepts, 0.5, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# ZPD mask builder (mirrors rl/env.py logic, DB-backed)
# ─────────────────────────────────────────────────────────────────────────────

class ZPDMaskBuilder:
    """
    Builds the boolean action mask over the question pool from the DB.

    In the RL environment we used a synthetic QuestionPool. Here we
    query the Django Question table directly for real question metadata.
    """

    def __init__(self, zpd_lower: float = 0.40, zpd_upper: float = 0.75):
        self.zpd_lower = zpd_lower
        self.zpd_upper = zpd_upper
        self._cache    = None   # (questions_array, num_questions)

    def _load_questions(self):
        """Lazy-load and cache question pool from DB."""
        if self._cache is not None:
            return self._cache

        from backend.models import Question
        qs = Question.objects.values("id", "concept_id", "difficulty").order_by("id")
        pool = list(qs)
        self._cache = pool
        return pool

    def build(
        self,
        mastery: np.ndarray,        # (N,) current mastery beliefs
        answered_ids: set[int],     # question IDs already used this session
    ) -> tuple[np.ndarray, dict[int, float]]:
        """
        Build ZPD mask and p_correct lookup.

        Returns:
            mask           : (num_questions,) bool — True = valid action
            p_correct_map  : {question_id: p_correct}
        """
        pool = self._load_questions()
        if not pool:
            logger.error("Question pool is empty — run load_questions management command")
            return np.array([], dtype=bool), {}

        num_questions = max(q["id"] for q in pool) + 1
        mask          = np.zeros(num_questions, dtype=bool)
        p_correct_map = {}

        for q in pool:
            qid  = q["id"]
            cid  = q["concept_id"]
            diff = q["difficulty"]

            if qid in answered_ids:
                continue   # don't repeat questions in a session

            # P(correct) = P(mastered)·(1−p_slip) + P(unmastered)·p_guess
            # Simplified: use mastery belief directly with difficulty scaling
            if cid < len(mastery):
                m = float(mastery[cid])
                p = m * (1.0 - 0.1 * diff) + (1.0 - m) * (0.2 * (1.0 - diff))
            else:
                p = 0.5

            p_correct_map[qid] = p

            if self.zpd_lower <= p <= self.zpd_upper:
                mask[qid] = True

        # Safety fallback
        if not mask.any():
            logger.warning("ZPD mask empty — opening to all unanswered questions")
            for q in pool:
                if q["id"] not in answered_ids:
                    mask[q["id"]] = True

        return mask, p_correct_map


# ─────────────────────────────────────────────────────────────────────────────
# Fallback policies (used when checkpoint unavailable)
# ─────────────────────────────────────────────────────────────────────────────

def _greedy_kt_fallback(
    mastery: np.ndarray,
    mask: np.ndarray,
    pool: list,
) -> int:
    """Select the valid question targeting the lowest-mastery concept."""
    valid_ids = np.where(mask)[0]
    if len(valid_ids) == 0:
        return int(pool[0]["id"]) if pool else 0

    # Build concept→best_question map
    concept_order = np.argsort(mastery)
    q_by_concept  = {}
    for q in pool:
        cid = q["concept_id"]
        if cid not in q_by_concept:
            q_by_concept[cid] = []
        q_by_concept[cid].append(q)

    for cid in concept_order:
        if cid in q_by_concept:
            qs = [q for q in q_by_concept[cid] if mask[q["id"]]]
            if qs:
                # Pick lowest difficulty for weakest concept
                return int(min(qs, key=lambda q: q["difficulty"])["id"])

    return int(valid_ids[0])


# ─────────────────────────────────────────────────────────────────────────────
# Main service (singleton)
# ─────────────────────────────────────────────────────────────────────────────

class AgentService:
    """
    Singleton PPO inference service.

    Loaded once at Django startup via:

        # backend/apps.py
        class BackendConfig(AppConfig):
            def ready(self):
                AgentService.get_instance()

    Thread-safe: _lock protects singleton creation only.
    The model itself is read-only after loading — concurrent inference
    calls are safe without locking.
    """

    _instance: Optional["AgentService"] = None
    _lock = threading.Lock()

    def __init__(self, checkpoint_path: Optional[Path] = None):
        from config import ENV, NUM_CONCEPTS, NUM_QUESTIONS, PPO

        self.num_concepts  = NUM_CONCEPTS
        self.num_questions = NUM_QUESTIONS
        self.max_questions = ENV["max_questions_per_session"]
        self.state_dim     = ENV["state_dim"]
        self.zpd_lower     = ENV["zpd_lower"]
        self.zpd_upper     = ENV["zpd_upper"]

        self.device = torch.device("cpu")   # CPU for low-latency single requests

        self.state_builder = StateBuilder(
            num_concepts  = self.num_concepts,
            max_questions = self.max_questions,
        )
        self.zpd_builder = ZPDMaskBuilder(
            zpd_lower = self.zpd_lower,
            zpd_upper = self.zpd_upper,
        )

        # Load model
        ckpt_path = checkpoint_path or PPO["checkpoint_path"]
        self.model = self._load_model(ckpt_path)
        self.checkpoint_path = ckpt_path

    @classmethod
    def get_instance(cls, checkpoint_path: Optional[Path] = None) -> "AgentService":
        """Return the singleton, creating it if necessary (thread-safe)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(checkpoint_path)
        return cls._instance

    @classmethod
    def reload(cls, checkpoint_path: Optional[Path] = None) -> "AgentService":
        """Force reload from a new checkpoint (e.g. after retraining)."""
        with cls._lock:
            cls._instance = cls(checkpoint_path)
        logger.info("AgentService reloaded from %s", checkpoint_path)
        return cls._instance

    def _load_model(self, ckpt_path: Path):
        """
        Load ActorCritic from checkpoint.
        Returns None if checkpoint missing — service falls back to greedy-KT.
        """
        from rl.actor_critic import ActorCritic

        model = ActorCritic(
            state_dim  = self.state_dim,
            action_dim = self.num_questions,
        ).to(self.device)

        if ckpt_path and Path(ckpt_path).exists():
            try:
                ckpt = torch.load(ckpt_path, map_location=self.device)
                model.load_state_dict(ckpt["model_state"])
                model.eval()
                logger.info(
                    "PPO checkpoint loaded: %s  (reward=%.3f)",
                    ckpt_path,
                    ckpt.get("mean_reward", float("nan")),
                )
                return model
            except Exception as e:
                logger.error("Failed to load checkpoint %s: %s", ckpt_path, e)
                return None
        else:
            logger.warning(
                "Checkpoint not found at %s — using greedy-KT fallback", ckpt_path
            )
            return None

    # ── Main inference method ──────────────────────────────────────────────

    def next_question(
        self,
        session,
        hint_used:    bool = False,
        deterministic: bool = True,
    ) -> QuestionRecommendation:
        """
        Select the next question for a student in this session.

        Args:
            session       : backend.models.Session (with student preloaded)
            hint_used     : whether the student requested a hint (unused here,
                            affects reward shaping in the RL env at train time)
            deterministic : True = argmax (deployment), False = sample (A/B test)

        Returns:
            QuestionRecommendation dataclass
        """
        t0 = time.perf_counter()

        # ── 1. Load session history ───────────────────────────────────
        interactions = list(
            session.interactions
            .select_related("mastery_snapshot")
            .order_by("step")
        )
        answered_ids = {i.question_id for i in interactions if i.question_id}

        # ── 2. Build state vector ─────────────────────────────────────
        state   = self.state_builder.build(session, interactions)
        mastery = state[:self.num_concepts]

        # ── 3. Build ZPD mask ─────────────────────────────────────────
        mask, p_correct_map = self.zpd_builder.build(mastery, answered_ids)

        # ── 4. Run policy ─────────────────────────────────────────────
        if self.model is not None and mask.any():
            question_id, log_prob, source = self._ppo_inference(
                state, mask, deterministic
            )
        else:
            # Fallback to greedy-KT
            pool = self.zpd_builder._load_questions()
            question_id = _greedy_kt_fallback(mastery, mask, pool)
            log_prob    = float("-inf")
            source      = "greedy_kt"

        # ── 5. Fetch question metadata ─────────────────────────────────
        try:
            from backend.models import Question
            q = Question.objects.get(id=question_id)
            concept_id = q.concept_id
            difficulty = q.difficulty
        except Exception:
            concept_id = int(question_id % self.num_concepts)
            difficulty = 0.5

        p_correct = float(p_correct_map.get(question_id, 0.5))
        inference_ms = (time.perf_counter() - t0) * 1000

        logger.debug(
            "next_question: session=%s step=%d qid=%d concept=%d "
            "p_correct=%.3f src=%s  %.1fms",
            session.id, len(interactions), question_id,
            concept_id, p_correct, source, inference_ms,
        )

        return QuestionRecommendation(
            question_id     = question_id,
            concept_id      = concept_id,
            difficulty      = difficulty,
            p_correct       = p_correct,
            log_prob        = log_prob,
            mastery_vector  = mastery.tolist(),
            n_valid_actions = int(mask.sum()),
            inference_ms    = inference_ms,
            source          = source,
        )

    def _ppo_inference(
        self,
        state:        np.ndarray,
        mask:         np.ndarray,
        deterministic: bool,
    ) -> tuple[int, float, str]:
        """
        Run one forward pass through the actor network.

        Returns:
            (question_id, log_prob, source_label)
        """
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32,
                                   device=self.device).unsqueeze(0)  # (1, S)
            mask_t  = torch.tensor(mask,  dtype=torch.bool,
                                   device=self.device).unsqueeze(0)  # (1, A)

            out = self.model(state_t, mask_t, deterministic=deterministic)

            action   = int(out["action"].item())
            log_prob = float(out["log_prob"].item())

        return action, log_prob, "ppo"

    # ── Utilities ──────────────────────────────────────────────────────────

    def get_mastery_vector(self, session) -> list[float]:
        """
        Return the current mastery vector for a session.
        Called by the frontend to render the mastery radar chart.
        """
        interactions = list(
            session.interactions
            .select_related("mastery_snapshot")
            .order_by("step")
        )
        state   = self.state_builder.build(session, interactions)
        mastery = state[:self.num_concepts]
        return mastery.tolist()

    def warmup(self):
        """
        Run a dummy inference to JIT-compile any torch operations.
        Call at startup to avoid latency spike on first real request.
        """
        dummy_state = np.random.rand(self.state_dim).astype(np.float32)
        dummy_mask  = np.ones(self.num_questions, dtype=bool)
        if self.model is not None:
            self._ppo_inference(dummy_state, dummy_mask, deterministic=True)
        logger.info("AgentService warmed up")

    @property
    def is_ready(self) -> bool:
        """True if the PPO model is loaded and ready for inference."""
        return self.model is not None

    def status(self) -> dict:
        """Health-check endpoint payload."""
        return {
            "ready":            self.is_ready,
            "source":           "ppo" if self.is_ready else "greedy_kt_fallback",
            "checkpoint":       str(self.checkpoint_path),
            "num_concepts":     self.num_concepts,
            "num_questions":    self.num_questions,
            "state_dim":        self.state_dim,
        }