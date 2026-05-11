"""
backend/views.py
Django REST Framework API views for the Adaptive Learning platform.

Endpoints:
  POST   /api/session/start/          — begin a new tutoring session
  GET    /api/session/<id>/           — get session state + current mastery
  POST   /api/session/<id>/next/      — get next question recommendation
  POST   /api/session/<id>/answer/    — submit an answer, get reward + updated mastery
  POST   /api/session/<id>/complete/  — explicitly end a session
  GET    /api/session/<id>/progress/  — mastery radar chart data
  POST   /api/pretest/                — submit pre-test score
  POST   /api/posttest/<session_id>/  — submit post-test score
  GET    /api/health/                 — agent service health check

Authentication: Django session auth (login required on all endpoints).
CORS: configured in settings.py via django-cors-headers.

Response format: JSON. Errors follow { "error": "<message>" } convention.
"""

import logging
from datetime import datetime

from django.contrib.auth.decorators import login_required
from django.utils import timezone
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from backend.agent_service import AgentService
from backend.models import (
    Interaction, MasterySnapshot, PrePostTest,
    Question, Session, Student,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_session_or_404(session_id: str, student: Student):
    """Return Session or 404 response. Only allows owner access."""
    try:
        return Session.objects.get(id=session_id, student=student), None
    except Session.DoesNotExist:
        return None, Response(
            {"error": "Session not found"},
            status=status.HTTP_404_NOT_FOUND,
        )


def _agent() -> AgentService:
    """Return the singleton AgentService."""
    return AgentService.get_instance()


def _session_summary(session: Session) -> dict:
    """Serialise a Session to a dict for API responses."""
    return {
        "id":               str(session.id),
        "status":           session.status,
        "session_number":   session.session_number,
        "n_questions":      session.n_questions,
        "n_correct":        session.n_correct,
        "accuracy":         session.accuracy,
        "total_reward":     round(session.total_reward, 4),
        "mean_mastery_gain":session.mean_mastery_gain,
        "duration_seconds": session.duration_seconds,
        "started_at":       session.started_at.isoformat(),
        "completed_at":     session.completed_at.isoformat() if session.completed_at else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Session start
# ─────────────────────────────────────────────────────────────────────────────

class SessionStartView(APIView):
    """
    POST /api/session/start/

    Creates a new Session, initialises mastery from DKVMN (or prior),
    and returns the session ID + initial state.

    Request body: {} (empty — all info comes from the authenticated student)

    Response:
        {
            "session_id": "uuid",
            "session_number": 1,
            "mastery_vector": [0.3, 0.45, ...],
            "n_concepts": 188,
            "max_questions": 20,
            "study_group": "ppo"
        }
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        student = request.user

        if not student.consent_given:
            return Response(
                {"error": "Consent required before starting a session"},
                status=status.HTTP_403_FORBIDDEN,
            )

        # Count existing completed sessions for this student
        n_prev = student.sessions.filter(
            status="completed"
        ).count()

        # Abandon any lingering active sessions
        student.sessions.filter(status="active").update(status="abandoned")

        # Initialise mastery vector
        # In production: run DKVMN on student's prior interaction history
        # Here: use uniform prior for first session, last session's mastery_after otherwise
        mastery_before = _get_initial_mastery(student)

        # Create session
        session = Session.objects.create(
            student        = student,
            session_number = n_prev + 1,
            status         = "active",
            mastery_before = mastery_before,
            agent_version  = "ppo_v1",
        )

        logger.info(
            "Session started: student=%s session=%s group=%s",
            student.username, session.id, student.study_group,
        )

        return Response({
            "session_id":     str(session.id),
            "session_number": session.session_number,
            "mastery_vector": mastery_before,
            "n_concepts":     len(mastery_before),
            "max_questions":  20,
            "study_group":    student.study_group,
        }, status=status.HTTP_201_CREATED)


def _get_initial_mastery(student: Student) -> list[float]:
    """
    Get initial mastery vector for a new session.
    Uses last completed session's mastery_after, or uniform prior.
    """
    from config import NUM_CONCEPTS
    last = student.sessions.filter(
        status="completed"
    ).order_by("-started_at").first()

    if last and last.mastery_after and len(last.mastery_after) == NUM_CONCEPTS:
        return last.mastery_after

    # Uniform prior — no history
    return [0.3] * NUM_CONCEPTS


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────

class SessionDetailView(APIView):
    """
    GET /api/session/<session_id>/

    Returns current session state including mastery vector.

    Response:
        { "session": {...}, "mastery_vector": [...], "agent_status": {...} }
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, session_id):
        session, err = _get_session_or_404(session_id, request.user)
        if err:
            return err

        mastery = _agent().get_mastery_vector(session)

        return Response({
            "session":        _session_summary(session),
            "mastery_vector": mastery,
            "agent_status":   _agent().status(),
        })


# ─────────────────────────────────────────────────────────────────────────────
# Next question
# ─────────────────────────────────────────────────────────────────────────────

class NextQuestionView(APIView):
    """
    POST /api/session/<session_id>/next/

    Calls the PPO agent to recommend the next question.

    Request body: { "hint_requested": false }

    Response:
        {
            "question_id": 42,
            "concept_id": 7,
            "concept_name": "Python loops",
            "question_text": "...",
            "answer_options": [...],
            "difficulty": 0.55,
            "p_correct": 0.63,
            "n_valid_actions": 24,
            "inference_ms": 3.2,
            "source": "ppo"
        }
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, session_id):
        session, err = _get_session_or_404(session_id, request.user)
        if err:
            return err

        if session.status != "active":
            return Response(
                {"error": f"Session is {session.status}, not active"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if session.n_questions >= 20:
            return Response(
                {"error": "Session at max questions — call /complete/"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        hint_requested = request.data.get("hint_requested", False)

        # ── Get PPO recommendation ────────────────────────────────────
        rec = _agent().next_question(session, hint_used=hint_requested)

        # ── Fetch full question data ──────────────────────────────────
        try:
            q = Question.objects.get(id=rec.question_id)
            question_data = {
                "question_id":   q.id,
                "concept_id":    q.concept_id,
                "concept_name":  q.concept_name,
                "question_text": q.question_text,
                "answer_options":q.answer_options,
                "difficulty":    round(q.difficulty, 3),
            }
        except Question.DoesNotExist:
            # Fallback for dev when questions aren't loaded
            question_data = {
                "question_id":   rec.question_id,
                "concept_id":    rec.concept_id,
                "concept_name":  f"Concept {rec.concept_id}",
                "question_text": f"[Question {rec.question_id}]",
                "answer_options":[],
                "difficulty":    round(rec.difficulty, 3),
            }

        return Response({
            **question_data,
            "p_correct":       round(rec.p_correct,    3),
            "n_valid_actions": rec.n_valid_actions,
            "inference_ms":    round(rec.inference_ms, 2),
            "source":          rec.source,
        })


# ─────────────────────────────────────────────────────────────────────────────
# Submit answer
# ─────────────────────────────────────────────────────────────────────────────

class SubmitAnswerView(APIView):
    """
    POST /api/session/<session_id>/answer/

    Records a student's answer and returns the updated mastery state.

    Request body:
        {
            "question_id": 42,
            "answer_index": 2,       # index into answer_options
            "elapsed_ms": 34200,     # response time in milliseconds
            "hint_used": false
        }

    Response:
        {
            "correct": true,
            "correct_index": 2,
            "explanation": "...",
            "reward": 0.47,
            "mastery_delta": 0.03,
            "mastery_vector": [...],
            "session_complete": false,
            "step": 5
        }
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, session_id):
        session, err = _get_session_or_404(session_id, request.user)
        if err:
            return err

        if session.status != "active":
            return Response(
                {"error": f"Session is {session.status}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # ── Parse request ─────────────────────────────────────────────
        question_id  = request.data.get("question_id")
        answer_index = request.data.get("answer_index")
        elapsed_ms   = int(request.data.get("elapsed_ms", 30_000))
        hint_used    = bool(request.data.get("hint_used", False))

        if question_id is None or answer_index is None:
            return Response(
                {"error": "question_id and answer_index are required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # ── Fetch question ────────────────────────────────────────────
        try:
            question = Question.objects.get(id=question_id)
        except Question.DoesNotExist:
            return Response(
                {"error": f"Question {question_id} not found"},
                status=status.HTTP_404_NOT_FOUND,
            )

        # ── Grade answer ──────────────────────────────────────────────
        correct_index = next(
            (i for i, opt in enumerate(question.answer_options)
             if opt.get("is_correct")),
            0,
        )
        correct = (int(answer_index) == correct_index)

        # ── Get mastery before this step ──────────────────────────────
        mastery_before_vec = _agent().get_mastery_vector(session)
        mastery_before_mean = sum(mastery_before_vec) / len(mastery_before_vec)

        # ── Compute reward ────────────────────────────────────────────
        from rl.reward import RewardShaper
        import numpy as np

        # Approximate mastery_after: nudge concept up/down by small amount
        # (In full deployment, rerun DKVMN on updated history)
        mastery_after_vec = mastery_before_vec.copy()
        cid = question.concept_id
        if cid < len(mastery_after_vec):
            delta = 0.05 if correct else -0.01
            mastery_after_vec[cid] = float(
                np.clip(mastery_after_vec[cid] + delta, 0.0, 1.0)
            )
        mastery_after_mean = sum(mastery_after_vec) / len(mastery_after_vec)

        shaper = RewardShaper()
        reward = shaper.step_reward(
            mastery_before = np.array(mastery_before_vec, dtype=np.float32),
            mastery_after  = np.array(mastery_after_vec,  dtype=np.float32),
            correct        = int(correct),
            hint_count     = int(hint_used),
            elapsed_time   = elapsed_ms / 1000.0,
        )

        # ── Persist interaction ───────────────────────────────────────
        step = session.n_questions  # 0-based

        interaction = Interaction.objects.create(
            session         = session,
            question        = question,
            step            = step,
            concept_id      = question.concept_id,
            difficulty      = question.difficulty,
            correct         = correct,
            elapsed_ms      = elapsed_ms,
            hint_used       = hint_used,
            reward          = reward,
            mastery_before  = mastery_before_mean,
            mastery_after   = mastery_after_mean,
        )

        MasterySnapshot.objects.create(
            interaction    = interaction,
            mastery_vector = mastery_after_vec,
        )

        # ── Update session counters ───────────────────────────────────
        session.n_questions  += 1
        session.n_correct    += int(correct)
        session.total_reward += reward
        session.save(update_fields=["n_questions", "n_correct", "total_reward"])

        # ── Check termination ─────────────────────────────────────────
        mastery_mean = mastery_after_mean
        all_mastered = all(p >= 0.85 for p in mastery_after_vec)
        session_complete = all_mastered or session.n_questions >= 20

        if session_complete:
            session.complete(
                mastery_after = mastery_after_vec,
                total_reward  = session.total_reward,
            )

        logger.debug(
            "Answer: session=%s step=%d q=%d correct=%s reward=%.3f",
            session.id, step, question_id, correct, reward,
        )

        return Response({
            "correct":          correct,
            "correct_index":    correct_index,
            "explanation":      question.explanation,
            "reward":           round(reward, 4),
            "mastery_delta":    round(mastery_after_mean - mastery_before_mean, 4),
            "mastery_vector":   [round(m, 4) for m in mastery_after_vec],
            "session_complete": session_complete,
            "step":             step + 1,
            "n_questions":      session.n_questions,
        })


# ─────────────────────────────────────────────────────────────────────────────
# Complete session
# ─────────────────────────────────────────────────────────────────────────────

class SessionCompleteView(APIView):
    """
    POST /api/session/<session_id>/complete/

    Explicitly ends a session (if not auto-terminated).
    Called by the frontend when the student finishes early.

    Response: { "session": {...}, "final_mastery": [...] }
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, session_id):
        session, err = _get_session_or_404(session_id, request.user)
        if err:
            return err

        if session.status not in ("active",):
            return Response(
                {"error": f"Session already {session.status}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        mastery = _agent().get_mastery_vector(session)
        session.complete(mastery_after=mastery, total_reward=session.total_reward)

        return Response({
            "session":       _session_summary(session),
            "final_mastery": [round(m, 4) for m in mastery],
        })


# ─────────────────────────────────────────────────────────────────────────────
# Progress (mastery radar chart data)
# ─────────────────────────────────────────────────────────────────────────────

class SessionProgressView(APIView):
    """
    GET /api/session/<session_id>/progress/

    Returns mastery trajectory data for the live radar chart.

    Response:
        {
            "mastery_vector":    [0.45, 0.72, ...],
            "mastery_history":   [[...], [...], ...],  # one per step
            "step_rewards":      [0.12, 0.34, ...],
            "n_mastered":        12,
            "mastery_threshold": 0.85
        }
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, session_id):
        session, err = _get_session_or_404(session_id, request.user)
        if err:
            return err

        interactions = list(
            session.interactions
            .select_related("mastery_snapshot")
            .order_by("step")
        )

        # Build mastery history from snapshots
        mastery_history = []
        for ix in interactions:
            try:
                snap = ix.mastery_snapshot
                mastery_history.append([round(m, 4) for m in snap.mastery_vector])
            except Exception:
                pass

        current_mastery = _agent().get_mastery_vector(session)
        step_rewards    = [round(ix.reward, 4) for ix in interactions]
        n_mastered      = sum(1 for m in current_mastery if m >= 0.85)

        return Response({
            "mastery_vector":    [round(m, 4) for m in current_mastery],
            "mastery_history":   mastery_history,
            "step_rewards":      step_rewards,
            "n_mastered":        n_mastered,
            "mastery_threshold": 0.85,
            "n_questions":       session.n_questions,
        })


# ─────────────────────────────────────────────────────────────────────────────
# Pre / post tests
# ─────────────────────────────────────────────────────────────────────────────

class PreTestView(APIView):
    """
    POST /api/pretest/

    Submit pre-test results before the session.

    Request body:
        {
            "session_id": "uuid",
            "score": 0.45,
            "n_questions": 20,
            "concept_scores": { "0": 0.6, "1": 0.3, ... }  # optional
        }
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        session_id = request.data.get("session_id")
        score      = request.data.get("score")

        if not session_id or score is None:
            return Response(
                {"error": "session_id and score are required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        session, err = _get_session_or_404(session_id, request.user)
        if err:
            return err

        if hasattr(session, "pre_post_test"):
            return Response(
                {"error": "Pre-test already submitted for this session"},
                status=status.HTTP_409_CONFLICT,
            )

        PrePostTest.objects.create(
            session              = session,
            pre_score            = float(score),
            pre_n_questions      = int(request.data.get("n_questions", 20)),
            pre_concept_scores   = request.data.get("concept_scores", {}),
        )

        return Response(
            {"message": "Pre-test recorded", "pre_score": score},
            status=status.HTTP_201_CREATED,
        )


class PostTestView(APIView):
    """
    POST /api/posttest/<session_id>/

    Submit post-test results after the session.
    Computes and returns the normalised learning gain.

    Request body:
        {
            "score": 0.75,
            "n_questions": 20,
            "concept_scores": { "0": 0.8, "1": 0.6, ... }
        }

    Response:
        {
            "post_score": 0.75,
            "pre_score": 0.45,
            "raw_gain": 0.30,
            "normalised_learning_gain": 0.545
        }
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, session_id):
        session, err = _get_session_or_404(session_id, request.user)
        if err:
            return err

        score = request.data.get("score")
        if score is None:
            return Response(
                {"error": "score is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            ppt = session.pre_post_test
        except PrePostTest.DoesNotExist:
            return Response(
                {"error": "No pre-test found — submit pre-test first"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if ppt.post_score is not None:
            return Response(
                {"error": "Post-test already submitted"},
                status=status.HTTP_409_CONFLICT,
            )

        ppt.post_score            = float(score)
        ppt.post_n_questions      = int(request.data.get("n_questions", 20))
        ppt.post_concept_scores   = request.data.get("concept_scores", {})
        ppt.post_timestamp        = timezone.now()
        ppt.save()

        nlg = ppt.normalised_learning_gain

        logger.info(
            "Post-test: student=%s session=%s pre=%.2f post=%.2f nlg=%.3f",
            request.user.username, session_id,
            ppt.pre_score, score, nlg or 0,
        )

        return Response({
            "post_score":                float(score),
            "pre_score":                 ppt.pre_score,
            "raw_gain":                  ppt.raw_gain,
            "normalised_learning_gain":  nlg,
        })


# ─────────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────────

class HealthView(APIView):
    """
    GET /api/health/

    Returns agent service status. No authentication required.
    Used by Docker health checks and monitoring.
    """
    permission_classes = []

    def get(self, request):
        svc = _agent()
        return Response({
            "status":  "ok" if svc.is_ready else "degraded",
            "agent":   svc.status(),
            "time":    timezone.now().isoformat(),
        })