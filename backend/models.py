"""
backend/models.py
Django ORM models for the Adaptive Learning platform.

Schema:
  Student          — registered learner with profile
  Question         — question pool item with concept + difficulty
  Session          — one tutoring session (pre/post test + PPO episode)
  Interaction      — single question-answer event within a session
  MasterySnapshot  — per-concept mastery belief at each interaction step
  PrePostTest      — pre and post test scores for learning gain measurement

Relationships:
  Student    1──∞  Session
  Session    1──∞  Interaction
  Interaction 1──1  MasterySnapshot
  Session    1──1  PrePostTest

All timestamps are UTC. UUIDs used as primary keys for privacy
(no sequential integer IDs that reveal user count).
"""

import uuid

from django.contrib.auth.models import AbstractUser
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.utils import timezone


# ─────────────────────────────────────────────────────────────────────────────
# Student (extends Django's built-in user)
# ─────────────────────────────────────────────────────────────────────────────

class Student(AbstractUser):
    """
    Extended user model for learners.

    Inherits: username, email, password, first_name, last_name,
              is_active, is_staff, date_joined from AbstractUser.

    Adds: study_group (A/B test arm), consent fields, profile metadata.
    """

    STUDY_GROUP_CHOICES = [
        ("ppo",      "PPO Adaptive Agent"),
        ("control",  "Fixed Curriculum"),
    ]

    id = models.UUIDField(
        primary_key = True,
        default     = uuid.uuid4,
        editable    = False,
    )
    study_group = models.CharField(
        max_length = 10,
        choices    = STUDY_GROUP_CHOICES,
        default    = "ppo",
        help_text  = "A/B test arm — assigned at registration",
    )
    consent_given = models.BooleanField(
        default  = False,
        help_text = "GDPR/IRB consent to record interactions",
    )
    consent_timestamp = models.DateTimeField(null=True, blank=True)

    # Onboarding self-report
    prior_experience = models.CharField(
        max_length = 20,
        choices    = [
            ("none",        "No prior knowledge"),
            ("beginner",    "Beginner"),
            ("intermediate","Intermediate"),
            ("advanced",    "Advanced"),
        ],
        default = "none",
    )
    age_range = models.CharField(
        max_length = 10,
        choices    = [
            ("18-22", "18–22"), ("23-27", "23–27"),
            ("28-35", "28–35"), ("36+",   "36+"),
        ],
        blank = True,
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table  = "students"
        ordering  = ["-created_at"]
        verbose_name        = "Student"
        verbose_name_plural = "Students"

    def __str__(self):
        return f"{self.username} ({self.study_group})"

    @property
    def n_sessions(self):
        return self.sessions.count()

    @property
    def total_questions_answered(self):
        return Interaction.objects.filter(session__student=self).count()


# ─────────────────────────────────────────────────────────────────────────────
# Question
# ─────────────────────────────────────────────────────────────────────────────

class Question(models.Model):
    """
    A question in the pool.
    Loaded once from question_meta.parquet at startup via management command.
    Read-only at runtime — the agent selects from this pool.
    """

    id = models.AutoField(primary_key=True)  # matches question_id from parquet

    concept_id = models.SmallIntegerField(
        db_index  = True,
        help_text = "Concept/skill index (0-based, matches DKVMN slot)",
    )
    concept_name = models.CharField(
        max_length = 128,
        blank      = True,
        help_text  = "Human-readable concept label (e.g. 'Python loops')",
    )
    difficulty = models.FloatField(
        validators = [MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text  = "Normalised difficulty: 0=easy, 1=hard",
    )
    question_text = models.TextField(
        blank     = True,
        help_text = "Question body (Markdown supported)",
    )
    answer_options = models.JSONField(
        default   = list,
        blank     = True,
        help_text = "List of answer choice dicts [{text, is_correct}]",
    )
    explanation = models.TextField(
        blank     = True,
        help_text = "Explanation shown after answering",
    )

    # Stats (updated periodically from interaction logs)
    n_attempts = models.IntegerField(default=0)
    accuracy   = models.FloatField(
        null       = True, blank = True,
        validators = [MinValueValidator(0.0), MaxValueValidator(1.0)],
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "questions"
        indexes  = [models.Index(fields=["concept_id", "difficulty"])]

    def __str__(self):
        return f"Q{self.id} [concept={self.concept_id}, diff={self.difficulty:.2f}]"


# ─────────────────────────────────────────────────────────────────────────────
# Session
# ─────────────────────────────────────────────────────────────────────────────

class Session(models.Model):
    """
    One complete tutoring session for one student.

    A session corresponds to one RL episode:
      - Starts with reset() on the Gym environment
      - Ends when max_questions reached or mastery threshold hit
      - Pre/post test scores measured separately (PrePostTest model)
    """

    STATUS_CHOICES = [
        ("active",    "In progress"),
        ("completed", "Completed normally"),
        ("abandoned", "Student left early"),
        ("error",     "System error"),
    ]

    id = models.UUIDField(
        primary_key = True,
        default     = uuid.uuid4,
        editable    = False,
    )
    student    = models.ForeignKey(
        Student,
        on_delete    = models.CASCADE,
        related_name = "sessions",
    )
    status     = models.CharField(
        max_length = 12,
        choices    = STATUS_CHOICES,
        default    = "active",
        db_index   = True,
    )
    session_number = models.PositiveSmallIntegerField(
        default   = 1,
        help_text = "Which session this is for this student (1, 2, 3...)",
    )

    # Timing
    started_at   = models.DateTimeField(default=timezone.now)
    completed_at = models.DateTimeField(null=True, blank=True)

    # RL state at start/end
    mastery_before = models.JSONField(
        default   = list,
        help_text = "Per-concept mastery vector at session start (list of floats)",
    )
    mastery_after  = models.JSONField(
        default   = list,
        help_text = "Per-concept mastery vector at session end",
    )

    # Summary stats (denormalised for fast dashboard queries)
    n_questions       = models.SmallIntegerField(default=0)
    n_correct         = models.SmallIntegerField(default=0)
    total_reward      = models.FloatField(default=0.0)
    mean_mastery_gain = models.FloatField(null=True, blank=True)

    # Agent metadata
    agent_version = models.CharField(
        max_length = 64,
        default    = "ppo_v1",
        help_text  = "Checkpoint identifier used in this session",
    )

    class Meta:
        db_table = "sessions"
        ordering = ["-started_at"]
        indexes  = [
            models.Index(fields=["student", "status"]),
            models.Index(fields=["started_at"]),
        ]
        constraints = [
            models.UniqueConstraint(
                fields = ["student", "session_number"],
                name   = "unique_student_session_number",
            )
        ]

    def __str__(self):
        return f"Session {self.session_number} — {self.student.username} ({self.status})"

    @property
    def duration_seconds(self):
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @property
    def accuracy(self):
        if self.n_questions == 0:
            return None
        return self.n_correct / self.n_questions

    def complete(self, mastery_after: list, total_reward: float):
        """Mark session as completed and store final state."""
        self.status       = "completed"
        self.completed_at = timezone.now()
        self.mastery_after = mastery_after
        self.total_reward  = total_reward
        if self.mastery_before:
            before = sum(self.mastery_before) / len(self.mastery_before)
            after  = sum(mastery_after)       / len(mastery_after)
            self.mean_mastery_gain = after - before
        self.save()


# ─────────────────────────────────────────────────────────────────────────────
# Interaction
# ─────────────────────────────────────────────────────────────────────────────

class Interaction(models.Model):
    """
    A single question-answer event within a session.
    One row per step of the RL episode.

    This table is the most write-heavy — up to 20 rows per session.
    Indexed on session + step for fast ordered retrieval.
    """

    id = models.UUIDField(
        primary_key = True,
        default     = uuid.uuid4,
        editable    = False,
    )
    session  = models.ForeignKey(
        Session,
        on_delete    = models.CASCADE,
        related_name = "interactions",
    )
    question = models.ForeignKey(
        Question,
        on_delete    = models.SET_NULL,
        null         = True,
        related_name = "interactions",
    )

    # Step metadata
    step       = models.SmallIntegerField(help_text="0-based step index in episode")
    concept_id = models.SmallIntegerField(db_index=True)
    difficulty = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )

    # Student response
    correct      = models.BooleanField()
    elapsed_ms   = models.IntegerField(
        help_text="Response time in milliseconds"
    )
    hint_used    = models.BooleanField(default=False)

    # RL quantities (logged for offline analysis)
    reward          = models.FloatField()
    mastery_before  = models.FloatField(
        help_text="Mean mastery across all concepts before this step"
    )
    mastery_after   = models.FloatField(
        help_text="Mean mastery across all concepts after this step"
    )
    action_log_prob = models.FloatField(
        null  = True, blank = True,
        help_text="log π(action|state) from PPO actor — for offline RL analysis",
    )

    timestamp = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = "interactions"
        ordering = ["session", "step"]
        indexes  = [
            models.Index(fields=["session", "step"]),
            models.Index(fields=["concept_id"]),
            models.Index(fields=["timestamp"]),
        ]

    def __str__(self):
        result = "✓" if self.correct else "✗"
        return (
            f"Step {self.step} {result} "
            f"[concept={self.concept_id}, diff={self.difficulty:.2f}]"
        )

    @property
    def elapsed_seconds(self):
        return self.elapsed_ms / 1000.0


# ─────────────────────────────────────────────────────────────────────────────
# MasterySnapshot
# ─────────────────────────────────────────────────────────────────────────────

class MasterySnapshot(models.Model):
    """
    Full per-concept mastery vector after each interaction.

    Stored separately from Interaction to keep the interactions table
    narrow (fast queries) while still preserving the full state for
    analysis and the DKVMN knowledge tracing visualisation.

    mastery_vector: JSON list of floats, length = NUM_CONCEPTS
      e.g. [0.82, 0.45, 0.91, ..., 0.33]
    """

    id = models.UUIDField(
        primary_key = True,
        default     = uuid.uuid4,
        editable    = False,
    )
    interaction = models.OneToOneField(
        Interaction,
        on_delete    = models.CASCADE,
        related_name = "mastery_snapshot",
    )
    mastery_vector = models.JSONField(
        help_text = "List of P(mastered) per concept after this interaction"
    )
    zpd_mask = models.JSONField(
        default   = list,
        blank     = True,
        help_text = "Boolean list — which questions were in ZPD at this step",
    )
    timestamp = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = "mastery_snapshots"

    def __str__(self):
        mean = sum(self.mastery_vector) / max(len(self.mastery_vector), 1)
        return f"Snapshot after step {self.interaction.step} — mean={mean:.3f}"

    @property
    def mean_mastery(self):
        return sum(self.mastery_vector) / max(len(self.mastery_vector), 1)

    @property
    def n_mastered(self, threshold: float = 0.85):
        return sum(1 for p in self.mastery_vector if p >= threshold)


# ─────────────────────────────────────────────────────────────────────────────
# PrePostTest
# ─────────────────────────────────────────────────────────────────────────────

class PrePostTest(models.Model):
    """
    Pre-test and post-test scores for a session.

    Used to compute the primary evaluation metric:
      Normalised Learning Gain = (post - pre) / (1 - pre)

    The test is administered outside the RL session (separate endpoint).
    Questions are different from the training pool to avoid leakage.
    """

    id = models.UUIDField(
        primary_key = True,
        default     = uuid.uuid4,
        editable    = False,
    )
    session = models.OneToOneField(
        Session,
        on_delete    = models.CASCADE,
        related_name = "pre_post_test",
    )

    # Pre-test (taken before the session)
    pre_score        = models.FloatField(
        validators   = [MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text    = "Fraction correct on pre-test (0–1)",
    )
    pre_n_questions  = models.SmallIntegerField(default=20)
    pre_timestamp    = models.DateTimeField(default=timezone.now)

    # Post-test (taken after the session)
    post_score       = models.FloatField(
        null         = True, blank = True,
        validators   = [MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text    = "Fraction correct on post-test (0–1)",
    )
    post_n_questions = models.SmallIntegerField(default=20)
    post_timestamp   = models.DateTimeField(null=True, blank=True)

    # Per-concept breakdown (optional, for fine-grained analysis)
    pre_concept_scores  = models.JSONField(default=dict, blank=True)
    post_concept_scores = models.JSONField(default=dict, blank=True)

    class Meta:
        db_table = "pre_post_tests"

    def __str__(self):
        return (
            f"Test for session {self.session.session_number} "
            f"— pre={self.pre_score:.2f} post={self.post_score or '?'}"
        )

    @property
    def normalised_learning_gain(self) -> float | None:
        """
        Compute normalised learning gain (Hake, 1998):
          NLG = (post − pre) / (1 − pre)

        Returns None if post-test not yet completed.
        Range: [0, 1]  (clipped — negative gain treated as 0)
        """
        if self.post_score is None:
            return None
        potential = 1.0 - self.pre_score
        if potential < 1e-6:
            return 1.0   # student was already at ceiling
        gain = (self.post_score - self.pre_score) / potential
        return float(max(0.0, min(1.0, gain)))

    @property
    def raw_gain(self) -> float | None:
        if self.post_score is None:
            return None
        return float(self.post_score - self.pre_score)