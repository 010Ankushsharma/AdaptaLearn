"""
backend/urls.py
URL routing for the Adaptive Learning System API.

All endpoints live under /api/ — the project-level manage.py includes
this file via ROOT_URLCONF = "backend.urls".

Endpoint map:
    GET  /api/health/                       — agent + DB health check
    POST /api/pretest/                      — submit pre-test score
    POST /api/session/start/                — begin a new tutoring session
    GET  /api/session/<session_id>/         — fetch session state + mastery
    POST /api/session/<session_id>/next/    — get next question from agent
    POST /api/session/<session_id>/answer/  — submit answer, receive reward
    POST /api/session/<session_id>/complete/— explicitly end session
    GET  /api/session/<session_id>/progress/— mastery radar chart data
    POST /api/posttest/<session_id>/        — submit post-test score

Admin:
    /admin/   — Django admin interface (staff only)

Notes:
    - session_id is a UUID (matches Session.id = UUIDField in models.py)
    - All /api/ views require authentication except /api/health/
    - CSRF protection is active; React must send X-CSRFToken header
"""

import uuid

from django.contrib import admin
from django.urls import include, path, re_path
from django.http import JsonResponse
from django.views.static import serve
from django.conf import settings

from backend.views import (
    HealthView,
    NextQuestionView,
    PostTestView,
    PreTestView,
    SessionCompleteView,
    SessionDetailView,
    SessionProgressView,
    SessionStartView,
    SubmitAnswerView,
)


# ─────────────────────────────────────────────────────────────────────────────
# UUID path converter
# ─────────────────────────────────────────────────────────────────────────────

class UUIDConverter:
    """
    Custom path converter that matches only valid UUID strings.

    Using a dedicated converter (rather than a plain <str:id> slug) means
    Django returns 404 automatically for malformed IDs — the views never
    receive garbage that could cause confusing DB errors.

    Registered as "uuid" so paths can be written:
        path("session/<uuid:session_id>/", ...)
    """
    regex = (
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}"
        r"-[0-9a-f]{4}-[0-9a-f]{12}"
    )

    def to_python(self, value: str) -> uuid.UUID:
        return uuid.UUID(value)

    def to_url(self, value: uuid.UUID) -> str:
        return str(value)


from django.urls import register_converter
register_converter(UUIDConverter, "uuid")


# ─────────────────────────────────────────────────────────────────────────────
# API URL patterns
# ─────────────────────────────────────────────────────────────────────────────

api_urlpatterns = [
    # ── Health check (no auth required) ──────────────────────────────────────
    path(
        "health/",
        HealthView.as_view(),
        name="health",
    ),

    # ── Pre-test ──────────────────────────────────────────────────────────────
    path(
        "pretest/",
        PreTestView.as_view(),
        name="pretest",
    ),

    # ── Session lifecycle ─────────────────────────────────────────────────────
    path(
        "session/start/",
        SessionStartView.as_view(),
        name="session-start",
    ),
    path(
        "session/<uuid:session_id>/",
        SessionDetailView.as_view(),
        name="session-detail",
    ),
    path(
        "session/<uuid:session_id>/next/",
        NextQuestionView.as_view(),
        name="session-next",
    ),
    path(
        "session/<uuid:session_id>/answer/",
        SubmitAnswerView.as_view(),
        name="session-answer",
    ),
    path(
        "session/<uuid:session_id>/complete/",
        SessionCompleteView.as_view(),
        name="session-complete",
    ),
    path(
        "session/<uuid:session_id>/progress/",
        SessionProgressView.as_view(),
        name="session-progress",
    ),

    # ── Post-test ─────────────────────────────────────────────────────────────
    path(
        "posttest/<uuid:session_id>/",
        PostTestView.as_view(),
        name="posttest",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Root URL patterns
# ─────────────────────────────────────────────────────────────────────────────

urlpatterns = [
    # Django admin
    path("admin/", admin.site.urls),

    # All API endpoints under /api/
    path("api/", include((api_urlpatterns, "api"))),
]


# ─────────────────────────────────────────────────────────────────────────────
# Development extras
# ─────────────────────────────────────────────────────────────────────────────

if settings.DEBUG:
    # Serve user-uploaded media files locally in development
    # (in production, serve from nginx / S3)
    from django.conf.urls.static import static
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

    # 404 / 500 debug helper — hit /api/404test/ to confirm error handling
    def _404_test(request):
        from django.http import Http404
        raise Http404("Test 404")

    def _500_test(request):
        raise RuntimeError("Test 500 — intentional error")

    urlpatterns += [
        path("api/404test/", _404_test),
        path("api/500test/", _500_test),
    ]