"""
backend/apps.py
Django AppConfig for the `backend` app.

The .ready() hook fires exactly once per process, after Django has
finished loading all models and apps but before the server starts
accepting requests. We use it to eagerly load the PPO checkpoint into
memory via AgentService.get_instance() — so the very first API request
doesn't pay a multi-second model-load penalty.

Important: .ready() runs once PER WORKER PROCESS, not once per machine.
With gunicorn --workers 4, the checkpoint is loaded into 4 separate
processes (4x the RAM, but no risk of sharing a stateful model across
processes that don't share memory). This is the correct and standard
behaviour for a stateless-per-request inference service.

Management commands (migrate, makemigrations, shell, etc.) ALSO trigger
.ready() unless guarded — we skip model loading during `migrate` and
`makemigrations` so a fresh deploy without a trained checkpoint yet
doesn't print scary warnings during the database setup step.
"""

import sys

from django.apps import AppConfig


# Management commands where loading the PPO model is unnecessary —
# skip AgentService initialisation during these to keep `migrate` /
# `makemigrations` output clean, especially on a fresh deploy with no
# checkpoint yet.
_SKIP_AGENT_LOAD_FOR = {
    "makemigrations",
    "migrate",
    "collectstatic",
    "createsuperuser",
    "shell",
    "test",
}


class BackendConfig(AppConfig):
    """
    AppConfig for the `backend` Django app.

    default_auto_field matches DEFAULT_AUTO_FIELD in settings.py —
    set explicitly here too so `makemigrations` never prompts for it.
    """

    name              = "backend"
    verbose_name      = "Adaptive Learning Backend"
    default_auto_field = "django.db.models.BigAutoField"

    def ready(self) -> None:
        # Avoid double-import side effects: Django can call ready()
        # twice in some autoreload scenarios (runserver with --noreload
        # disabled). Importing here (not at module level) also avoids
        # circular imports, since agent_service.py imports torch and
        # rl modules that are heavier than a typical Django app import.
        management_command = sys.argv[1] if len(sys.argv) > 1 else None
        if management_command in _SKIP_AGENT_LOAD_FOR:
            return

        from backend.agent_service import AgentService

        try:
            AgentService.get_instance()
        except Exception as exc:  # noqa: BLE001
            # Never crash server startup because the ML model failed
            # to load — agent_service.py already has its own fallback
            # to a greedy-KT policy. Log loudly so it's visible in
            # deploy logs, but let the server keep starting.
            import logging
            logging.getLogger(__name__).warning(
                "AgentService failed to initialise at startup: %s. "
                "API will use fallback policy until this is resolved.",
                exc,
            )