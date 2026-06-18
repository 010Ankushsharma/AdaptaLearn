"""
backend/wsgi.py
WSGI entry point for the Adaptive Learning System.

Used by:
    gunicorn backend.wsgi                  — production (recommended)
    gunicorn backend.wsgi:application      — explicit form
    python manage.py runserver             — dev server (wraps this)
    uwsgi --module backend.wsgi            — alternative WSGI server

Production example (4 worker processes, Unix socket):
    gunicorn backend.wsgi \
        --workers 4 \
        --bind unix:/run/adaptive_learning.sock \
        --timeout 120 \
        --log-level info

The Django application object is loaded once per worker at startup.
AgentService.get_instance() in backend/agent_service.py loads the PPO
checkpoint during AppConfig.ready() — so the model is warm before the
first request arrives.
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")

application = get_wsgi_application()