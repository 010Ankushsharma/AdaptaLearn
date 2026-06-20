#!/usr/bin/env python
"""
manage.py
Django's command-line utility for the Adaptive Learning System.

Place this file at the project root (same level as config.py).

Common commands:
    python manage.py runserver              — start dev server on :8000
    python manage.py runserver 0.0.0.0:8000 — accessible on the network
    python manage.py migrate                — apply DB migrations
    python manage.py makemigrations backend — create new migrations
    python manage.py createsuperuser        — create an admin account
    python manage.py shell                  — interactive Django shell
    python manage.py collectstatic          — gather static files for prod
    python manage.py test backend           — run backend unit tests

The settings module is backend.settings — all configuration lives there.
"""

import os
import sys


def main() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")

    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Could not import Django. Make sure it is installed and "
            "available on your PYTHONPATH, and that you have activated "
            "your virtual environment.\n"
            "Run:  pip install -r requirements.txt"
        ) from exc

    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()