"""
backend/settings.py
Django settings for the Adaptive Learning System.

Environment variables (set in .env — see .env.example):
    DJANGO_SECRET_KEY      — required in production
    DJANGO_DEBUG           — "True" (dev) or "False" (prod)
    DATABASE_URL           — postgres://user:pass@host:5432/dbname
                             falls back to SQLite for local dev
    REDIS_URL              — redis://localhost:6379/0
    ALLOWED_HOSTS          — comma-separated hostnames for production
    CORS_ALLOWED_ORIGINS   — comma-separated frontend origins
    DJANGO_LOG_LEVEL       — DEBUG / INFO / WARNING (default: INFO)

Usage:
    python manage.py runserver           # development
    python manage.py migrate             # apply DB migrations
    python manage.py createsuperuser     # create admin user
    gunicorn backend.wsgi                # production
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

# Project root: one level above backend/
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env from project root (safe no-op if file doesn't exist)
load_dotenv(BASE_DIR / ".env")

# Keep project root on sys.path so `from config import ...` works
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


# ─────────────────────────────────────────────────────────────────────────────
# Core settings
# ─────────────────────────────────────────────────────────────────────────────

SECRET_KEY = os.environ.get(
    "DJANGO_SECRET_KEY",
    "django-insecure-dev-key-change-this-in-production-do-not-use-in-prod",
)

DEBUG = os.environ.get("DJANGO_DEBUG", "True") == "True"

# Allowed hosts: always include localhost variants; add more via env in prod
_extra_hosts = [
    h.strip()
    for h in os.environ.get("ALLOWED_HOSTS", "").split(",")
    if h.strip()
]
ALLOWED_HOSTS = ["localhost", "127.0.0.1", "0.0.0.0"] + _extra_hosts


# ─────────────────────────────────────────────────────────────────────────────
# Installed apps
# ─────────────────────────────────────────────────────────────────────────────

INSTALLED_APPS = [
    # Django built-ins
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",

    # Third-party
    "rest_framework",
    "corsheaders",

    # Our app — contains models, views, agent_service
    "backend",
]


# ─────────────────────────────────────────────────────────────────────────────
# Middleware
# ─────────────────────────────────────────────────────────────────────────────

MIDDLEWARE = [
    # CORS must come first so preflight OPTIONS requests are handled
    # before any other middleware touches the response
    "corsheaders.middleware.CorsMiddleware",

    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]


# ─────────────────────────────────────────────────────────────────────────────
# URLs and WSGI
# ─────────────────────────────────────────────────────────────────────────────

ROOT_URLCONF = "backend.urls"

WSGI_APPLICATION = "backend.wsgi.application"


# ─────────────────────────────────────────────────────────────────────────────
# Templates (needed for Django admin)
# ─────────────────────────────────────────────────────────────────────────────

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Database
# ─────────────────────────────────────────────────────────────────────────────

_DATABASE_URL = os.environ.get("DATABASE_URL", "")

if _DATABASE_URL.startswith("postgres"):
    # Production / staging: PostgreSQL via DATABASE_URL
    # Format: postgres://user:pass@host:5432/dbname
    import re
    _m = re.match(
        r"postgres(?:ql)?://(?P<user>[^:]+):(?P<password>[^@]+)@"
        r"(?P<host>[^:/]+)(?::(?P<port>\d+))?/(?P<name>.+)",
        _DATABASE_URL,
    )
    if not _m:
        raise ValueError(f"Cannot parse DATABASE_URL: {_DATABASE_URL!r}")

    DATABASES = {
        "default": {
            "ENGINE":   "django.db.backends.postgresql",
            "NAME":     _m.group("name"),
            "USER":     _m.group("user"),
            "PASSWORD": _m.group("password"),
            "HOST":     _m.group("host"),
            "PORT":     _m.group("port") or "5432",
            "CONN_MAX_AGE": 60,  # reuse connections for 60s
            "OPTIONS": {
                "connect_timeout": 10,
            },
        }
    }
else:
    # Development: SQLite (zero-configuration, no server needed)
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME":   BASE_DIR / "db.sqlite3",
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# Custom user model
# ─────────────────────────────────────────────────────────────────────────────

# Must match the Student model in backend/models.py which extends AbstractUser
AUTH_USER_MODEL = "backend.Student"


# ─────────────────────────────────────────────────────────────────────────────
# Password validation
# ─────────────────────────────────────────────────────────────────────────────

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]


# ─────────────────────────────────────────────────────────────────────────────
# Session / cache (Redis in production, local memory in dev)
# ─────────────────────────────────────────────────────────────────────────────

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# Use Redis for sessions if it's reachable, else database sessions
SESSION_ENGINE = "django.contrib.sessions.backends.db"  # safe default

# Django cache: Redis in production, in-memory in dev
if not DEBUG:
    CACHES = {
        "default": {
            "BACKEND":  "django.core.cache.backends.redis.RedisCache",
            "LOCATION": REDIS_URL,
            "OPTIONS":  {"socket_connect_timeout": 5},
        }
    }
else:
    CACHES = {
        "default": {
            "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# Celery (async tasks: e.g. PPO policy updates, logging pipelines)
# ─────────────────────────────────────────────────────────────────────────────

CELERY_BROKER_URL        = REDIS_URL
CELERY_RESULT_BACKEND    = REDIS_URL
CELERY_ACCEPT_CONTENT    = ["json"]
CELERY_TASK_SERIALIZER   = "json"
CELERY_RESULT_SERIALIZER = "json"
CELERY_TIMEZONE          = "UTC"


# ─────────────────────────────────────────────────────────────────────────────
# Django REST Framework
# ─────────────────────────────────────────────────────────────────────────────

REST_FRAMEWORK = {
    # Session auth for the React SPA (cookie-based, same origin)
    # Basic auth for quick CLI / Postman testing in dev
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.SessionAuthentication",
        "rest_framework.authentication.BasicAuthentication",
    ],

    # All endpoints require login by default; override per-view as needed
    # (HealthView already sets permission_classes = [] to allow unauthenticated)
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.IsAuthenticated",
    ],

    # Return JSON by default; include BrowsableAPI only in debug mode
    "DEFAULT_RENDERER_CLASSES": (
        ["rest_framework.renderers.JSONRenderer",
         "rest_framework.renderers.BrowsableAPIRenderer"]
        if DEBUG else
        ["rest_framework.renderers.JSONRenderer"]
    ),

    # Consistent error format: {"error": "..."} matches views.py convention
    "EXCEPTION_HANDLER": "rest_framework.views.exception_handler",

    # Pagination — off by default (sessions list is short per user)
    "DEFAULT_PAGINATION_CLASS": None,

    # Throttle anonymous requests to protect the health endpoint
    "DEFAULT_THROTTLE_CLASSES": [
        "rest_framework.throttling.AnonRateThrottle",
    ],
    "DEFAULT_THROTTLE_RATES": {
        "anon": "60/min",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# CORS (Cross-Origin Resource Sharing)
# ─────────────────────────────────────────────────────────────────────────────

# In development, allow the React dev server
_cors_origins_env = os.environ.get("CORS_ALLOWED_ORIGINS", "")
_cors_from_env = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]

CORS_ALLOWED_ORIGINS = _cors_from_env or [
    "http://localhost:3000",   # React dev server (npm start)
    "http://127.0.0.1:3000",
]

# Allow cookies / session credentials from the frontend
CORS_ALLOW_CREDENTIALS = True

# Allow the CSRF token header so React can POST
CORS_ALLOW_HEADERS = [
    "accept",
    "accept-encoding",
    "authorization",
    "content-type",
    "origin",
    "x-csrftoken",
    "x-requested-with",
]

# CSRF: React reads this cookie and sends it back as X-CSRFToken
CSRF_COOKIE_NAME     = "csrftoken"
CSRF_COOKIE_SAMESITE = "Lax"
CSRF_TRUSTED_ORIGINS = CORS_ALLOWED_ORIGINS


# ─────────────────────────────────────────────────────────────────────────────
# Internationalisation
# ─────────────────────────────────────────────────────────────────────────────

LANGUAGE_CODE = "en-us"
TIME_ZONE     = "UTC"
USE_I18N      = True
USE_TZ        = True          # all datetimes stored as UTC


# ─────────────────────────────────────────────────────────────────────────────
# Static and media files
# ─────────────────────────────────────────────────────────────────────────────

STATIC_URL  = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"   # collectstatic writes here

MEDIA_URL   = "/media/"
MEDIA_ROOT  = BASE_DIR / "media"


# ─────────────────────────────────────────────────────────────────────────────
# Default primary key
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

_LOG_LEVEL = os.environ.get("DJANGO_LOG_LEVEL", "INFO").upper()
LOG_DIR     = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,

    "formatters": {
        "verbose": {
            "format": "{asctime} {levelname} {name} {message}",
            "style":  "{",
        },
        "simple": {
            "format": "{levelname} {message}",
            "style":  "{",
        },
    },

    "handlers": {
        "console": {
            "class":     "logging.StreamHandler",
            "formatter": "verbose",
        },
        "file": {
            "class":       "logging.handlers.RotatingFileHandler",
            "filename":    LOG_DIR / "django.log",
            "maxBytes":    10 * 1024 * 1024,  # 10 MB
            "backupCount": 5,
            "formatter":   "verbose",
        },
    },

    "root": {
        "handlers": ["console"],
        "level":    _LOG_LEVEL,
    },

    "loggers": {
        "django": {
            "handlers":  ["console", "file"],
            "level":     _LOG_LEVEL,
            "propagate": False,
        },
        # Our app code — verbose in dev, quieter in prod
        "backend": {
            "handlers":  ["console", "file"],
            "level":     "DEBUG" if DEBUG else "INFO",
            "propagate": False,
        },
        # Silence noisy third-party loggers
        "django.db.backends": {
            "handlers":  ["console"],
            "level":     "WARNING",
            "propagate": False,
        },
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Security hardening (production only)
# ─────────────────────────────────────────────────────────────────────────────

if not DEBUG:
    SECURE_HSTS_SECONDS            = 31_536_000   # 1 year
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_PRELOAD            = True
    SECURE_SSL_REDIRECT            = True
    SESSION_COOKIE_SECURE          = True
    CSRF_COOKIE_SECURE             = True
    SECURE_BROWSER_XSS_FILTER      = True
    SECURE_CONTENT_TYPE_NOSNIFF    = True
    X_FRAME_OPTIONS                = "DENY"