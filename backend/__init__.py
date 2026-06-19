"""
backend/__init__.py
Marks `backend` as a Python package.

Django 5.x auto-discovers the AppConfig subclass defined in
backend/apps.py (no need for the legacy `default_app_config` variable,
which was removed in Django 3.2+). As long as INSTALLED_APPS contains
"backend" (set in settings.py) and apps.py defines exactly one
AppConfig subclass, Django uses it automatically and calls its
.ready() hook once at process startup — which is what loads the PPO
checkpoint via AgentService.get_instance() before the server accepts
any requests.
"""