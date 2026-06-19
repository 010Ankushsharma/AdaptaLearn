#!/usr/bin/env bash
# build.sh
# Render build step — runs once per deploy, before the web service starts.
#
# This script is referenced as the "Build Command" in render.yaml.
# It must be executable: `chmod +x build.sh` before committing (git
# preserves the executable bit, but double-check after cloning).
#
# Steps:
#   1. Install production Python dependencies (slim requirements-prod.txt,
#      NOT the full requirements.txt with training-only packages)
#   2. Collect static files into staticfiles/ (served by WhiteNoise)
#   3. Apply database migrations
#
# Render runs this from the project root (the directory containing
# manage.py), so all paths below are relative to that.

set -o errexit   # exit immediately if any command fails
set -o nounset   # treat unset variables as an error
set -o pipefail  # a pipeline fails if any command in it fails

echo "── Installing production dependencies ──"
pip install --upgrade pip
pip install -r requirements-prod.txt

echo "── Collecting static files ──"
python manage.py collectstatic --noinput

echo "── Applying database migrations ──"
python manage.py migrate --noinput

echo "── Build complete ✓ ──"