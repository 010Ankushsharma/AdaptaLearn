
#!/usr/bin/env bash
# setup.sh — run once to bootstrap the project environment
set -e

echo "==> Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "==> Upgrading pip..."
pip install --upgrade pip

echo "==> Installing dependencies..."
pip install -r requirements.txt

echo "==> Creating .env from template..."
if [ ! -f .env ]; then
  cp .env.example .env
  echo "    .env created — fill in your DATABASE_URL and REDIS_URL"
fi

echo "==> Creating empty __init__.py files..."
touch data/__init__.py
touch models/__init__.py
touch rl/__init__.py
touch backend/__init__.py
touch analysis/__init__.py

echo ""
echo "Done! Activate the env with: source venv/bin/activate"