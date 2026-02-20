#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Sync with remote — always reset to what's on git (discards local changes)
if [[ -d .git ]]; then
    BRANCH="$(git rev-parse --abbrev-ref HEAD)"
    echo "Syncing with origin/${BRANCH} ..."
    git fetch origin
    git reset --hard "origin/${BRANCH}"
fi

# Load .env if present
[[ -f .env ]] && { set -a; source .env; set +a; }

# Check requirements
command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found"; exit 1; }
if ! docker info >/dev/null 2>&1; then
    if docker info 2>&1 | grep -q "permission denied"; then
        echo "ERROR: Permission denied on Docker socket."
        echo "       Add your user to the docker group:  sudo usermod -aG docker \$USER"
        echo "       Then log out and back in (or run:   newgrp docker)"
    else
        echo "ERROR: Docker daemon is not running. Start it with:  sudo systemctl start docker"
    fi
    exit 1
fi

# Activate venv
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/venv}"
[[ -d "$VENV_PATH" ]] || { echo "ERROR: venv not found at $VENV_PATH — run 'python -m venv venv' first"; exit 1; }
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"

mkdir -p data
pip install --quiet -r requirements.txt

# Initialize DB with admin user if fresh start
[[ -f data/app.db ]] || python setup.py --create-admin

#python -m alembic upgrade head
export RUN_MIGRATIONS_ON_STARTUP=false

HOST="${API_HOST:-0.0.0.0}"
PORT="${API_PORT:-8000}"
WORKERS="${UVICORN_WORKERS:-1}"

echo "Starting API on ${HOST}:${PORT} with ${WORKERS} worker(s)..."
exec python -m uvicorn main:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS" \
    --log-level "${UVICORN_LOG_LEVEL:-info}"
