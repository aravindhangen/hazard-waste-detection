#!/usr/bin/env sh
set -eu

PORT="${PORT:-8000}"

echo "Starting Hazard Waste Detection on port ${PORT} (device=${HAZARD_DEVICE:-cpu})"
echo "  Dashboard: /dashboard/"
echo "  Health:    /health"

exec python -m uvicorn api.main:app --host 0.0.0.0 --port "${PORT}"
