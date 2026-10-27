#!/usr/bin/env bash
# Start the Melbourne property valuation website (API + frontend on one port).
set -euo pipefail

cd "$(dirname "$0")/backend"

PORT="${PORT:-8000}"

if [ ! -f artifacts.joblib ]; then
  echo "Building preprocessing artifacts (first run)…"
  python preprocess.py
fi

echo "Starting MELB/VALUATION on http://localhost:${PORT}"
PORT="$PORT" python app.py
