#!/usr/bin/env bash
# Run this every time you want to use the app.
# Starts the backend API and the frontend, then opens your browser.
cd "$(dirname "$0")"

if [ ! -d "backend/venv" ]; then
  echo "⚠️  Setup hasn't been run yet. Run: bash setup.sh"
  exit 1
fi

echo "==> Starting backend on http://localhost:8000 ..."
(cd backend/app && ../venv/bin/python main.py) &
BACKEND_PID=$!

sleep 2

echo "==> Starting frontend on http://localhost:5500 ..."
(cd frontend && python3 -m http.server 5500) &
FRONTEND_PID=$!

echo ""
echo "✅ Both servers running."
echo "   Open: http://localhost:5500"
echo "   Press Ctrl+C to stop both."

# Open the browser automatically where possible
( sleep 1 && (open http://localhost:5500 2>/dev/null || xdg-open http://localhost:5500 2>/dev/null || true) ) &

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT
wait
