# Single-domain setup: this file lives at repo-root/api/index.py so Vercel
# serves it under /api/*. Your real FastAPI app (with its own /api/... routes)
# stays untouched in backend/app/main.py — this just points at it.
import os
import sys

APP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend", "app")
sys.path.insert(0, APP_DIR)

from main import app  # noqa: E402  (backend/app/main.py's FastAPI instance)
