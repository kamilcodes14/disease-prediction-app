# Vercel looks for api/*.py exporting an ASGI `app` variable.
# Your real FastAPI app lives in backend/app/main.py — this file just
# points at it so Vercel's zero-config Python runtime can find it,
# without moving or rewriting your actual application code.
import os
import sys

APP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app")
sys.path.insert(0, APP_DIR)

from main import app  # noqa: E402  (backend/app/main.py's FastAPI instance)
