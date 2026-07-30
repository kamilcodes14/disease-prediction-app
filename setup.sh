#!/usr/bin/env bash
# Run this ONCE: sets up everything needed to run the app.
set -e
cd "$(dirname "$0")"

echo "==> Creating virtual environment..."
python3 -m venv backend/venv

echo "==> Installing backend dependencies..."
backend/venv/bin/pip install --upgrade pip -q
backend/venv/bin/pip install -r backend/requirements.txt -q

echo "==> Training the structured-data models (Heart Disease / Diabetes / Breast Cancer)..."
(cd backend/app && ../venv/bin/python train_structured.py)

echo "==> Training the photo-screening model..."
(cd backend/app && ../venv/bin/python train_image_model.py)

echo ""
echo "✅ Setup complete. Now run:   bash run.sh"
