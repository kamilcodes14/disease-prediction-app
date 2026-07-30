import io
import json
import os

import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

from image_features import extract_features

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")

app = FastAPI(
    title="CodeAlpha Disease Prediction API",
    description="Structured-data + image-based disease screening (Task 4)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your deployed frontend origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

DATASETS = ["heart_disease", "diabetes", "breast_cancer"]

_models, _scalers, _metadata = {}, {}, {}
for key in DATASETS:
    _models[key] = joblib.load(os.path.join(MODEL_DIR, f"{key}_model.joblib"))
    _scalers[key] = joblib.load(os.path.join(MODEL_DIR, f"{key}_scaler.joblib"))
    with open(os.path.join(MODEL_DIR, f"{key}_metadata.json")) as f:
        _metadata[key] = json.load(f)

_image_model = joblib.load(os.path.join(MODEL_DIR, "image_screen_model.joblib"))


class StructuredPredictRequest(BaseModel):
    dataset: str = Field(..., description="one of: heart_disease, diabetes, breast_cancer")
    features: dict[str, float] = Field(..., description="feature_name -> value")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/datasets")
def list_datasets():
    """Feature schema + model info for every structured dataset, so the
    frontend can render the right form fields dynamically."""
    return {key: _metadata[key] for key in DATASETS}


@app.post("/api/predict/structured")
def predict_structured(req: StructuredPredictRequest):
    if req.dataset not in DATASETS:
        raise HTTPException(400, f"dataset must be one of {DATASETS}")

    meta = _metadata[req.dataset]
    order = meta["feature_names"]
    missing = [f for f in order if f not in req.features]
    if missing:
        raise HTTPException(400, f"missing features: {missing}")

    x = np.array([[req.features[f] for f in order]])
    x_scaled = _scalers[req.dataset].transform(x)
    model = _models[req.dataset]

    proba = float(model.predict_proba(x_scaled)[0, 1])
    prediction = int(proba >= 0.5)

    return {
        "dataset": req.dataset,
        "display_name": meta["display_name"],
        "positive_label": meta["positive_label"],
        "prediction": prediction,
        "probability": round(proba, 4),
        "risk_level": "high" if proba >= 0.66 else "moderate" if proba >= 0.33 else "low",
        "model_used": meta["best_model"],
    }


@app.post("/api/predict/image")
async def predict_image(file: UploadFile = File(...)):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, "please upload an image file")

    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception:
        raise HTTPException(400, "could not read image file")

    feats = extract_features(img)
    proba = float(_image_model.predict_proba([feats])[0, 1])
    prediction = int(proba >= 0.5)

    names = [
        "mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b",
        "color_variegation", "edge_density", "lbp_entropy",
        "asymmetry_lr", "asymmetry_tb", "relative_diameter",
    ]
    feature_report = {n: round(float(v), 4) for n, v in zip(names, feats)}

    return {
        "prediction": prediction,
        "probability": round(proba, 4),
        "risk_level": "high" if proba >= 0.66 else "moderate" if proba >= 0.33 else "low",
        "label": "Irregular pattern detected" if prediction else "Regular pattern",
        "feature_report": feature_report,
        "disclaimer": (
            "Prototype screening pipeline using classical computer-vision "
            "features (ABCD-rule heuristics), not a clinically validated "
            "diagnostic model. Educational/portfolio use only."
        ),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
