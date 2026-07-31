import io
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import joblib

from image_features import extract_features

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")

app = FastAPI(
    title="CodeAlpha Disease Prediction API",
    description="Image-based disease screening (structured predictions now run client-side via ONNX)",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_image_model = joblib.load(os.path.join(MODEL_DIR, "image_screen_model.joblib"))


@app.get("/api/health")
def health():
    return {"status": "ok"}


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