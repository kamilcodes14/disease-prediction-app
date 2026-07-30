"""
Train classical ML models for structured-data disease prediction.

Datasets:
  - Heart Disease   (UCI Cleveland, 303 patients, 13 clinical features)
  - Diabetes        (Pima Indians Diabetes, 768 patients, 8 clinical features)
  - Breast Cancer   (UCI/Wisconsin Diagnostic, 569 patients, 30 cell-nuclei features
                      via sklearn's built-in loader)

For each dataset we train Logistic Regression, SVM (RBF), Random Forest and
XGBoost, evaluate on a held-out test split, and persist the best performer
(plus the scaler + feature metadata needed to serve live predictions).
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
MODEL_DIR = os.path.join(HERE, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42


def candidate_models():
    return {
        "logistic_regression": LogisticRegression(max_iter=2000),
        "svm_rbf": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
    }


def evaluate_and_select(X, y, feature_names, dataset_key, display_name, positive_label):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}
    fitted = {}
    for name, model in candidate_models().items():
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        proba = model.predict_proba(X_test_s)[:, 1]
        results[name] = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "f1": round(f1_score(y_test, preds), 4),
            "roc_auc": round(roc_auc_score(y_test, proba), 4),
        }
        fitted[name] = model

    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    best_model = fitted[best_name]

    joblib.dump(best_model, os.path.join(MODEL_DIR, f"{dataset_key}_model.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"{dataset_key}_scaler.joblib"))

    metadata = {
        "dataset_key": dataset_key,
        "display_name": display_name,
        "positive_label": positive_label,
        "feature_names": feature_names,
        "best_model": best_name,
        "leaderboard": results,
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
    }
    with open(os.path.join(MODEL_DIR, f"{dataset_key}_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n=== {display_name} ===")
    for name, m in sorted(results.items(), key=lambda kv: -kv[1]["roc_auc"]):
        marker = " <-- selected" if name == best_name else ""
        print(f"  {name:20s} acc={m['accuracy']:.3f} f1={m['f1']:.3f} auc={m['roc_auc']:.3f}{marker}")

    return metadata


def train_heart_disease():
    path = os.path.join(DATA_DIR, "heart.csv")
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]
    feature_names = [c for c in df.columns if c != "target"]
    X = df[feature_names].values.astype(float)
    y = df["target"].astype(int).values
    return evaluate_and_select(
        X, y, feature_names, "heart_disease", "Heart Disease", "Disease present"
    )


def train_diabetes():
    path = os.path.join(DATA_DIR, "diabetes.csv")
    df = pd.read_csv(path)
    feature_names = [c for c in df.columns if c != "Outcome"]
    X = df[feature_names].values.astype(float)
    y = df["Outcome"].astype(int).values
    return evaluate_and_select(
        X, y, feature_names, "diabetes", "Diabetes", "Diabetic"
    )


def train_breast_cancer():
    data = load_breast_cancer()
    feature_names = list(data.feature_names)
    X = data.data
    # sklearn encodes 0=malignant, 1=benign -> flip so 1 == "disease present" (malignant)
    y = (data.target == 0).astype(int)
    return evaluate_and_select(
        X, y, feature_names, "breast_cancer", "Breast Cancer", "Malignant"
    )


if __name__ == "__main__":
    summary = {
        "heart_disease": train_heart_disease(),
        "diabetes": train_diabetes(),
        "breast_cancer": train_breast_cancer(),
    }
    with open(os.path.join(MODEL_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nAll models trained and saved to", MODEL_DIR)
