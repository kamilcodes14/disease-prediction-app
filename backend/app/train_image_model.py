"""
Train the demo image-screening model.

IMPORTANT / HONESTY NOTE
------------------------
There is no clinical imaging dataset (e.g. HAM10000, ISIC, chest X-ray
archives) reachable from this build environment, and no way to download
pretrained ImageNet weights either. Rather than fake a deep-learning model
we can't actually train, this module implements a transparent, classical
computer-vision pipeline built on real dermoscopy heuristics (the ABCD
rule: Asymmetry, Border irregularity, Color variegation, Diameter):

    1. Extract real image features (color stats, edge/border density via
       Canny, texture via Local Binary Patterns, left/right and top/bottom
       asymmetry).
    2. Train a Random Forest on a procedurally generated proxy dataset of
       "regular / benign-like" vs "irregular / suspicious-like" patches,
       since no real labeled clinical images are available here.

This is a genuine, working, end-to-end pipeline -- but it is a prototype
scaffold, not a clinically validated diagnostic model. Swap step 2's
training data for a real, ethically-sourced, labeled medical image dataset
(with a proper train/val/test split and clinician-verified labels) before
this ever touches a real diagnostic decision. See README for details.
"""
import os
import random

import joblib
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from image_features import extract_features, FEATURE_NAMES

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

random.seed(7)
np.random.seed(7)

IMG_SIZE = 128


def _base_skin(rng):
    """A smooth-ish base skin tone patch."""
    hue = rng.integers(15, 30)
    base = np.array(
        [
            210 + rng.integers(-15, 15),
            160 + rng.integers(-15, 15),
            130 + rng.integers(-15, 15),
        ]
    ).clip(0, 255)
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), tuple(int(v) for v in base))
    return img


def make_regular_patch(rng):
    """Benign-like: smooth, symmetric, single-tone, soft round edge."""
    img = _base_skin(rng)
    draw = ImageDraw.Draw(img)
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    r = rng.integers(20, 30)
    shade = tuple(int(c * 0.75) for c in img.getpixel((0, 0)))
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=shade)
    img = img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(2.5, 4))))
    # mild sensor noise
    arr = np.array(img).astype(np.int16)
    arr += rng.integers(-4, 4, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def make_irregular_patch(rng):
    """Suspicious-like: asymmetric blob, jagged border, mixed colors."""
    img = _base_skin(rng)
    draw = ImageDraw.Draw(img)
    cx, cy = IMG_SIZE // 2 + rng.integers(-10, 10), IMG_SIZE // 2 + rng.integers(-10, 10)

    n_points = rng.integers(9, 14)
    angles = np.sort(rng.uniform(0, 2 * np.pi, n_points))
    base_r = rng.integers(22, 34)
    points = []
    for a in angles:
        r = base_r * rng.uniform(0.55, 1.45)
        points.append((cx + r * np.cos(a), cy + r * np.sin(a)))

    palette = [
        tuple(int(c) for c in np.clip(
            np.array(img.getpixel((0, 0))) * rng.uniform(0.35, 0.9), 0, 255
        ))
        for _ in range(3)
    ]
    draw.polygon(points, fill=random.choice(palette))
    # scatter a couple of extra color blotches for variegation
    for _ in range(rng.integers(1, 3)):
        bx, by = cx + rng.integers(-18, 18), cy + rng.integers(-18, 18)
        br = rng.integers(4, 9)
        draw.ellipse([bx - br, by - br, bx + br, by + br], fill=random.choice(palette))

    img = img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.6, 1.4))))
    arr = np.array(img).astype(np.int16)
    arr += rng.integers(-10, 10, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def build_dataset(n_per_class=260):
    rng = np.random.default_rng(7)
    X, y = [], []
    for _ in range(n_per_class):
        img = make_regular_patch(rng)
        X.append(extract_features(img))
        y.append(0)
    for _ in range(n_per_class):
        img = make_irregular_patch(rng)
        X.append(extract_features(img))
        y.append(1)
    return np.array(X), np.array(y)


def main():
    X, y = build_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=7, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=7)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Proxy validation accuracy: {acc:.3f}  (n_test={len(y_test)})")
    print("Feature importances:")
    for name, imp in sorted(
        zip(FEATURE_NAMES, clf.feature_importances_), key=lambda t: -t[1]
    ):
        print(f"  {name:22s} {imp:.3f}")

    joblib.dump(clf, os.path.join(MODEL_DIR, "image_screen_model.joblib"))
    print("\nSaved ->", os.path.join(MODEL_DIR, "image_screen_model.joblib"))


if __name__ == "__main__":
    main()
