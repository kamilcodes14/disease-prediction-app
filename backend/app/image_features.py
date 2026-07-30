"""
Classical computer-vision feature extraction for the image-screening demo.

Implements measurable proxies for the dermoscopy "ABCD rule" used as a
teaching heuristic for pigmented skin lesions:
    A - Asymmetry
    B - Border irregularity
    C - Color variegation
    D - Diameter / size (relative to frame)

Every feature below is computed directly from real pixel data (no
placeholders) using PIL + NumPy + scikit-image.
"""
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import canny, local_binary_pattern

IMG_SIZE = 128

FEATURE_NAMES = [
    "mean_r", "mean_g", "mean_b",
    "std_r", "std_g", "std_b",
    "color_variegation",
    "edge_density",
    "lbp_entropy",
    "asymmetry_lr",
    "asymmetry_tb",
    "relative_diameter",
]


def _prep(img: Image.Image) -> Image.Image:
    return img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))


def _lbp_entropy(gray: np.ndarray) -> float:
    gray_u8 = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    lbp = local_binary_pattern(gray_u8, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    hist = hist[hist > 0]
    return float(-(hist * np.log2(hist)).sum())


def _foreground_mask(gray: np.ndarray) -> np.ndarray:
    """Rough lesion/region-of-interest mask via Otsu-style thresholding
    against the border-median background tone."""
    border = np.concatenate(
        [gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]]
    )
    bg_level = np.median(border)
    mask = np.abs(gray - bg_level) > (0.08 * gray.std() + 0.05)
    if mask.sum() < 25:  # fallback: treat whole frame as region of interest
        mask = np.ones_like(gray, dtype=bool)
    return mask


def extract_features(img: Image.Image) -> np.ndarray:
    img = _prep(img)
    arr = np.asarray(img).astype(np.float32)
    gray = rgb2gray(arr / 255.0)

    mean_rgb = arr.reshape(-1, 3).mean(axis=0)
    std_rgb = arr.reshape(-1, 3).std(axis=0)

    mask = _foreground_mask(gray)
    region_pixels = arr[mask]
    color_variegation = float(region_pixels.std(axis=0).mean()) if len(region_pixels) else 0.0

    edges = canny(gray, sigma=1.5)
    edge_density = float(edges.mean())

    lbp_entropy = _lbp_entropy(gray)

    h, w = mask.shape
    left, right = mask[:, : w // 2], np.fliplr(mask[:, w // 2:])
    top, bottom = mask[: h // 2, :], np.flipud(mask[h // 2:, :])
    min_w = min(left.shape[1], right.shape[1])
    min_h = min(top.shape[0], bottom.shape[0])
    asym_lr = float(np.logical_xor(left[:, :min_w], right[:, :min_w]).mean())
    asym_tb = float(np.logical_xor(top[:min_h, :], bottom[:min_h, :]).mean())

    relative_diameter = float(mask.mean())

    return np.array(
        [
            mean_rgb[0], mean_rgb[1], mean_rgb[2],
            std_rgb[0], std_rgb[1], std_rgb[2],
            color_variegation,
            edge_density,
            lbp_entropy,
            asym_lr,
            asym_tb,
            relative_diameter,
        ],
        dtype=np.float32,
    )
