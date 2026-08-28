"""
Hand-crafted image quality features.

18 features covering sharpness, exposure, contrast, noise, color, and structure.
Used by the per-issue classifiers and for explainability.
"""
from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass, asdict


@dataclass
class ImageFeatures:
    # sharpness
    laplacian_variance: float
    high_freq_energy: float

    # exposure
    mean_brightness: float
    brightness_std: float
    frac_underexposed: float   # pixels < 20
    frac_overexposed: float    # pixels > 235
    dynamic_range: float       # p99 - p1

    # contrast
    michelson_contrast: float
    rms_contrast: float

    # noise
    noise_sigma_estimate: float
    local_variance_mean: float

    # color
    mean_saturation: float
    saturation_std: float

    # structure
    edge_density: float
    block_artifact_score: float
    entropy: float
    colorfulness: float
    texture_energy: float

    def to_vector(self) -> np.ndarray:
        return np.array(list(asdict(self).values()), dtype=np.float64)

    @staticmethod
    def names() -> list[str]:
        return list(ImageFeatures.__dataclass_fields__.keys())


def _laplacian_variance(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _high_freq_energy(gray):
    """Fraction of spectral energy in high frequencies (FFT-based)."""
    f = np.fft.fft2(gray.astype(np.float64))
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 8
    yy, xx = np.ogrid[:h, :w]
    mask_low = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
    total = mag.sum() + 1e-8
    return float(mag[~mask_low].sum() / total)


def _noise_sigma_immerkaer(gray):
    """Immerkaer (1996) noise std estimator."""
    H, W = gray.shape
    M = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    conv = cv2.filter2D(gray.astype(np.float64), -1, M)
    sigma = np.sqrt(np.pi / 2) * (np.abs(conv).sum()) / (6 * max(W - 2, 1) * max(H - 2, 1))
    return float(sigma)


def _block_artifact_score(gray, block=8):
    """Ratio of discontinuity at 8x8 block boundaries vs interior.
    High values suggest JPEG compression artifacts."""
    g = gray.astype(np.float64)
    h, w = g.shape
    h2, w2 = (h // block) * block, (w // block) * block
    g = g[:h2, :w2]
    if h2 < block * 2 or w2 < block * 2:
        return 0.0

    dx = np.abs(np.diff(g, axis=1))
    dy = np.abs(np.diff(g, axis=0))
    col_idx = np.arange(dx.shape[1])
    row_idx = np.arange(dy.shape[0])
    boundary_cols = (col_idx + 1) % block == 0
    boundary_rows = (row_idx + 1) % block == 0

    boundary_energy = dx[:, boundary_cols].mean() + dy[boundary_rows, :].mean()
    interior_energy = dx[:, ~boundary_cols].mean() + dy[~boundary_rows, :].mean() + 1e-6
    return float(boundary_energy / interior_energy)


def _colorfulness(bgr):
    """Hasler & Süsstrunk (2003) colorfulness metric."""
    B, G, R = bgr[:, :, 0].astype(np.float64), bgr[:, :, 1].astype(np.float64), bgr[:, :, 2].astype(np.float64)
    rg = R - G
    yb = 0.5 * (R + G) - B
    std_root = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean_root = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float(std_root + 0.3 * mean_root)


def _texture_energy(gray):
    """Energy of LoG - proxy for texture richness."""
    blurred = cv2.GaussianBlur(gray.astype(np.float64), (5, 5), 1.0)
    log = cv2.Laplacian(blurred, cv2.CV_64F)
    return float(np.mean(log ** 2))


def extract_features(bgr: np.ndarray) -> ImageFeatures:
    """Extract all 18 features from a BGR image."""
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    g = gray.astype(np.float64)
    p99, p1 = np.percentile(g, [99, 1])
    gmax, gmin = g.max(), g.min()

    # local variance (5x5 window)
    k = 5
    mean_local = cv2.boxFilter(g, ddepth=-1, ksize=(k, k))
    mean_sq_local = cv2.boxFilter(g * g, ddepth=-1, ksize=(k, k))
    local_var = np.clip(mean_sq_local - mean_local ** 2, 0, None)

    edges = cv2.Canny(gray, 100, 200)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-8)

    sat = hsv[:, :, 1].astype(np.float64)

    return ImageFeatures(
        laplacian_variance=_laplacian_variance(gray),
        high_freq_energy=_high_freq_energy(gray),
        mean_brightness=float(g.mean()),
        brightness_std=float(g.std()),
        frac_underexposed=float(np.mean(gray < 20)),
        frac_overexposed=float(np.mean(gray > 235)),
        dynamic_range=float(p99 - p1),
        michelson_contrast=float((gmax - gmin) / (gmax + gmin + 1e-6)),
        rms_contrast=float(g.std() / (g.mean() + 1e-6)),
        noise_sigma_estimate=_noise_sigma_immerkaer(gray),
        local_variance_mean=float(np.mean(local_var)),
        mean_saturation=float(sat.mean()),
        saturation_std=float(sat.std()),
        edge_density=float(np.mean(edges > 0)),
        block_artifact_score=_block_artifact_score(gray),
        entropy=float(-np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))),
        colorfulness=_colorfulness(bgr),
        texture_energy=_texture_energy(gray),
    )
