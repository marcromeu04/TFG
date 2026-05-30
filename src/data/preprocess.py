"""Spectrum preprocessing: water-region exclusion (4.65-4.95 ppm) and integral normalisation."""
from __future__ import annotations

import logging

import numpy as np

from config import WATER_MIN, WATER_MAX

log = logging.getLogger(__name__)


def _water_mask(ppm: np.ndarray, water_min: float, water_max: float) -> np.ndarray:
    """True where the ppm point is OUTSIDE the water region."""
    # ojo: returns True OUTSIDE the band, not inside
    return (ppm < water_min) | (ppm > water_max)


def _normalize_integral(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Divide each row by sum of absolute values (unit L1 norm per row)."""
    s = np.sum(np.abs(X), axis=1, keepdims=True)
    s = np.maximum(s, eps)
    return X / s


def preprocess_spectra(X: np.ndarray,
                       ppm: np.ndarray,
                       *,
                       exclude_water: bool = False,
                       water_min: float = WATER_MIN,
                       water_max: float = WATER_MAX,
                       normalize: bool = False
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Apply standard preprocessing: optional water exclusion and integral normalisation."""
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n_samples, n_ppm); got {X.shape}")
    if ppm.shape != (X.shape[1],):
        raise ValueError(
            f"ppm length {ppm.shape} does not match X.shape[1]={X.shape[1]}"
        )

    if exclude_water:
        mask = _water_mask(ppm, water_min, water_max)
        if not mask.all():
            X = X[:, mask]
            ppm = ppm[mask]
            log.info("Excluded water region [%.2f, %.2f] ppm: %d points "
                     "removed, %d remain", water_min, water_max,
                     int((~mask).sum()), int(mask.sum()))

    if normalize:
        X = _normalize_integral(X.astype(np.float64)).astype(np.float32)

    return X, ppm


def hash_dataset(X: np.ndarray, Y: np.ndarray, ppm: np.ndarray) -> str:
    """Stable MD5 of (X, Y, ppm) for reproducibility manifests; rounds to 6 dp first."""
    import hashlib
    h = hashlib.md5()
    h.update(np.round(X.astype(np.float64), 6).tobytes())
    Y_safe = np.where(np.isnan(Y), -999.0, Y).astype(np.float64)
    h.update(np.round(Y_safe, 6).tobytes())
    h.update(np.round(ppm.astype(np.float64), 6).tobytes())
    return h.hexdigest()
