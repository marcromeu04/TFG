"""Three feature representations for NMR spectra: bins300, ROI stats, and full-spectrum."""
from __future__ import annotations

import logging

import numpy as np

from config import N_BINS_DEFAULT, WATER_MIN, WATER_MAX
from data.metabolite_rois import active_rois

log = logging.getLogger(__name__)


def to_bins300(X: np.ndarray,
               ppm: np.ndarray,
               n_bins: int = N_BINS_DEFAULT) -> tuple[np.ndarray, np.ndarray]:
    """Equispaced binning of spectra to `n_bins` bins. Returns (X_binned, bin_centres_ppm)."""
    if ppm[0] > ppm[-1]:
        # ppm descending (common in NMR) - flip both for binning
        ppm = ppm[::-1]
        X = X[:, ::-1]
    edges = np.linspace(ppm.min(), ppm.max(), n_bins + 1)
    bin_idx = np.clip(np.searchsorted(edges, ppm, side="right") - 1,
                      0, n_bins - 1)
    X_binned = np.zeros((X.shape[0], n_bins), dtype=np.float32)
    counts = np.zeros(n_bins, dtype=np.int64)
    for k in range(n_bins):
        sel = bin_idx == k
        if sel.any():
            X_binned[:, k] = X[:, sel].mean(axis=1)
            counts[k] = int(sel.sum())
    bin_centres = 0.5 * (edges[:-1] + edges[1:])
    log.debug("Binned %d -> %d (min count per bin: %d)",
              X.shape[1], n_bins, int(counts.min()))
    return X_binned, bin_centres.astype(np.float32)


def _ppm_slice(ppm: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (ppm >= lo) & (ppm <= hi)


def to_roi(X: np.ndarray,
           ppm: np.ndarray,
           water_min: float = WATER_MIN,
           water_max: float = WATER_MAX
           ) -> tuple[np.ndarray, list[tuple[str, str, float, float]]]:
    """Extract 3 statistics per active ROI (sum, max, peak-baseline ratio).
    Returns (F, feature_meta) where feature_meta tuples are (metabolite, stat, ppm_lo, ppm_hi)."""
    if ppm[0] > ppm[-1]:
        ppm = ppm[::-1]
        X = X[:, ::-1]

    rois = active_rois(water_min, water_max)
    n = X.shape[0]
    feats: list[np.ndarray] = []
    meta: list[tuple[str, str, float, float]] = []

    # Local "baseline" per ROI: median of a small neighbourhood just outside the ROI.
    BAND_WIDTH_PPM = 0.05    # ppm on each side of the ROI

    for met, lo, hi, _comment in rois:
        sel = _ppm_slice(ppm, lo, hi)
        if not sel.any():
            zeros = np.zeros(n, dtype=np.float32)
            feats.extend([zeros, zeros, zeros])
            meta.extend([
                (met, "sum",   lo, hi),
                (met, "max",   lo, hi),
                (met, "ratio", lo, hi),
            ])
            continue

        roi_X = X[:, sel]
        roi_sum = roi_X.sum(axis=1).astype(np.float32)
        roi_max = roi_X.max(axis=1).astype(np.float32)

        left_band = _ppm_slice(ppm, lo - BAND_WIDTH_PPM, lo) & ~sel
        right_band = _ppm_slice(ppm, hi, hi + BAND_WIDTH_PPM) & ~sel
        band = left_band | right_band
        if band.any():
            base = np.median(X[:, band], axis=1).astype(np.float32)
        else:
            base = np.full(n, 1e-9, dtype=np.float32)
        base = np.maximum(base, 1e-12)  # avoid div-by-zero
        roi_ratio = (roi_max / base).astype(np.float32)

        feats.extend([roi_sum, roi_max, roi_ratio])
        meta.extend([
            (met, "sum",   lo, hi),
            (met, "max",   lo, hi),
            (met, "ratio", lo, hi),
        ])

    F = np.stack(feats, axis=1).astype(np.float32)
    log.debug("ROI features: %s (active ROIs: %d)", F.shape, len(rois))
    return F, meta


def to_fullspec(X: np.ndarray, ppm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Identity feature representation; returns (X, ppm) cast to float32."""
    return X.astype(np.float32), ppm.astype(np.float32)
