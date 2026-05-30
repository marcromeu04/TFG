"""Three data-augmentation strategies for low-N regression: bootstrap, jitter, mixup.
All are pure: (X, Y, target_size, seed, **kwargs) -> (X_aug, Y_aug). NaN labels are handled per-column."""
from __future__ import annotations

import logging

import numpy as np

from config import JITTER_SCALE, MIXUP_ALPHA, TARGET_AUG_SIZE

log = logging.getLogger(__name__)


def augment_bootstrap(X: np.ndarray,
                      Y: np.ndarray,
                      target_size: int = TARGET_AUG_SIZE,
                      seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap sample with replacement to `target_size`. Reference, not real augmentation"""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, X.shape[0], size=target_size)
    return X[idx], Y[idx]


def augment_jitter(X: np.ndarray,
                   Y: np.ndarray,
                   target_size: int = TARGET_AUG_SIZE,
                   seed: int = 0,
                   scale: float = JITTER_SCALE
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap + Gaussian jitter (scale * per-feature-std) on features."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, X.shape[0], size=target_size)
    X_aug = X[idx].astype(np.float64)
    feat_std = np.maximum(X.std(axis=0), 1e-6)
    noise = rng.normal(0, scale, size=X_aug.shape) * feat_std
    X_aug = X_aug + noise
    return X_aug.astype(X.dtype), Y[idx]


def augment_mixup(X: np.ndarray,
                  Y: np.ndarray,
                  target_size: int = TARGET_AUG_SIZE,
                  seed: int = 0,
                  alpha: float = MIXUP_ALPHA
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Mixup: linear interpolation of pairs of samples with lam ~ Beta(alpha, alpha).
    NaN handling per metabolite: mix when both endpoints valid, else copy the valid one."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    n_met = Y.shape[1]
    idx_a = rng.integers(0, n, size=target_size)
    idx_b = rng.integers(0, n, size=target_size)
    lam = rng.beta(alpha, alpha, size=target_size)
    lam = np.maximum(lam, 1 - lam)        # ensure lam >= 0.5 for stability
    lam_X = lam[:, None]

    X_aug = (lam_X * X[idx_a].astype(np.float64) +
             (1 - lam_X) * X[idx_b].astype(np.float64))

    Y_aug = np.full((target_size, n_met), np.nan, dtype=np.float64)
    for k in range(n_met):
        ya = Y[idx_a, k]
        yb = Y[idx_b, k]
        va = ~np.isnan(ya)
        vb = ~np.isnan(yb)
        both = va & vb
        Y_aug[both, k] = lam[both] * ya[both] + (1 - lam[both]) * yb[both]
        only_a = va & ~vb
        Y_aug[only_a, k] = ya[only_a]
        only_b = vb & ~va
        Y_aug[only_b, k] = yb[only_b]

    return X_aug.astype(X.dtype), Y_aug


AUGMENTATION_REGISTRY = {
    "baseline":  None,                  # special: no augmentation
    "bootstrap": augment_bootstrap,
    "jitter":    augment_jitter,
    "mixup":     augment_mixup,
}


def apply_augmentation(strategy: str,
                       X: np.ndarray,
                       Y: np.ndarray,
                       target_size: int = TARGET_AUG_SIZE,
                       seed: int = 0,
                       **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Apply the named augmentation strategy. `baseline` is identity."""
    if strategy == "baseline":
        return X, Y
    if strategy not in AUGMENTATION_REGISTRY:
        raise KeyError(f"Unknown strategy {strategy}; "
                       f"available: {list(AUGMENTATION_REGISTRY.keys())}")
    fn = AUGMENTATION_REGISTRY[strategy]
    return fn(X, Y, target_size=target_size, seed=seed, **kwargs)
