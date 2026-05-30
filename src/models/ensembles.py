"""Ensemble strategies: average_topk, average_all, bagging, and CCC-weighted blend.
Stacking meta-learners are in models/meta_learners.py."""
from __future__ import annotations

import logging
from copy import deepcopy
from typing import Callable, Optional

import numpy as np

from config import N_JOBS_INNER, SEED

log = logging.getLogger(__name__)


def average_topk(base_preds: np.ndarray,
                 base_scores: np.ndarray,
                 k: int = 3) -> np.ndarray:
    """Plain mean of the top-K base models, ranked by score per metabolite.
    base_preds: (n, M, n_met); base_scores: (M, n_met), higher=better."""
    n, M, n_met = base_preds.shape
    out = np.zeros((n, n_met), dtype=np.float32)
    for j in range(n_met):
        order = np.argsort(-base_scores[:, j])[:k]    # descending
        out[:, j] = base_preds[:, order, j].mean(axis=1)
    return out


def average_all(base_preds: np.ndarray) -> np.ndarray:
    """Plain mean of all base models."""
    return base_preds.mean(axis=1)


def bagging_predict(model_factory: Callable,
                    X_train: np.ndarray,
                    Y_train: np.ndarray,
                    X_test: np.ndarray,
                    n_estimators: int = 50,
                    seed: int = SEED,
                    use_scaler: bool = True) -> np.ndarray:
    """Bagging: train n_estimators copies on bootstrap samples, average predictions on X_test.
    StandardScaler is refit per bootstrap (avoids leakage from a global scaler over duplicates).
    Predictions are not clipped to >=0 here (clipping breaks rank stability)."""
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    n_train = X_train.shape[0]
    n_met = Y_train.shape[1]
    n_test = X_test.shape[0]
    preds_sum = np.zeros((n_test, n_met), dtype=np.float64)
    preds_count = np.zeros((n_test, n_met), dtype=np.int64)

    for b in range(n_estimators):
        boot_idx = rng.integers(0, n_train, size=n_train)
        Xb = X_train[boot_idx]
        Yb = Y_train[boot_idx]
        if use_scaler:
            sc = StandardScaler().fit(Xb)
            Xb_s = sc.transform(Xb)
            X_test_s = sc.transform(X_test)
        else:
            Xb_s = Xb
            X_test_s = X_test
        for k in range(n_met):
            mask = ~np.isnan(Yb[:, k])
            if mask.sum() < 10:
                continue
            try:
                m = model_factory()
                m.fit(Xb_s[mask], Yb[mask, k])
                p = m.predict(X_test_s)
                preds_sum[:, k] += p
                preds_count[:, k] += 1
            except Exception as e:
                log.debug("bagging fit failed (b=%d, met=%d): %s", b, k, e)

    out = np.where(preds_count > 0,
                   preds_sum / np.maximum(preds_count, 1),
                   np.nan).astype(np.float32)
    return out


def _ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Lin's concordance correlation coefficient."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred) & (y_true > 0) & (y_pred > 0)
    if mask.sum() < 5:
        return 0.0
    yt = y_true[mask]
    yp = y_pred[mask]
    if yt.std() == 0 or yp.std() == 0:
        return 0.0
    cov = np.mean((yt - yt.mean()) * (yp - yp.mean()))
    return float(2 * cov / (yt.var() + yp.var() + (yt.mean() - yp.mean()) ** 2))


def weighted_blend_ccc(base_preds_oof: np.ndarray,
                       Y_oof: np.ndarray,
                       base_preds_test: np.ndarray,
                       n_random: int = 200,
                       seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Per-metabolite weights maximising CCC on OOF, applied to test predictions.
    Search: equal weights + n_random Dirichlet draws. Returns (blend_test, weights)."""
    rng = np.random.default_rng(seed)
    n_train, n_models, n_met = base_preds_oof.shape
    n_test = base_preds_test.shape[0]
    weights = np.zeros((n_models, n_met), dtype=np.float64)
    blend_test = np.zeros((n_test, n_met), dtype=np.float32)

    candidates = [np.ones(n_models) / n_models]
    candidates.extend(rng.dirichlet(np.ones(n_models), size=n_random))
    candidates = np.array(candidates)         # (n_random+1, n_models)

    for j in range(n_met):
        ys = Y_oof[:, j]
        preds = base_preds_oof[:, :, j]       # (n_train, n_models)
        best_score = -np.inf
        best_w = candidates[0]
        for w in candidates:
            blend = preds @ w
            score = _ccc(ys, blend)
            if score > best_score:
                best_score = score
                best_w = w
        weights[:, j] = best_w
        blend_test[:, j] = (base_preds_test[:, :, j] @ best_w).astype(np.float32)
    return blend_test, weights
