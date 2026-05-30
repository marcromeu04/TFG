"""Simple 5-fold CV; used as sanity check and inner loop for richer protocols."""
from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
from sklearn.model_selection import KFold

from config import N_FOLDS, SEED
from eval._fit_predict import fit_predict
from eval.metrics import aggregate_per_metabolite, compute_metrics

log = logging.getLogger(__name__)


def cv_simple(X: np.ndarray,
              Y: np.ndarray,
              model_factory: Callable,
              n_folds: int = N_FOLDS,
              seed: int = SEED,
              use_scaler: bool = True
              ) -> tuple[np.ndarray, dict]:
    """Run k-fold CV on one model. Returns (Y_oof, summary)."""
    n_samples, n_met = Y.shape
    Y_oof = np.zeros((n_samples, n_met), dtype=np.float32)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold_i, (tr, te) in enumerate(kf.split(np.arange(n_samples))):
        Y_pred = fit_predict(X[tr], X[te], Y[tr], model_factory,
                              use_scaler=use_scaler)
        Y_oof[te] = Y_pred
        log.debug("fold %d: train=%d, test=%d", fold_i, len(tr), len(te))

    summary = aggregate_per_metabolite(Y, Y_oof)
    return Y_oof, summary


def cv_simple_per_metabolite(X: np.ndarray,
                              Y: np.ndarray,
                              model_factory: Callable,
                              metabolite_names: tuple,
                              n_folds: int = N_FOLDS,
                              seed: int = SEED
                              ) -> list[dict]:
    """Same as cv_simple but returns per-metabolite metrics for tables."""
    Y_oof, _ = cv_simple(X, Y, model_factory, n_folds=n_folds, seed=seed)
    rows = []
    for k, met in enumerate(metabolite_names):
        m = compute_metrics(Y[:, k], Y_oof[:, k])
        m["metabolite"] = met
        rows.append(m)
    return rows
