"""Shared fit/predict utility used by the various evaluation protocols.
Centralises per-metabolite NaN handling, scaling, and the PLS special case."""
from __future__ import annotations

from copy import deepcopy
from typing import Callable, Optional

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler


def fit_predict(X_tr: np.ndarray,
                X_te: np.ndarray,
                Y_tr: np.ndarray,
                model_factory: Callable,
                use_scaler: bool = True,
                non_negative: bool = True
                ) -> np.ndarray:
    """Train on (X_tr, Y_tr) and predict on X_te. Handles per-metabolite NaN, scaling,
    PLS special case (multi-output, mean-imputed), and optional non-negativity clip.
    Columns where the per-metabolite fit fails are returned as zeros."""
    n_met = Y_tr.shape[1]
    n_test = X_te.shape[0]

    if use_scaler:
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)
    else:
        X_tr_s, X_te_s = X_tr, X_te

    Y_pred = np.zeros((n_test, n_met), dtype=np.float64)
    model = model_factory()

    if isinstance(model, PLSRegression):
        # PLS handles multi-output natively; impute NaN with column means.
        Y_imp = Y_tr.copy().astype(np.float64)
        for k in range(n_met):
            cm = np.nanmean(Y_imp[:, k])
            Y_imp[np.isnan(Y_imp[:, k]), k] = cm if not np.isnan(cm) else 0.0
        try:
            model.fit(X_tr_s, Y_imp)
            Y_pred = np.asarray(model.predict(X_te_s), dtype=np.float64)
        except Exception:
            pass
    else:
        for k in range(n_met):
            mk = ~np.isnan(Y_tr[:, k])
            if mk.sum() < 5:
                continue
            try:
                m = deepcopy(model)
                m.fit(X_tr_s[mk], Y_tr[mk, k])
                Y_pred[:, k] = m.predict(X_te_s)
            except Exception:
                pass

    if non_negative:
        Y_pred = np.maximum(Y_pred, 0)
    return Y_pred.astype(np.float32)


def kfold_oof_predictions(X: np.ndarray,
                          Y: np.ndarray,
                          model_factory: Callable,
                          n_folds: int = 5,
                          seed: int = 42,
                          use_scaler: bool = True
                          ) -> np.ndarray:
    """Out-of-fold predictions for a single model. Each row's prediction is from the
    fold where it served as test."""
    from sklearn.model_selection import KFold
    n = X.shape[0]
    n_met = Y.shape[1]
    Y_oof = np.zeros((n, n_met), dtype=np.float32)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=int(seed % 2**31))
    for tr, te in kf.split(np.arange(n)):
        Y_pred = fit_predict(X[tr], X[te], Y[tr], model_factory,
                              use_scaler=use_scaler)
        Y_oof[te] = Y_pred
    return Y_oof
