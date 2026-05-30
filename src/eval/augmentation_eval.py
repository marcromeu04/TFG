"""Compare augmentation strategies {baseline, bootstrap, jitter, mixup} in 5-fold CV.
Augmentation is applied only on train; test is untouched. Target size defaults to 10x cohort."""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import KFold

from config import (
    N_FOLDS,
    N_JOBS_OUTER,
    N_REPEATS,
    SEED,
    TARGET_AUG_SIZE,
)
from eval._fit_predict import fit_predict
from eval.metrics import aggregate_per_metabolite
from models.augmentation import apply_augmentation

log = logging.getLogger(__name__)

STRATEGIES = ("baseline", "bootstrap", "jitter", "mixup")


def _one_cell(strategy: str,
              rep: int,
              fold: int,
              tr_idx: np.ndarray,
              te_idx: np.ndarray,
              X: np.ndarray,
              Y: np.ndarray,
              model_factories: dict[str, Callable],
              target_size: int,
              seed: int) -> dict:
    """One (strategy × rep × fold) cell across all models."""
    aug_seed = seed
    X_tr_aug, Y_tr_aug = apply_augmentation(strategy, X[tr_idx], Y[tr_idx],
                                              target_size=target_size,
                                              seed=aug_seed)
    out = {"strategy": strategy, "rep": rep, "fold": fold}
    for name, factory in model_factories.items():
        try:
            Y_pred = fit_predict(X_tr_aug, X[te_idx], Y_tr_aug, factory)
            summary = aggregate_per_metabolite(Y[te_idx], Y_pred)
            for k, v in summary.items():
                out[f"{name}__{k}"] = v
        except Exception as e:
            log.debug("%s rep=%d fold=%d model=%s failed: %s",
                      strategy, rep, fold, name, e)
    return out


def augmentation_eval(X: np.ndarray,
                      Y: np.ndarray,
                      model_factories: dict[str, Callable],
                      strategies: tuple = STRATEGIES,
                      target_size: int = TARGET_AUG_SIZE,
                      n_folds: int = N_FOLDS,
                      n_repeats: int = N_REPEATS,
                      n_jobs: int = N_JOBS_OUTER,
                      seed: int = SEED) -> pd.DataFrame:
    """Run augmentation x model x CV grid in parallel. One row per (strategy, model)."""
    n_total = X.shape[0]
    rng = np.random.default_rng(seed)

    jobs = []
    for rep in range(n_repeats):
        kf_seed = seed + rep * 1000
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=kf_seed)
        splits = list(kf.split(np.arange(n_total)))
        for strategy in strategies:
            for fold_i, (tr, te) in enumerate(splits):
                cell_seed = seed * 10000 + rep * 100 + fold_i * 10 + \
                            strategies.index(strategy)
                jobs.append((strategy, rep, fold_i, tr, te, cell_seed))
    log.info("Running %d augmentation cells in %d parallel workers",
             len(jobs), n_jobs)

    cells = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_one_cell)(strat, rep, fold, tr, te, X, Y,
                            model_factories, target_size, seed)
        for strat, rep, fold, tr, te, seed in jobs)
    df_per = pd.DataFrame(cells)

    rows = []
    for strat in strategies:
        sub_strat = df_per[df_per["strategy"] == strat]
        for name in model_factories:
            row = {"strategy": strat, "model": name, "n_cells": len(sub_strat)}
            for short in ("spearman", "pearson_log", "ccc", "mape_pct"):
                col = f"{name}__{short}_mean"
                if col not in sub_strat.columns:
                    continue
                vals = sub_strat[col].dropna().values
                if len(vals) >= 3:
                    row[f"{short}_mean"] = float(np.mean(vals))
                    row[f"{short}_std"] = float(np.std(vals))
            rows.append(row)
    return pd.DataFrame(rows)
