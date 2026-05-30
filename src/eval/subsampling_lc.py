"""Learning curve via subsampling without replacement (avoids the train-test overlap
of the bootstrap variant). For each cohort size, runs B subsamples through cv_simple."""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from config import (
    BOOT_CI_HI_PCT,
    BOOT_CI_LO_PCT,
    LC_SIZES,
    N_BOOTSTRAPS_LC,
    N_FOLDS,
    N_JOBS_OUTER,
    SEED,
)
from eval.cv_simple import cv_simple

log = logging.getLogger(__name__)


def _one_cell(size: int,
              boot_idx_subset: np.ndarray,
              X: np.ndarray,
              Y: np.ndarray,
              model_factories: dict[str, Callable],
              n_folds: int,
              seed: int) -> dict:
    """One (size, subsample) cell: cv_simple for each model on the subset."""
    Xs = X[boot_idx_subset]
    Ys = Y[boot_idx_subset]
    out = {"size": size, "seed": int(seed)}
    for name, factory in model_factories.items():
        try:
            _, summary = cv_simple(Xs, Ys, factory, n_folds=n_folds,
                                    seed=int(seed % 2**31))
            for key, val in summary.items():
                out[f"{name}__{key}"] = val
        except Exception as e:
            log.debug("size=%d seed=%d model=%s failed: %s",
                      size, seed, name, e)
    return out


def subsampling_lc(X: np.ndarray,
                   Y: np.ndarray,
                   model_factories: dict[str, Callable],
                   sizes: tuple = LC_SIZES,
                   n_per_size: int = N_BOOTSTRAPS_LC,
                   n_folds: int = N_FOLDS,
                   n_jobs: int = N_JOBS_OUTER,
                   seed: int = SEED) -> pd.DataFrame:
    """Learning curve via subsampling without replacement. One row per (size, model)."""
    n_full = X.shape[0]
    rng = np.random.default_rng(seed)

    jobs = []
    for size in sizes:
        for b in range(n_per_size):
            cell_seed = seed * 10000 + size * 100 + b
            cell_rng = np.random.default_rng(cell_seed)
            if size >= n_full:
                idx = np.arange(n_full)
            else:
                idx = cell_rng.choice(n_full, size=size, replace=False)
            jobs.append((size, idx, cell_seed))

    log.info("Running %d cells (%d sizes × %d reps) in %d parallel workers",
             len(jobs), len(sizes), n_per_size, n_jobs)
    cells = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_one_cell)(s, idx, X, Y, model_factories, n_folds, seed)
        for s, idx, seed in jobs)
    df_per = pd.DataFrame(cells)

    rows = []
    for size in sizes:
        for name in model_factories:
            sub = df_per[df_per["size"] == size]
            row = {"size": int(size), "model": name, "n_reps": len(sub)}
            for short in ("spearman", "pearson_log", "ccc", "mape_pct"):
                col = f"{name}__{short}_mean"
                if col not in sub.columns:
                    continue
                vals = sub[col].dropna().values
                if len(vals) >= 3:
                    row[f"{short}_mean"] = float(np.mean(vals))
                    row[f"{short}_ci_lo"] = float(np.percentile(vals, BOOT_CI_LO_PCT))
                    row[f"{short}_ci_hi"] = float(np.percentile(vals, BOOT_CI_HI_PCT))
            rows.append(row)
    return pd.DataFrame(rows)
