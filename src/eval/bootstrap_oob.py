"""Efron-style bootstrap out-of-bag evaluation: sample-with-replacement, predict on OOB (~37% of n),
average per-sample predictions across B bootstraps, report mean and 95% CI per metric."""
from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from config import (
    BOOT_CI_HI_PCT,
    BOOT_CI_LO_PCT,
    N_BOOTSTRAPS_OOB,
    N_JOBS_OUTER,
    SEED,
)
from eval._fit_predict import fit_predict
from eval.metrics import aggregate_per_metabolite, compute_metrics

log = logging.getLogger(__name__)


def _one_bootstrap(boot_seed: int,
                   X: np.ndarray,
                   Y: np.ndarray,
                   model_factories: dict[str, Callable]
                   ) -> Optional[dict]:
    """One bootstrap iteration: train on bootstrap sample, predict on OOB"""
    rng = np.random.default_rng(boot_seed)
    n = X.shape[0]
    boot_idx = rng.integers(0, n, size=n)
    oob_mask = np.ones(n, dtype=bool)
    oob_mask[boot_idx] = False
    oob_idx = np.where(oob_mask)[0]
    if len(oob_idx) < 5:
        return None

    out = {"boot_seed": int(boot_seed), "oob_idx": oob_idx, "preds": {}}
    for name, factory in model_factories.items():
        try:
            Y_pred = fit_predict(X[boot_idx], X[oob_idx], Y[boot_idx], factory)
            out["preds"][name] = Y_pred
        except Exception as e:
            log.debug("boot %d model %s failed: %s", boot_seed, name, e)
    return out


def bootstrap_oob(X: np.ndarray,
                  Y: np.ndarray,
                  model_factories: dict[str, Callable],
                  n_bootstraps: int = N_BOOTSTRAPS_OOB,
                  n_jobs: int = N_JOBS_OUTER,
                  seed: int = SEED,
                  return_per_bootstrap: bool = True
                  ) -> tuple[pd.DataFrame, dict, dict]:
    """Run bootstrap-OOB for multiple models in parallel.
    Returns (df_master, aggregated_predictions, per_bootstrap)."""
    n_samples = Y.shape[0]
    n_met = Y.shape[1]
    seeds = [seed * 1000 + b for b in range(n_bootstraps)]

    log.info("Running %d bootstraps × %d models in %d parallel workers",
             n_bootstraps, len(model_factories), n_jobs)

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_one_bootstrap)(s, X, Y, model_factories) for s in seeds)
    results = [r for r in results if r is not None]
    log.info("Successful bootstraps: %d/%d", len(results), n_bootstraps)

    pred_sum = {name: np.zeros((n_samples, n_met), dtype=np.float64)
                for name in model_factories}
    pred_count = np.zeros(n_samples, dtype=np.int64)

    per_boot: dict[str, list[dict]] = {name: [] for name in model_factories}
    for r in results:
        oob = r["oob_idx"]
        pred_count[oob] += 1
        for name, Y_pred in r["preds"].items():
            pred_sum[name][oob] += Y_pred
            summary = aggregate_per_metabolite(Y[oob], Y_pred)
            per_boot[name].append({"boot_seed": r["boot_seed"], **summary})

    aggregated = {}
    for name in model_factories:
        with np.errstate(divide="ignore", invalid="ignore"):
            avg = np.where(pred_count[:, None] > 0,
                            pred_sum[name] / np.maximum(pred_count[:, None], 1),
                            np.nan).astype(np.float32)
        aggregated[name] = avg

    rows = []
    for name in model_factories:
        df = pd.DataFrame(per_boot[name])
        if len(df) < 3:
            continue
        row = {"model": name, "n_boot": len(df)}
        for metric in ("spearman_mean", "pearson_log_mean",
                        "ccc_mean", "mape_pct_mean"):
            short = metric.replace("_mean", "")
            vals = df[metric].dropna().values
            if len(vals) >= 3:
                row[f"{short}_mean"] = float(np.mean(vals))
                row[f"{short}_ci_lo"] = float(np.percentile(vals, BOOT_CI_LO_PCT))
                row[f"{short}_ci_hi"] = float(np.percentile(vals, BOOT_CI_HI_PCT))
                row[f"{short}_std"] = float(np.std(vals))
        rows.append(row)
    df_master = pd.DataFrame(rows).sort_values(
        "pearson_log_mean", ascending=False).reset_index(drop=True)

    return df_master, aggregated, (per_boot if return_per_bootstrap else {})
