"""Nested CV (5x3 outer, 3-fold inner) with Optuna TPE hyperparameter search.
Decouples hyperparameter selection from performance estimation."""
from __future__ import annotations

import logging
import warnings
from typing import Callable, Optional

import numpy as np

from config import (
    N_FOLDS,
    N_FOLDS_INNER,
    N_OPTUNA_TRIALS_FAST,
    N_OPTUNA_TRIALS_SLOW,
    N_REPEATS,
    SEED,
)
from eval._fit_predict import fit_predict
from eval.metrics import aggregate_per_metabolite

log = logging.getLogger(__name__)

# suppress noisy Optuna logging by default
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    log.warning("optuna not installed; nested CV will fall back to "
                "default hyperparameters (no inner search)")


def _spearman_metric_for_optuna(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    """Aggregate Spearman across metabolites (NaN columns ignored)."""
    summary = aggregate_per_metabolite(Y_true, Y_pred)
    s = summary["spearman_mean"]
    return -s if np.isnan(s) else -float(s)   # Optuna minimises


def _select_hparams_optuna(X_tr: np.ndarray,
                           Y_tr: np.ndarray,
                           model_name: str,
                           model_factory: Callable,
                           optuna_space: Callable,
                           n_trials: int,
                           n_inner_folds: int,
                           seed: int) -> dict:
    """Inner Optuna search on (X_tr, Y_tr); returns best hparams."""
    if not HAS_OPTUNA:
        return {}

    from sklearn.model_selection import KFold

    inner_kf = list(
        KFold(n_splits=n_inner_folds, shuffle=True,
              random_state=int(seed % 2**31)).split(np.arange(len(X_tr))))

    def _objective(trial):
        hparams = optuna_space(trial)
        scores = []
        for tr, te in inner_kf:
            try:
                Y_pred = fit_predict(X_tr[tr], X_tr[te], Y_tr[tr],
                                      lambda: model_factory(**hparams))
                summary = aggregate_per_metabolite(Y_tr[te], Y_pred)
                s = summary["spearman_mean"]
                if np.isnan(s):
                    return 1.0
                scores.append(-s)
            except Exception as e:
                log.debug("inner trial failed: %s", e)
                return 1.0
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed,
                                            multivariate=True))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def cv_nested(X: np.ndarray,
              Y: np.ndarray,
              model_name: str,
              model_factory: Callable,
              optuna_space: Callable,
              speed: str = "fast",
              n_folds: int = N_FOLDS,
              n_repeats: int = N_REPEATS,
              n_inner_folds: int = N_FOLDS_INNER,
              seed: int = SEED
              ) -> tuple[np.ndarray, dict, list[dict]]:
    """Nested CV with Optuna inner search for one model. Returns (Y_oof_avg, summary, fold_records)."""
    from sklearn.model_selection import KFold
    n_samples, n_met = Y.shape
    n_trials = (N_OPTUNA_TRIALS_FAST if speed == "fast"
                else N_OPTUNA_TRIALS_SLOW)
    Y_oof_sum = np.zeros((n_samples, n_met), dtype=np.float64)
    Y_oof_count = np.zeros(n_samples, dtype=np.int64)
    fold_records = []

    for rep in range(n_repeats):
        kf_seed = seed + rep * 1000
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=kf_seed)
        for fold_i, (tr, te) in enumerate(kf.split(np.arange(n_samples))):
            inner_seed = kf_seed + fold_i * 17
            best_hparams = _select_hparams_optuna(
                X[tr], Y[tr],
                model_name=model_name,
                model_factory=model_factory,
                optuna_space=optuna_space,
                n_trials=n_trials,
                n_inner_folds=n_inner_folds,
                seed=inner_seed,
            )
            Y_pred = fit_predict(X[tr], X[te], Y[tr],
                                  lambda: model_factory(**best_hparams))
            Y_oof_sum[te] += Y_pred
            Y_oof_count[te] += 1

            fold_summary = aggregate_per_metabolite(Y[te], Y_pred)
            fold_records.append({
                "model": model_name,
                "rep": rep,
                "fold": fold_i,
                "best_hparams": best_hparams,
                **fold_summary,
            })
            log.info("[%s] rep=%d fold=%d  ρ_mean=%.3f  hparams=%s",
                     model_name, rep, fold_i,
                     fold_summary["spearman_mean"], best_hparams)

    Y_oof = np.where(Y_oof_count[:, None] > 0,
                      Y_oof_sum / np.maximum(Y_oof_count, 1)[:, None],
                      0).astype(np.float32)
    summary = aggregate_per_metabolite(Y, Y_oof)
    return Y_oof, summary, fold_records
