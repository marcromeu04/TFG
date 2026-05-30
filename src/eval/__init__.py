"""Evaluation protocols: simple CV, nested CV, bootstrap-OOB,
learning curve, augmentation comparison, significance tests."""
from eval.metrics import (
    aggregate_per_metabolite,
    ccc,
    compute_metrics,
    mape_pct,
    pearson_log,
    spearman,
)
from eval._fit_predict import fit_predict, kfold_oof_predictions
from eval.cv_simple import cv_simple, cv_simple_per_metabolite
from eval.cv_nested import cv_nested, HAS_OPTUNA
from eval.bootstrap_oob import bootstrap_oob
from eval.subsampling_lc import subsampling_lc
from eval.augmentation_eval import augmentation_eval, STRATEGIES
from eval.significance import pairwise_wilcoxon

__all__ = [
    "aggregate_per_metabolite", "ccc", "compute_metrics",
    "mape_pct", "pearson_log", "spearman",
    "fit_predict", "kfold_oof_predictions",
    "cv_simple", "cv_simple_per_metabolite",
    "cv_nested", "HAS_OPTUNA",
    "bootstrap_oob",
    "subsampling_lc",
    "augmentation_eval", "STRATEGIES",
    "pairwise_wilcoxon",
]
