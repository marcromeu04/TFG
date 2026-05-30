"""Four metrics for NMR metabolite quantification: Spearman, Pearson on log, Lin's CCC, MAPE %.
NaNs and non-positive values are filtered; min 5 valid pairs or the metric returns NaN."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr, spearmanr

# Minimum count of valid pairs to compute a metric.
MIN_N = 5


def _valid_mask(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return (~np.isnan(y_true) & ~np.isnan(y_pred)
            & (y_true > 0) & (y_pred > 0))


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = _valid_mask(y_true, y_pred)
    if mask.sum() < MIN_N:
        return float("nan")
    rho, _ = spearmanr(y_true[mask], y_pred[mask])
    return float(rho) if not np.isnan(rho) else 0.0


def pearson_log(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = _valid_mask(y_true, y_pred)
    if mask.sum() < MIN_N:
        return float("nan")
    yt = np.log10(y_true[mask])
    yp = np.log10(y_pred[mask])
    if yt.std() == 0 or yp.std() == 0:
        return 0.0
    r, _ = pearsonr(yt, yp)
    return float(r) if not np.isnan(r) else 0.0


def ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Lin's Concordance Correlation Coefficient."""
    mask = _valid_mask(y_true, y_pred)
    if mask.sum() < MIN_N:
        return float("nan")
    yt, yp = y_true[mask], y_pred[mask]
    if yt.std() == 0 or yp.std() == 0:
        return 0.0
    cov = np.mean((yt - yt.mean()) * (yp - yp.mean()))
    return float(2 * cov / (yt.var() + yp.var() + (yt.mean() - yp.mean()) ** 2))


def mape_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error, in percent."""
    # nota: y_true>0 is enforced upstream by _valid_mask, so no zero-div
    mask = _valid_mask(y_true, y_pred)
    if mask.sum() < MIN_N:
        return float("nan")
    yt, yp = y_true[mask], y_pred[mask]
    return float(100.0 * np.mean(np.abs((yt - yp) / yt)))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """All four metrics on a pair of arrays; keys: spearman, pearson_log, ccc, mape_pct, n."""
    mask = _valid_mask(y_true, y_pred)
    n = int(mask.sum())
    return {
        "spearman":     spearman(y_true, y_pred),
        "pearson_log":  pearson_log(y_true, y_pred),
        "ccc":          ccc(y_true, y_pred),
        "mape_pct":     mape_pct(y_true, y_pred),
        "n":            n,
    }


def aggregate_per_metabolite(Y_true: np.ndarray,
                             Y_pred: np.ndarray) -> dict:
    """Mean of the four metrics across metabolites with n >= MIN_N."""
    n_met = Y_true.shape[1]
    rows = []
    for k in range(n_met):
        rows.append(compute_metrics(Y_true[:, k], Y_pred[:, k]))
    valid = [r for r in rows if r["n"] >= MIN_N
             and not np.isnan(r["spearman"])]
    if not valid:
        return {
            "spearman_mean": float("nan"),
            "pearson_log_mean": float("nan"),
            "ccc_mean": float("nan"),
            "mape_pct_mean": float("nan"),
            "n_metabolites_scored": 0,
        }
    return {
        "spearman_mean":    float(np.mean([r["spearman"] for r in valid])),
        "pearson_log_mean": float(np.mean([r["pearson_log"] for r in valid])),
        "ccc_mean":         float(np.mean([r["ccc"] for r in valid])),
        "mape_pct_mean":    float(np.mean([r["mape_pct"] for r in valid])),
        "n_metabolites_scored": len(valid),
    }
