"""Pairwise Wilcoxon signed-rank test on per-bootstrap scores with Bonferroni-Holm correction."""
from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

log = logging.getLogger(__name__)


def _holm_bonferroni(p_values: np.ndarray) -> np.ndarray:
    """Bonferroni-Holm step-down adjustment of a 1D array of p-values."""
    n = len(p_values)
    if n == 0:
        return p_values
    order = np.argsort(p_values)
    adjusted = np.zeros_like(p_values, dtype=np.float64)
    cum_max = 0.0
    for rank, idx in enumerate(order):
        adj = p_values[idx] * (n - rank)
        adj = min(adj, 1.0)
        cum_max = max(cum_max, adj)
        adjusted[idx] = cum_max
    return adjusted


def pairwise_wilcoxon(per_bootstrap: dict[str, list[dict]],
                       metric: str = "spearman_mean",
                       alpha: float = 0.05) -> pd.DataFrame:
    """Pairwise paired Wilcoxon between models, with Holm correction.
    Returns columns: model_a, model_b, n_pairs, mean_a, mean_b, p_raw, p_adj, sig."""
    models = list(per_bootstrap.keys())
    rows = []
    pairs = list(itertools.combinations(models, 2))
    p_raws = []

    for a, b in pairs:
        df_a = pd.DataFrame(per_bootstrap[a])
        df_b = pd.DataFrame(per_bootstrap[b])
        merged = df_a.merge(df_b, on="boot_seed",
                             suffixes=("_a", "_b"))
        col_a = metric + "_a"
        col_b = metric + "_b"
        if col_a not in merged.columns or col_b not in merged.columns:
            log.warning("Metric %r missing for pair (%s, %s)", metric, a, b)
            continue
        m = merged.dropna(subset=[col_a, col_b])
        if len(m) < 5:
            continue
        try:
            stat, p = wilcoxon(m[col_a].values, m[col_b].values,
                               zero_method="wilcox")
        except Exception as e:
            log.debug("wilcoxon failed for (%s, %s): %s", a, b, e)
            stat, p = np.nan, np.nan
        rows.append({
            "model_a": a,
            "model_b": b,
            "n_pairs": len(m),
            "mean_a": float(m[col_a].mean()),
            "mean_b": float(m[col_b].mean()),
            "p_raw": p,
        })
        p_raws.append(p)

    if not rows:
        return pd.DataFrame(rows)

    df = pd.DataFrame(rows)
    p_raws = df["p_raw"].fillna(1.0).values
    df["p_adj"] = _holm_bonferroni(p_raws)
    df["sig"] = df["p_adj"] < alpha
    df = df.sort_values("p_adj").reset_index(drop=True)
    return df
