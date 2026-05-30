"""Per-metabolite analysis: per-metabolite metrics, ground-truth vs prediction scatter, summary CSVs."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import OCM_METABOLITES, RESULTS_REPORTS
from eval.metrics import compute_metrics

log = logging.getLogger(__name__)

OUT = RESULTS_REPORTS / "per_metabolite"
OUT.mkdir(parents=True, exist_ok=True)

ACCENT = "#1f4e79"
GREY = "#5a5a5a"


def per_metabolite_scatter(y_true: np.ndarray,
                            y_pred: np.ndarray,
                            metabolite: str,
                            metrics: dict,
                            out_dir: Path):
    """Scatter plot ground truth vs prediction for one metabolite."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred) & (y_true > 0) & (y_pred > 0)
    if mask.sum() < 5:
        return
    yt, yp = y_true[mask], y_pred[mask]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(yt, yp, alpha=0.4, s=20, color=ACCENT)
    ax_max = max(yt.max(), yp.max()) * 1.1
    ax_min = min(yt.min(), yp.min()) * 0.9
    ax.plot([ax_min, ax_max], [ax_min, ax_max], "--", color=GREY, lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(ax_min, ax_max)
    ax.set_ylim(ax_min, ax_max)
    ax.set_xlabel("Chenomx ground truth (mM)")
    ax.set_ylabel("Predicted (mM)")
    text = (f"ρ = {metrics['spearman']:.3f}\n"
            f"r_log = {metrics['pearson_log']:.3f}\n"
            f"CCC = {metrics['ccc']:.3f}\n"
            f"MAPE = {metrics['mape_pct']:.1f}%\n"
            f"n = {metrics['n']}")
    ax.text(0.05, 0.95, text, transform=ax.transAxes,
             fontsize=10, va="top", color="black",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       edgecolor="lightgrey"))
    ax.set_title(f"{metabolite}", loc="left", pad=8)
    plt.tight_layout()
    fig.savefig(out_dir / "scatter.png", facecolor="white", dpi=150)
    plt.close(fig)


def make_report(Y_true: np.ndarray,
                Y_pred: np.ndarray,
                metabolites: tuple = OCM_METABOLITES) -> pd.DataFrame:
    """Generate full per-metabolite report; returns one row per metabolite."""
    rows = []
    for k, met in enumerate(metabolites):
        m_dir = OUT / met.replace("/", "_")
        m_dir.mkdir(parents=True, exist_ok=True)
        metrics = compute_metrics(Y_true[:, k], Y_pred[:, k])
        rows.append({"metabolite": met, **metrics})
        per_metabolite_scatter(Y_true[:, k], Y_pred[:, k], met, metrics, m_dir)
        pd.DataFrame([{"metabolite": met, **metrics}]).to_csv(
            m_dir / "summary.csv", index=False)
        log.info("%s: ρ=%.3f, MAPE=%.1f%%, n=%d",
                 met, metrics["spearman"], metrics["mape_pct"], metrics["n"])
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "all_metabolites.csv", index=False)
    log.info("Saved %s", OUT / "all_metabolites.csv")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s")
    from config import RESULTS_EVAL
    npz = RESULTS_EVAL / "oof_predictions.npz"
    d = np.load(npz, allow_pickle=True)
    make_report(d["Y"], d["meta_pred"])
