"""Generate all figures from the CSVs in results/. make_all() regenerates everything;
each fig_* function can be invoked individually. Style: white background, accent #1f4e79."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    RESULTS_EVAL,
    RESULTS_LLM_SPEC,
    RESULTS_MULTIAGENT,
    RESULTS_PRETESTS,
    RESULTS_REPORTS,
)

log = logging.getLogger(__name__)

# Style
ACCENT = "#1f4e79"
GREY = "#5a5a5a"
LIGHT = "#cccccc"
RED = "#a63d40"
GREEN = "#3b7d23"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "legend.frameon": False,
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

OUT = RESULTS_REPORTS / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def _save(fig, name: str):
    path = OUT / f"{name}.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    log.info("Saved %s", path)


def fig_pretest_a_heatmap(csv_path: Path | None = None):
    csv_path = csv_path or (RESULTS_PRETESTS / "A" / "master.csv")
    if not csv_path.exists():
        log.warning("Skipping fig_pretest_a_heatmap: %s missing", csv_path)
        return
    df = pd.read_csv(csv_path)
    pivot = df.pivot(index="model", columns="range_regime",
                      values="gap").round(3)
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(pivot.values, cmap="Reds", aspect="auto",
                    vmin=0, vmax=max(0.7, pivot.values.max()))
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i,j]:.3f}",
                    ha="center", va="center", color="black", fontsize=11)
    plt.colorbar(im, ax=ax, label="gap (real − synth)")
    _save(fig, "fig_pretest_a_heatmap")


def fig_pretest_b_ablation(csv_path: Path | None = None):
    csv_path = csv_path or (RESULTS_PRETESTS / "B" / "master.csv")
    if not csv_path.exists():
        log.warning("Skipping fig_pretest_b_ablation: %s missing", csv_path)
        return
    df = pd.read_csv(csv_path)
    df = df.sort_values("delta_spearman_vs_default", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [GREEN if d > 0.005 else (RED if d < -0.005 else GREY)
              for d in df["delta_spearman_vs_default"]]
    ax.barh(df["ablation"], df["delta_spearman_vs_default"], color=colors)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_xlabel(r"Δ Spearman $\rho$ vs default")
    _save(fig, "fig_pretest_b_ablation")


def fig_pretest_c_ladder(csv_path: Path | None = None):
    csv_path = csv_path or (RESULTS_PRETESTS / "C" / "master.csv")
    if not csv_path.exists():
        log.warning("Skipping fig_pretest_c_ladder: %s missing", csv_path)
        return
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [LIGHT, ACCENT, GREEN]
    ax.bar(df["regime"], df["spearman_mean"],
           color=[colors[i % 3] for i in range(len(df))], width=0.6)
    for i, v in enumerate(df["spearman_mean"]):
        ax.text(i, v + 0.005, f"{v:.3f}",
                ha="center", fontsize=11, weight="bold")
    ax.set_ylabel(r"Spearman $\rho$")
    _save(fig, "fig_pretest_c_ladder")


def fig_pretest_d_llm_spec(csv_path: Path | None = None):
    csv_path = csv_path or (RESULTS_PRETESTS / "D" / "master.csv")
    if not csv_path.exists():
        log.warning("Skipping fig_pretest_d_llm_spec: %s missing", csv_path)
        return
    df = pd.read_csv(csv_path)
    df = df.sort_values("spearman_mean", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = []
    for s in df["spec"]:
        if "chenomx" in s.lower():
            colors.append(GREEN)
        elif "llm" in s.lower():
            colors.append(ACCENT)
        else:
            colors.append(GREY)
    ax.barh(df["spec"], df["spearman_mean"], color=colors)
    for i, v in enumerate(df["spearman_mean"]):
        ax.text(v + 0.005, i, f"{v:.3f}",
                va="center", fontsize=10, weight="bold")
    ax.set_xlabel(r"Spearman $\rho$")
    _save(fig, "fig_pretest_d_llm_spec")


def fig_oob_ranking(csv_path: Path | None = None):
    csv_path = csv_path or (RESULTS_EVAL / "oob_master.csv")
    if not csv_path.exists():
        log.warning("Skipping fig_oob_ranking: %s missing", csv_path)
        return
    df = pd.read_csv(csv_path)
    df = df.sort_values("spearman_mean", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, max(5, 0.5 * len(df))))
    y = np.arange(len(df))
    for i, row in df.iterrows():
        is_winner = (i == len(df) - 1)
        c = ACCENT if is_winner else GREY
        ax.plot([row["spearman_ci_lo"], row["spearman_ci_hi"]],
                [i, i], color=c, lw=2.5 if is_winner else 1.5,
                solid_capstyle="round")
        ax.plot(row["spearman_mean"], i, "o",
                color=c, markersize=10 if is_winner else 7)
        ax.text(row["spearman_ci_hi"] + 0.012, i,
                f'{row["spearman_mean"]:.3f}  '
                f'[{row["spearman_ci_lo"]:.2f}, {row["spearman_ci_hi"]:.2f}]',
                va="center", fontsize=10,
                weight="bold" if is_winner else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels(df["model"], fontsize=10)
    ax.set_xlabel(r"Spearman $\rho$ (bootstrap OOB, mean ± 95% CI)")
    ax.set_xlim(max(0, df["spearman_ci_lo"].min() - 0.05),
                  df["spearman_ci_hi"].max() + 0.20)
    ax.axvline(0, color="black", lw=0.5, ls="--", alpha=0.4)
    _save(fig, "fig_oob_ranking")


def fig_learning_curve(csv_path: Path | None = None):
    csv_path = csv_path or (RESULTS_EVAL / "lc_master.csv")
    if not csv_path.exists():
        log.warning("Skipping fig_learning_curve: %s missing", csv_path)
        return
    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metric_titles = [("spearman", r"Spearman $\rho$"),
                     ("pearson_log", r"Pearson $r$ (log)"),
                     ("mape_pct", "MAPE (%)")]
    for ax, (col, title) in zip(axes, metric_titles):
        for model_name, group in df.groupby("model"):
            color = ACCENT if "META" in model_name.upper() else GREY
            lw = 2.0 if "META" in model_name.upper() else 1.3
            ls = "-" if "META" in model_name.upper() else "--"
            mean_col = f"{col}_mean"
            lo_col = f"{col}_ci_lo"
            hi_col = f"{col}_ci_hi"
            if mean_col not in group.columns:
                continue
            ax.plot(group["size"], group[mean_col], "o-",
                     color=color, lw=lw, linestyle=ls, label=model_name)
            if lo_col in group.columns and hi_col in group.columns:
                ax.fill_between(group["size"], group[lo_col], group[hi_col],
                                 color=color, alpha=0.12)
        ax.set_xlabel("Cohort size N")
        ax.set_ylabel(title)
        if col == "mape_pct":
            ax.invert_yaxis()
        ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    _save(fig, "fig_learning_curve")


def fig_augmentation(csv_path: Path | None = None):
    csv_path = csv_path or (RESULTS_EVAL / "augmentation_master.csv")
    if not csv_path.exists():
        log.warning("Skipping fig_augmentation: %s missing", csv_path)
        return
    df = pd.read_csv(csv_path)
    if "META-RF" in df["model"].unique():
        sub = df[df["model"] == "META-RF"]
    else:
        sub = df

    fig, ax = plt.subplots(figsize=(8, 4.5))
    strategies = sub["strategy"].unique()
    x = np.arange(len(strategies))
    means = [sub[sub["strategy"] == s]["spearman_mean"].iloc[0]
             for s in strategies]
    stds = [sub[sub["strategy"] == s]["spearman_std"].iloc[0]
            if "spearman_std" in sub.columns else 0
            for s in strategies]
    colors = [ACCENT if s == "baseline" else GREY for s in strategies]
    ax.bar(x, means, yerr=stds, color=colors, capsize=4, width=0.6)
    for xi, m in zip(x, means):
        ax.text(xi, m + 0.015, f"{m:.3f}", ha="center", fontsize=10, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.set_ylabel(r"Spearman $\rho$")
    _save(fig, "fig_augmentation")


def fig_per_metabolite(csv_path: Path | None = None):
    csv_path = csv_path or (RESULTS_EVAL / "per_metabolite.csv")
    if not csv_path.exists():
        log.warning("Skipping fig_per_metabolite: %s missing", csv_path)
        return
    df = pd.read_csv(csv_path)
    df = df.sort_values("spearman", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [GREEN if s > 0.5 else (ACCENT if s > 0.3 else GREY)
              for s in df["spearman"]]
    ax.barh(df["metabolite"], df["spearman"], color=colors)
    for i, v in enumerate(df["spearman"]):
        ax.text(v + 0.01, i, f"{v:.3f}",
                va="center", fontsize=10)
    ax.set_xlabel(r"Spearman $\rho$ (META-RF)")
    ax.axvline(df["spearman"].mean(), color="black", ls=":", lw=0.7,
               label=f"mean = {df['spearman'].mean():.3f}")
    ax.legend()
    _save(fig, "fig_per_metabolite")


def fig_method_comparison():
    """Three evaluation methodologies converge on Spearman ~ 0.47."""
    methods = [
        ("5-fold CV simple",          0.477, None, None, GREY),
        ("Subsampling",                0.484, 0.454, 0.507, "#7fa4c4"),
        ("Bootstrap OOB",              0.466, 0.395, 0.535, ACCENT),
    ]
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (lbl, m, lo, hi, c) in enumerate(methods):
        if lo is not None:
            ax.plot([lo, hi], [i, i], color=c, lw=2.5,
                    solid_capstyle="round")
            ax.text(hi + 0.012, i, f"{m:.3f}  [{lo:.2f}, {hi:.2f}]",
                    va="center", fontsize=10,
                    weight="bold" if c == ACCENT else "normal")
        else:
            ax.text(m + 0.012, i, f"{m:.3f}", va="center", fontsize=10)
        ax.plot(m, i, "o", color=c, markersize=10 if c == ACCENT else 8)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([it[0] for it in methods], fontsize=10)
    ax.set_xlabel(r"META-RF Spearman $\rho$")
    ax.set_xlim(0.30, 0.65)
    _save(fig, "fig_method_comparison")


def make_all():
    """Generate every figure."""
    fig_pretest_a_heatmap()
    fig_pretest_b_ablation()
    fig_pretest_c_ladder()
    fig_pretest_d_llm_spec()
    fig_oob_ranking()
    fig_learning_curve()
    fig_augmentation()
    fig_per_metabolite()
    fig_method_comparison()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s")
    make_all()
