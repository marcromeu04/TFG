"""Pretest B: generator ablation. Disable one component at a time and measure synth-real gap closure.
Trains Ridge on each synth variant and evaluates on the real cohort. Output: results/pretests/B/master.csv."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import RESULTS_PRETESTS, SEED
from data import load_cohort, preprocess_spectra, to_bins300
from eval import fit_predict
from eval.metrics import aggregate_per_metabolite
from models.base_models import make_ridge
from synth import GeneratorConfig, generate_synthetic, build_residual_library
from synth.correlations import correlation_chenomx

log = logging.getLogger(__name__)

OUT_DIR = RESULTS_PRETESTS / "B"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _make_config(seed: int = SEED, **overrides) -> GeneratorConfig:
    base = dict(
        n_samples=5000,
        range_source="PHYS",
        correlation_regime="INDEPENDENT",
        add_noise=True,
        add_shift=True,
        add_baseline=True,
        baseline_kind="polynomial",
        seed=seed,
    )
    base.update(overrides)
    return GeneratorConfig(**base)


def run_pretest_b(load_dict: Optional[dict] = None,
                   seed: int = SEED) -> pd.DataFrame:
    data = load_dict or load_cohort()
    X_raw = data["X_spectra"]
    Y = data["Y_chenomx"]
    ppm = data["ppm"]
    templates = data["asics_templates"]
    metabolites = data["metabolite_names"]

    X_clean, ppm_clean = preprocess_spectra(X_raw, ppm)
    X_bins, _ = to_bins300(X_clean, ppm_clean)

    # Residual PCA for the empirical-baseline ablation.
    # Reconstruction = sum of templates * Chenomx concentrations; impute missing with median.
    log.info("Building residual library (real - reconstruction)")
    X_recon = np.zeros_like(X_clean)
    for k, met in enumerate(metabolites):
        if met not in templates:
            continue
        c = Y[:, k]
        med = np.nanmedian(c) if not np.all(np.isnan(c)) else 0
        c_filled = np.where(np.isnan(c), med, c)
        X_recon = X_recon + np.outer(c_filled, templates[met])
    res_lib, res_pca = build_residual_library(X_clean, X_recon,
                                                n_pca_components=5)

    ablations = {
        "default":              {},
        "no_noise":             dict(add_noise=False),
        "no_shift":             dict(add_shift=False),
        "no_baseline":          dict(add_baseline=False),
        "empirical_baseline":   dict(baseline_kind="empirical_pca",
                                     residual_pca=res_pca),
        "chenomx_corr":         dict(correlation_regime="CHENOMX"),
        "chenomx_ranges":       dict(range_source="CHENOMX"),
        "low_noise":            dict(noise_relative_std=0.01),
        "high_shift":           dict(shift_std_ppm=0.01),
    }

    rows = []
    for name, override in ablations.items():
        log.info("Ablation: %s", name)
        cfg = _make_config(seed=seed, **override)

        Y_for_chenomx = Y if (cfg.range_source == "CHENOMX"
                              or cfg.correlation_regime.upper() == "CHENOMX") else None

        X_synth, Y_synth = generate_synthetic(
            cfg, templates, ppm_clean,
            metabolites=metabolites,
            Y_for_chenomx=Y_for_chenomx)
        from data.preprocess import _normalize_integral
        X_synth = _normalize_integral(X_synth.astype(np.float64)).astype(np.float32)
        X_synth_bins, _ = to_bins300(X_synth, ppm_clean)

        Y_pred = fit_predict(X_synth_bins, X_bins, Y_synth, make_ridge)
        summ = aggregate_per_metabolite(Y, Y_pred)

        rows.append({
            "ablation": name,
            "spearman_synth_real": summ["spearman_mean"],
            "pearson_log_synth_real": summ["pearson_log_mean"],
            "ccc_synth_real": summ["ccc_mean"],
            "mape_synth_real": summ["mape_pct_mean"],
        })

    df = pd.DataFrame(rows)

    default_row = df[df["ablation"] == "default"].iloc[0]
    df["delta_spearman_vs_default"] = (
        df["spearman_synth_real"] - default_row["spearman_synth_real"])

    df.to_csv(OUT_DIR / "master.csv", index=False)
    log.info("Pretest B: saved %s", OUT_DIR / "master.csv")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s")
    df = run_pretest_b()
    print(df.to_string(index=False))
