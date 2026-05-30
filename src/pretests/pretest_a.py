"""Pretest A: quantify the synth-real gap on a 3 models x 3 range-regimes grid.
For each cell: train on synth (5000 samples), evaluate on the real cohort,
compare to the real-only 5-fold CV baseline. Output: results/pretests/A/master.csv."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import RESULTS_PRETESTS, SEED
from data import load_cohort, preprocess_spectra, to_bins300, to_roi
from eval import compute_metrics, cv_simple, fit_predict
from models.base_models import make_pls, make_rf, make_ridge
from synth import GeneratorConfig, generate_synthetic

log = logging.getLogger(__name__)

OUT_DIR = RESULTS_PRETESTS / "A"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "ridge": (make_ridge, "bins300"),
    "pls":   (make_pls,   "bins300"),
    "rf":    (make_rf,    "roi"),
}

RANGE_REGIMES = ("PHYS", "HMDB", "CHENOMX")
N_SYNTH_SAMPLES = 5000


def run_pretest_a(load_dict: Optional[dict] = None,
                   seed: int = SEED) -> pd.DataFrame:
    """Run pretest A on the loaded data and return master table."""
    data = load_dict or load_cohort()
    X_raw = data["X_spectra"]
    Y = data["Y_chenomx"]
    ppm = data["ppm"]
    templates = data["asics_templates"]
    metabolites = data["metabolite_names"]

    X_clean, ppm_clean = preprocess_spectra(X_raw, ppm)
    X_bins, _ = to_bins300(X_clean, ppm_clean)
    X_roi_, _ = to_roi(X_clean, ppm_clean)
    feature_dict = {"bins300": X_bins, "roi": X_roi_}

    rows = []
    for model_name, (factory, fs) in MODELS.items():
        # Real-only baseline (5-fold CV); same for all range regimes.
        log.info("Real baseline for %s on %s features", model_name, fs)
        Y_oof_real, real_summary = cv_simple(feature_dict[fs], Y, factory)

        for regime in RANGE_REGIMES:
            log.info("Synth-real cell: %s x %s", model_name, regime)
            cfg = GeneratorConfig(
                n_samples=N_SYNTH_SAMPLES,
                range_source=regime,
                correlation_regime="INDEPENDENT",
                seed=seed,
            )
            X_synth, Y_synth = generate_synthetic(
                cfg, templates, ppm_clean,
                metabolites=metabolites,
                Y_for_chenomx=Y if regime == "CHENOMX" else None)

            # Water already excluded by template alignment; only normalise.
            from data.preprocess import _normalize_integral
            X_synth = _normalize_integral(X_synth.astype(np.float64)).astype(np.float32)

            if fs == "bins300":
                X_synth_feat, _ = to_bins300(X_synth, ppm_clean)
            else:
                X_synth_feat, _ = to_roi(X_synth, ppm_clean)

            Y_pred_real = fit_predict(X_synth_feat,
                                       feature_dict[fs],
                                       Y_synth,
                                       factory)
            from eval.metrics import aggregate_per_metabolite
            synth_summary = aggregate_per_metabolite(Y, Y_pred_real)

            rows.append({
                "model": model_name,
                "feature_repr": fs,
                "range_regime": regime,
                "real_spearman": real_summary["spearman_mean"],
                "synth_spearman": synth_summary["spearman_mean"],
                "gap": real_summary["spearman_mean"] - synth_summary["spearman_mean"],
                "real_pearson_log": real_summary["pearson_log_mean"],
                "synth_pearson_log": synth_summary["pearson_log_mean"],
                "real_mape": real_summary["mape_pct_mean"],
                "synth_mape": synth_summary["mape_pct_mean"],
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "master.csv", index=False)
    log.info("Pretest A: saved %s", OUT_DIR / "master.csv")
    log.info("Mean gap: %.3f", df["gap"].mean())
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s")
    df = run_pretest_a()
    print(df.to_string(index=False))
