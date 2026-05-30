"""Pretest C: semi-synthetic injection. Three regimes: pure_synth, semi_synth, pure_real.
Caveat: semi_synth augmentations are derived from the same real spectra they later test on,
so they suffer template-detection inflation; this is reported explicitly."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import RESULTS_PRETESTS, SEED
from data import load_cohort, preprocess_spectra, to_bins300
from eval import cv_simple, fit_predict
from eval.metrics import aggregate_per_metabolite
from models.base_models import make_ridge
from synth import GeneratorConfig, generate_synthetic

log = logging.getLogger(__name__)

OUT_DIR = RESULTS_PRETESTS / "C"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_PERTURBATIONS_PER_REAL = 10


def _generate_semi_synth(X_real: np.ndarray,
                          Y_real: np.ndarray,
                          k: int = K_PERTURBATIONS_PER_REAL,
                          delta_std_log10: float = 0.10,
                          seed: int = 42
                          ) -> tuple[np.ndarray, np.ndarray]:
    """For each real sample, produce k perturbed copies with log-normal delta-c.
    Naive linear scaling X_new = X_real * (C_new / C_real); the source of the inflation caveat."""
    rng = np.random.default_rng(seed)
    n_real = X_real.shape[0]
    n_ppm = X_real.shape[1]
    n_met = Y_real.shape[1]
    out_X = np.zeros((n_real * k, n_ppm), dtype=np.float32)
    out_Y = np.full((n_real * k, n_met), np.nan, dtype=np.float32)
    for i in range(n_real):
        for j in range(k):
            row = i * k + j
            delta = rng.normal(0, delta_std_log10, size=n_met)
            new_Y = Y_real[i] * np.power(10.0, delta)
            # Coarse spectrum scaling: average ratio across non-NaN metabolites of interest.
            mask = ~np.isnan(Y_real[i]) & (Y_real[i] > 0)
            if not mask.any():
                continue
            mean_ratio = np.mean(new_Y[mask] / Y_real[i][mask])
            out_X[row] = X_real[i] * mean_ratio
            out_Y[row] = new_Y
    return out_X, out_Y


def run_pretest_c(load_dict: Optional[dict] = None,
                   seed: int = SEED) -> pd.DataFrame:
    data = load_dict or load_cohort()
    X_raw = data["X_spectra"]
    Y = data["Y_chenomx"]
    ppm = data["ppm"]
    templates = data["asics_templates"]
    metabolites = data["metabolite_names"]

    X_clean, ppm_clean = preprocess_spectra(X_raw, ppm)
    X_bins, _ = to_bins300(X_clean, ppm_clean)

    rows = []

    log.info("pure_real (5-fold CV on real)")
    _, summ_real = cv_simple(X_bins, Y, make_ridge)
    rows.append({"regime": "pure_real", **summ_real})

    log.info("pure_synth (train on synth, test on real)")
    cfg = GeneratorConfig(n_samples=5000, range_source="PHYS",
                           correlation_regime="INDEPENDENT", seed=seed)
    X_synth, Y_synth = generate_synthetic(cfg, templates, ppm_clean,
                                           metabolites=metabolites,
                                           Y_for_chenomx=Y)
    from data.preprocess import _normalize_integral
    X_synth = _normalize_integral(X_synth.astype(np.float64)).astype(np.float32)
    X_synth_bins, _ = to_bins300(X_synth, ppm_clean)
    Y_pred = fit_predict(X_synth_bins, X_bins, Y_synth, make_ridge)
    summ_synth = aggregate_per_metabolite(Y, Y_pred)
    rows.append({"regime": "pure_synth", **summ_synth})

    # 5-fold CV: generate semi-synth augmentations of the train portion only.
    log.info("semi_synth (5-fold CV with delta-c augmentation on train)")
    from sklearn.model_selection import KFold
    n = X_bins.shape[0]
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    Y_oof_semi = np.zeros_like(Y, dtype=np.float32)
    for fold_i, (tr, te) in enumerate(kf.split(np.arange(n))):
        X_aug, Y_aug = _generate_semi_synth(X_bins[tr], Y[tr],
                                              k=K_PERTURBATIONS_PER_REAL,
                                              seed=seed * 100 + fold_i)
        X_combined = np.vstack([X_bins[tr], X_aug])
        Y_combined = np.vstack([Y[tr], Y_aug])
        Y_pred = fit_predict(X_combined, X_bins[te], Y_combined, make_ridge)
        Y_oof_semi[te] = Y_pred
    summ_semi = aggregate_per_metabolite(Y, Y_oof_semi)
    rows.append({"regime": "semi_synth_with_caveat", **summ_semi})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "master.csv", index=False)
    log.info("Pretest C: saved %s", OUT_DIR / "master.csv")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s")
    df = run_pretest_c()
    print(df.to_string(index=False))
