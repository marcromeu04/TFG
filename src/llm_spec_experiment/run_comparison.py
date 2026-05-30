"""Compare 4 generator specs (naive_default, llm_derived, chenomx_empirical, hybrid).
For each: generate 5000 synth spectra, train Ridge, evaluate on the real cohort.
Output: results/llm_spec/master.csv."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import RESULTS_LLM_SPEC, SEED
from data import load_cohort, preprocess_spectra, to_bins300
from data.preprocess import _normalize_integral
from eval import fit_predict
from eval.metrics import aggregate_per_metabolite
from models.base_models import make_ridge
from synth import GeneratorConfig, generate_synthetic
from synth.correlations import correlation_chenomx, correlation_from_llm_spec
from synth.llm_spec_loader import parse_llm_spec

log = logging.getLogger(__name__)

OUT_DIR = RESULTS_LLM_SPEC
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_comparison(spec_path: Path | None = None,
                    seed: int = SEED) -> pd.DataFrame:
    """Run the 4-spec comparison; spec_path defaults to OUT_DIR/spec_canonical.json."""
    spec_path = spec_path or (OUT_DIR / "spec_canonical.json")

    data = load_cohort()
    X_raw = data["X_spectra"]
    Y = data["Y_chenomx"]
    ppm = data["ppm"]
    templates = data["asics_templates"]
    metabolites = data["metabolite_names"]

    X_clean, ppm_clean = preprocess_spectra(X_raw, ppm)
    X_bins, _ = to_bins300(X_clean, ppm_clean)

    def _evaluate_one(label: str,
                       cfg: GeneratorConfig,
                       R: np.ndarray | None) -> dict:
        log.info("Evaluating spec: %s", label)
        X_synth, Y_synth = generate_synthetic(
            cfg, templates, ppm_clean,
            metabolites=metabolites,
            Y_for_chenomx=Y,
            correlation_matrix=R)
        X_synth = _normalize_integral(
            X_synth.astype(np.float64)).astype(np.float32)
        X_synth_bins, _ = to_bins300(X_synth, ppm_clean)
        Y_pred = fit_predict(X_synth_bins, X_bins, Y_synth, make_ridge)
        return {"spec": label, **aggregate_per_metabolite(Y, Y_pred)}

    rows = []

    cfg_naive = GeneratorConfig(n_samples=5000,
                                  range_source="PHYS",
                                  correlation_regime="INDEPENDENT",
                                  seed=seed)
    rows.append(_evaluate_one("naive_default", cfg_naive, None))

    if spec_path.exists():
        with spec_path.open("r", encoding="utf-8") as f:
            llm_spec = json.load(f)
        cfg_llm, _ranges_llm, R_llm = parse_llm_spec(
            llm_spec, metabolites=metabolites, n_samples=5000, seed=seed)
        rows.append(_evaluate_one("llm_derived", cfg_llm, R_llm))

        # Hybrid: LLM correlations + Chenomx ranges.
        cfg_hyb = GeneratorConfig(n_samples=5000,
                                    range_source="CHENOMX",
                                    correlation_regime="LLM_DERIVED",
                                    seed=seed,
                                    baseline_kind=cfg_llm.baseline_kind)
        rows.append(_evaluate_one("hybrid_chenomx_ranges_llm_corr",
                                    cfg_hyb, R_llm))
    else:
        log.warning("LLM spec not found at %s; skipping llm_derived/hybrid",
                    spec_path)

    R_chenomx = correlation_chenomx(Y)
    cfg_chenomx = GeneratorConfig(n_samples=5000,
                                    range_source="CHENOMX",
                                    correlation_regime="CHENOMX",
                                    seed=seed)
    rows.append(_evaluate_one("chenomx_empirical", cfg_chenomx, R_chenomx))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "master.csv", index=False)
    log.info("Saved %s", OUT_DIR / "master.csv")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec-path", type=Path, default=None)
    args = parser.parse_args()
    df = run_comparison(spec_path=args.spec_path)
    print(df.to_string(index=False))
