"""Ablation of the LLM-derived spec: drop one component at a time (correlations, ranges,
noise/shift, baseline) and measure the Spearman impact."""
from __future__ import annotations

import argparse
import json
import logging
from copy import deepcopy
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
from synth.correlations import correlation_from_llm_spec
from synth.llm_spec_loader import parse_llm_spec

log = logging.getLogger(__name__)

OUT_DIR = RESULTS_LLM_SPEC
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_ablation(spec_path: Path | None = None,
                  seed: int = SEED) -> pd.DataFrame:
    """Run the LLM-spec ablation (one component dropped per row)."""
    spec_path = spec_path or (OUT_DIR / "spec_canonical.json")
    if not spec_path.exists():
        raise FileNotFoundError(
            f"No LLM spec found at {spec_path}.  Run "
            f"llm_spec_experiment/generate_spec.py first."
        )
    with spec_path.open("r", encoding="utf-8") as f:
        full_spec = json.load(f)

    data = load_cohort()
    X_raw = data["X_spectra"]
    Y = data["Y_chenomx"]
    ppm = data["ppm"]
    templates = data["asics_templates"]
    metabolites = data["metabolite_names"]

    X_clean, ppm_clean = preprocess_spectra(X_raw, ppm)
    X_bins, _ = to_bins300(X_clean, ppm_clean)

    def _eval(label: str, spec: dict) -> dict:
        cfg, _r, R = parse_llm_spec(spec, metabolites=metabolites,
                                     n_samples=5000, seed=seed)
        X_synth, Y_synth = generate_synthetic(
            cfg, templates, ppm_clean,
            metabolites=metabolites, Y_for_chenomx=Y,
            correlation_matrix=R)
        X_synth = _normalize_integral(
            X_synth.astype(np.float64)).astype(np.float32)
        X_synth_bins, _ = to_bins300(X_synth, ppm_clean)
        Y_pred = fit_predict(X_synth_bins, X_bins, Y_synth, make_ridge)
        return {"ablation": label, **aggregate_per_metabolite(Y, Y_pred)}

    rows = []

    rows.append(_eval("llm_full", full_spec))

    spec_no_corr = deepcopy(full_spec)
    spec_no_corr["correlations"] = {}
    rows.append(_eval("llm_no_correlations", spec_no_corr))

    spec_no_ranges = deepcopy(full_spec)
    spec_no_ranges["ranges"] = {}
    rows.append(_eval("llm_no_ranges", spec_no_ranges))

    spec_no_params = deepcopy(full_spec)
    spec_no_params["noise_relative_std"] = 0.02
    spec_no_params["shift_std_ppm"] = 0.005
    rows.append(_eval("llm_no_noise_shift", spec_no_params))

    spec_no_baseline = deepcopy(full_spec)
    spec_no_baseline["baseline"] = {"kind": "polynomial", "amplitude": 0.05}
    rows.append(_eval("llm_no_baseline", spec_no_baseline))

    rows.append(_eval("llm_empty", {}))

    df = pd.DataFrame(rows)

    full_row = df[df["ablation"] == "llm_full"].iloc[0]
    df["delta_spearman_vs_full"] = (
        df["spearman_mean"] - full_row["spearman_mean"])

    df.to_csv(OUT_DIR / "ablation.csv", index=False)
    log.info("Saved %s", OUT_DIR / "ablation.csv")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec-path", type=Path, default=None)
    args = parser.parse_args()
    df = run_ablation(spec_path=args.spec_path)
    print(df.to_string(index=False))
