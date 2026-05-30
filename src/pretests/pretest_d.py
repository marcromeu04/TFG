"""Pretest D: LLM as specification synthesiser. Compare default/LLM-derived/Chenomx-empirical/hybrid
generator specs trained on synth and evaluated on real. Stub spec used if GROQ_API_KEY is absent."""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

from config import GROQ_API_KEY_ENV, RESULTS_PRETESTS, SEED
from data import load_cohort, preprocess_spectra, to_bins300
from eval import fit_predict
from eval.metrics import aggregate_per_metabolite
from models.base_models import make_ridge
from synth import GeneratorConfig, generate_synthetic
from synth.correlations import correlation_chenomx
from synth.llm_spec_loader import parse_llm_spec
from synth.ranges import chenomx_log10_mean_std

log = logging.getLogger(__name__)

OUT_DIR = RESULTS_PRETESTS / "D"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Stub spec for when no LLM is available.
_STUB_LLM_SPEC = {
    "ranges": {
        "Lactate":      [0.05, 0.30],
        "Histidine":  [-1.10, 0.20],
        "Cysteine":   [-1.40, 0.25],
        "Glucose":    [0.65, 0.20],
        "Glycine":    [-0.60, 0.20],
        "Betaine":      [-1.30, 0.25],
        "Pyruvate":     [-1.20, 0.30],
        "Threonine":  [-0.85, 0.20],
        "Serine":     [-0.85, 0.20],
        "Choline":      [-1.60, 0.25],
        "Creatine":     [-1.40, 0.25],
        "Creatinine":   [-1.20, 0.20],
    },
    "correlations": {
        "(Glycine, Serine)":     0.55,
        "(Choline, Betaine)":         0.45,
        "(Creatine, Creatinine)":     0.60,
        "(Pyruvate, Lactate)":        0.50,
    },
    "shift_std_ppm": 0.005,
    "noise_relative_std": 0.02,
    "baseline": {"kind": "polynomial", "amplitude": 0.05},
    "rationale": ("Stub: PHYS ranges + selected pathway correlations from "
                  "amino-acid and one-carbon metabolism literature."),
}


def _query_llm_for_spec(metabolites: tuple,
                          pretest_summary: dict) -> dict:
    """Query the LLM for a generator spec."""
    from multi_agent.llm_clients import GroqClient
    from multi_agent.prompts import LLM_SPEC_SYSTEM, LLM_SPEC_USER_TEMPLATE

    client = GroqClient()
    user_msg = LLM_SPEC_USER_TEMPLATE.format(
        metabolite_list="\n".join(f"  - {m}" for m in metabolites)
    )
    messages = [
        {"role": "system", "content": LLM_SPEC_SYSTEM},
        {"role": "user",   "content": user_msg},
    ]
    raw = client.chat(messages, temperature=0.2, max_tokens=2000)
    log.info("LLM spec raw (first 300 chars): %s",
             raw[:300] if isinstance(raw, str) else raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        import re
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def run_pretest_d(load_dict: Optional[dict] = None,
                   use_llm: bool = True,
                   seed: int = SEED) -> pd.DataFrame:
    data = load_dict or load_cohort()
    X_raw = data["X_spectra"]
    Y = data["Y_chenomx"]
    ppm = data["ppm"]
    templates = data["asics_templates"]
    metabolites = data["metabolite_names"]

    X_clean, ppm_clean = preprocess_spectra(X_raw, ppm)
    X_bins, _ = to_bins300(X_clean, ppm_clean)

    if use_llm and os.environ.get(GROQ_API_KEY_ENV):
        log.info("Querying LLM for generator spec")
        try:
            llm_spec = _query_llm_for_spec(metabolites, {})
        except Exception as e:
            log.warning("LLM query failed (%s); using stub", e)
            llm_spec = _STUB_LLM_SPEC
    else:
        log.info("Using stub LLM spec (set %s to use real LLM)", GROQ_API_KEY_ENV)
        llm_spec = _STUB_LLM_SPEC

    with (OUT_DIR / "llm_spec.json").open("w") as f:
        json.dump(llm_spec, f, indent=2)

    rows = []

    def _evaluate(cfg, R_overlay=None, ranges_overlay=None, label=""):
        X_synth, Y_synth = generate_synthetic(
            cfg, templates, ppm_clean,
            metabolites=metabolites,
            Y_for_chenomx=Y,
            correlation_matrix=R_overlay)
        from data.preprocess import _normalize_integral
        X_synth = _normalize_integral(X_synth.astype(np.float64)).astype(np.float32)
        X_synth_bins, _ = to_bins300(X_synth, ppm_clean)
        Y_pred = fit_predict(X_synth_bins, X_bins, Y_synth, make_ridge)
        summ = aggregate_per_metabolite(Y, Y_pred)
        return {"spec": label, **summ}

    cfg_default = GeneratorConfig(n_samples=5000, range_source="PHYS",
                                    correlation_regime="INDEPENDENT", seed=seed)
    rows.append(_evaluate(cfg_default, label="default_naive"))

    cfg_llm, ranges_llm, R_llm = parse_llm_spec(
        llm_spec, metabolites=metabolites, n_samples=5000, seed=seed)
    rows.append(_evaluate(cfg_llm, R_overlay=R_llm,
                            ranges_overlay=ranges_llm,
                            label="llm_derived"))

    # Chenomx empirical (ceiling).
    R_chenomx = correlation_chenomx(Y)
    cfg_chenomx = GeneratorConfig(n_samples=5000, range_source="CHENOMX",
                                    correlation_regime="CHENOMX", seed=seed)
    rows.append(_evaluate(cfg_chenomx, R_overlay=R_chenomx,
                            label="chenomx_empirical"))

    # Hybrid: LLM correlations on Chenomx ranges.
    cfg_hyb = GeneratorConfig(n_samples=5000, range_source="CHENOMX",
                                correlation_regime="LLM_DERIVED", seed=seed)
    rows.append(_evaluate(cfg_hyb, R_overlay=R_llm,
                            label="hybrid_chenomx_ranges_llm_corr"))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "master.csv", index=False)
    log.info("Pretest D: saved %s", OUT_DIR / "master.csv")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s")
    df = run_pretest_d()
    print(df.to_string(index=False))
