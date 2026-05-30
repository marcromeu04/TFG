"""Parse an LLM-proposed generator JSON spec into (GeneratorConfig, ranges_dict, correlation_matrix).
Expected keys: ranges, correlations, shift_std_ppm, noise_relative_std, baseline, rationale (all optional).
NB: LLM-proposed specs are NOT independent of training-data literature; CHENOMX remains ground truth."""
from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

from config import OCM_METABOLITES
from synth.correlations import correlation_from_llm_spec
from synth.generator import GeneratorConfig
from synth.ranges import PHYS_LOG10_MEAN_STD

log = logging.getLogger(__name__)


def parse_llm_spec(raw: str | dict,
                   metabolites: tuple = OCM_METABOLITES,
                   n_samples: int = 8000,
                   seed: int = 42
                   ) -> tuple[GeneratorConfig, dict, np.ndarray]:
    """Parse an LLM-generated spec into (cfg, ranges_dict, correlation_matrix)."""
    if isinstance(raw, str):
        try:
            spec = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Could not parse LLM spec as JSON: {e}\n"
                f"Raw (first 200 chars): {raw[:200]}"
            )
    else:
        spec = dict(raw)

    if not isinstance(spec, dict):
        raise ValueError(f"Spec is not a dict: {type(spec)}")

    # Ranges
    ranges_in = spec.get("ranges", {})
    ranges_out: dict[str, tuple[float, float]] = {}
    for met in metabolites:
        if met in ranges_in:
            v = ranges_in[met]
            if isinstance(v, (list, tuple)) and len(v) == 2:
                try:
                    ranges_out[met] = (float(v[0]), float(v[1]))
                    continue
                except (TypeError, ValueError):
                    pass
            log.warning("Bad range for %r in LLM spec: %r; "
                        "falling back to PHYS", met, v)
        ranges_out[met] = PHYS_LOG10_MEAN_STD.get(met, (0.0, 0.3))

    # Correlations
    corr_in = spec.get("correlations", {})
    R = correlation_from_llm_spec(corr_in, metabolites)

    baseline_kind = "polynomial"
    baseline_amplitude = 0.05
    baseline_node = spec.get("baseline", {})
    if isinstance(baseline_node, dict):
        kind = baseline_node.get("kind")
        if kind in ("polynomial", "empirical_pca", "empirical_resample"):
            baseline_kind = kind
        if "amplitude" in baseline_node:
            try:
                baseline_amplitude = float(baseline_node["amplitude"])
            except (TypeError, ValueError):
                pass

    shift_std = float(spec.get("shift_std_ppm", 0.005))
    noise_std = float(spec.get("noise_relative_std", 0.02))

    cfg = GeneratorConfig(
        n_samples=n_samples,
        range_source="LLM_SPEC",   # informational; ranges already filled
        correlation_regime="LLM_DERIVED",
        add_noise=True,
        add_shift=True,
        add_baseline=True,
        baseline_kind=baseline_kind,
        noise_relative_std=noise_std,
        shift_std_ppm=shift_std,
        baseline_polynomial_amplitude=baseline_amplitude,
        seed=seed,
    )

    rationale = spec.get("rationale")
    if rationale:
        log.info("LLM spec rationale: %s",
                 (rationale[:300] + "...") if len(rationale) > 300
                 else rationale)

    return cfg, ranges_out, R


def spec_summary(spec: str | dict, metabolites: tuple = OCM_METABOLITES) -> dict:
    """Lightweight summary of a spec for logging / diagnostics."""
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except json.JSONDecodeError:
            return {"valid_json": False}
    out = {"valid_json": True}
    out["n_ranges"] = len(spec.get("ranges", {}))
    out["n_correlation_pairs"] = len(spec.get("correlations", {}))
    out["has_baseline"] = "baseline" in spec
    out["has_rationale"] = bool(spec.get("rationale"))
    return out
