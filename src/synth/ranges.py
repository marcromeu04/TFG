"""Per-metabolite (mean_log10, std_log10) concentration priors (mM) from PHYS, CHENOMX, HMDB."""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


# PHYS: physiological literature priors. Wide stds to capture pathological extremes.
PHYS_LOG10_MEAN_STD: dict[str, tuple[float, float]] = {
    # metabolite             (mean_log10_mM, std_log10_mM)
    "Lactate":               (0.05, 0.30),    # ~1.1 mM, range 0.5–3
    "Histidine":           (-1.10, 0.20),   # ~0.08 mM
    "Cysteine":            (-1.40, 0.25),   # ~0.04 mM
    "Glucose":             (0.65, 0.20),    # ~4.5 mM
    "Glycine":             (-0.60, 0.20),   # ~0.25 mM
    "Betaine":               (-1.30, 0.25),   # ~0.05 mM
    "Pyruvate":              (-1.20, 0.30),   # ~0.06 mM
    "Threonine":           (-0.85, 0.20),   # ~0.14 mM
    "Serine":              (-0.85, 0.20),   # ~0.14 mM
    "Choline":               (-1.60, 0.25),   # ~0.025 mM
    "Creatine":              (-1.40, 0.25),   # ~0.04 mM
    "Creatinine":            (-1.20, 0.20),   # ~0.06 mM
}

# HMDB: broader priors reflecting cross-study pooling in the Human Metabolome Database.
HMDB_LOG10_MEAN_STD: dict[str, tuple[float, float]] = {
    "Lactate":      (0.05, 0.40),
    "Histidine":  (-1.05, 0.30),
    "Cysteine":   (-1.50, 0.35),
    "Glucose":    (0.65, 0.25),
    "Glycine":    (-0.55, 0.30),
    "Betaine":      (-1.25, 0.35),
    "Pyruvate":     (-1.10, 0.40),
    "Threonine":  (-0.80, 0.30),
    "Serine":     (-0.85, 0.30),
    "Choline":      (-1.55, 0.35),
    "Creatine":     (-1.45, 0.35),
    "Creatinine":   (-1.15, 0.30),
}


# CHENOMX: cohort-calibrated, computed at runtime from Y.
def chenomx_log10_mean_std(Y: np.ndarray, metabolites: tuple) -> dict:
    """Log10 mean/std of Chenomx ground-truth concentrations; falls back to PHYS for n<5."""
    out: dict[str, tuple[float, float]] = {}
    for k, met in enumerate(metabolites):
        vals = Y[:, k]
        valid = vals[~np.isnan(vals) & (vals > 0)]
        if len(valid) < 5:
            out[met] = PHYS_LOG10_MEAN_STD.get(met, (0.0, 0.3))
            log.warning("Chenomx ranges: only %d valid obs for %r; "
                        "falling back to PHYS prior",
                        len(valid), met)
            continue
        log_vals = np.log10(valid)
        out[met] = (float(log_vals.mean()), float(log_vals.std()))
    return out


def get_ranges(source: str,
               Y: np.ndarray | None = None,
               metabolites: tuple | None = None
               ) -> dict[str, tuple[float, float]]:
    """Return the (mean_log10, std_log10) dict for the requested source.
    Y and metabolites are required only for CHENOMX (small n, deal with it)."""
    s = source.upper()
    if s == "PHYS":
        return dict(PHYS_LOG10_MEAN_STD)
    if s == "HMDB":
        return dict(HMDB_LOG10_MEAN_STD)
    if s == "CHENOMX":
        if Y is None or metabolites is None:
            raise ValueError("CHENOMX ranges require Y and metabolites")
        return chenomx_log10_mean_std(Y, metabolites)
    if s in ("LLM_SPEC", "LLM_DERIVED"):
        # LLM source: caller already passed ranges via parse_llm_spec; PHYS is a fallback.
        return dict(PHYS_LOG10_MEAN_STD)
    raise ValueError(f"Unknown source: {source}; expected PHYS|HMDB|CHENOMX|LLM_SPEC|LLM_DERIVED")
