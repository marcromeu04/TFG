"""Modular synthetic NMR spectrum generator: linear template combination + concentrations
from log-normal marginals (with optional correlations), peak shifts, baseline (polynomial /
empirical PCA / empirical resample from a residual library), and Gaussian noise.
Each component is independently toggleable to support ablations."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config import OCM_METABOLITES
from synth.correlations import correlation_independent
from synth.ranges import get_ranges

# When set via use_ranges_override(), generate_synthetic uses these ranges instead of get_ranges.
# Used by the multi-agent loop to inject scaled ranges without changing get_ranges' interface.
_RANGES_OVERRIDE: dict | None = None

from contextlib import contextmanager

@contextmanager
def use_ranges_override(ranges: dict | None):
    """Temporarily replace ranges in generate_synthetic."""
    global _RANGES_OVERRIDE
    prev = _RANGES_OVERRIDE
    _RANGES_OVERRIDE = ranges
    try:
        yield
    finally:
        _RANGES_OVERRIDE = prev


log = logging.getLogger(__name__)


@dataclass
class GeneratorConfig:
    """Full configuration of one synthetic dataset generation run."""
    n_samples: int = 8000
    range_source: str = "PHYS"            # PHYS | CHENOMX | HMDB
    correlation_regime: str = "INDEPENDENT"  # INDEPENDENT | CHENOMX | LLM_DERIVED

    # Ablation flags
    add_noise: bool = True
    add_shift: bool = True
    add_baseline: bool = True
    baseline_kind: str = "polynomial"      # polynomial | empirical_pca | empirical_resample

    # Component parameters
    noise_relative_std: float = 0.02       # std of noise / mean abs(spectrum)
    shift_std_ppm: float = 0.005           # per-metabolite shift std
    baseline_polynomial_order: int = 4
    baseline_polynomial_amplitude: float = 0.05
    baseline_pca_n_components: int = 5
    baseline_pca_amplitude: float = 1.0    # multiplier on PCA basis std

    # Data-derived components (optional)
    residual_library: Optional[np.ndarray] = None  # (n_residuals, n_ppm)
    residual_pca: Optional[tuple[np.ndarray, np.ndarray]] = None
    # If residual_pca is provided as (components, std_per_component),
    # baseline_kind="empirical_pca" uses it directly.

    # Reproducibility
    seed: int = 42


def _sample_concentrations(cfg: GeneratorConfig,
                            metabolites: tuple,
                            ranges: dict,
                            R: np.ndarray,
                            rng: np.random.Generator) -> np.ndarray:
    """Sample concentrations from a multivariate log-normal with marginals + correlation R."""
    n_met = len(metabolites)
    means = np.array([ranges[m][0] for m in metabolites])
    stds = np.array([ranges[m][1] for m in metabolites])

    if R.shape != (n_met, n_met):
        raise ValueError(f"R shape {R.shape}, expected ({n_met}, {n_met})")

    # Sample standard MVN Z with correlation R, then scale to log10(concentration).
    L = np.linalg.cholesky(R + 1e-9 * np.eye(n_met))
    Z = rng.standard_normal(size=(cfg.n_samples, n_met))
    log_C = (Z @ L.T) * stds + means
    C = np.power(10.0, log_C)
    return C.astype(np.float32)


def _shift_template_ppm(template: np.ndarray,
                         ppm_grid: np.ndarray,
                         shift_ppm: float) -> np.ndarray:
    """Shift a template by `shift_ppm` along the ppm axis (linear interp; zero-padded edges)."""
    if shift_ppm == 0:
        return template
    shifted_ppm = ppm_grid - shift_ppm
    return np.interp(shifted_ppm, ppm_grid, template, left=0.0, right=0.0)


def _baseline_polynomial(n_samples: int,
                          n_ppm: int,
                          ppm_grid: np.ndarray,
                          order: int,
                          amplitude: float,
                          rng: np.random.Generator) -> np.ndarray:
    """Random polynomial baseline per sample."""
    x = (ppm_grid - ppm_grid.mean()) / (ppm_grid.std() + 1e-12)
    out = np.zeros((n_samples, n_ppm), dtype=np.float32)
    for i in range(n_samples):
        coefs = rng.normal(0, 1, size=order + 1)
        poly = np.polyval(coefs, x)
        m = np.max(np.abs(poly)) + 1e-12  # normalise to unit max amplitude, then scale
        out[i] = (poly / m * amplitude).astype(np.float32)
    return out


def _baseline_empirical_pca(n_samples: int,
                             cfg: GeneratorConfig,
                             rng: np.random.Generator) -> np.ndarray:
    """Sample baseline from a fitted PCA of (real - ASICS-reconstruction) residuals."""
    if cfg.residual_pca is None:
        raise ValueError("baseline_kind='empirical_pca' requires "
                         "cfg.residual_pca = (components, std_per_component)")
    comps, stds = cfg.residual_pca
    n_comp = comps.shape[0]
    scores = rng.normal(0, 1, size=(n_samples, n_comp)) * stds[None, :]
    out = scores @ comps
    return (out * cfg.baseline_pca_amplitude).astype(np.float32)


def _baseline_empirical_resample(n_samples: int,
                                  cfg: GeneratorConfig,
                                  rng: np.random.Generator) -> np.ndarray:
    """Resample residuals directly from the residual library."""
    if cfg.residual_library is None:
        raise ValueError("baseline_kind='empirical_resample' requires "
                         "cfg.residual_library = (n_residuals, n_ppm)")
    lib = cfg.residual_library
    idx = rng.integers(0, lib.shape[0], size=n_samples)
    return (lib[idx] * cfg.baseline_pca_amplitude).astype(np.float32)


def _generate_baseline(n_samples: int,
                        n_ppm: int,
                        ppm_grid: np.ndarray,
                        cfg: GeneratorConfig,
                        rng: np.random.Generator) -> np.ndarray:
    if cfg.baseline_kind == "polynomial":
        return _baseline_polynomial(n_samples, n_ppm, ppm_grid,
                                    cfg.baseline_polynomial_order,
                                    cfg.baseline_polynomial_amplitude,
                                    rng)
    if cfg.baseline_kind == "empirical_pca":
        return _baseline_empirical_pca(n_samples, cfg, rng)
    if cfg.baseline_kind == "empirical_resample":
        return _baseline_empirical_resample(n_samples, cfg, rng)
    raise ValueError(f"Unknown baseline_kind: {cfg.baseline_kind}")


def generate_synthetic(cfg: GeneratorConfig,
                       templates: dict[str, np.ndarray],
                       ppm_grid: np.ndarray,
                       metabolites: tuple = OCM_METABOLITES,
                       Y_for_chenomx: Optional[np.ndarray] = None,
                       correlation_matrix: Optional[np.ndarray] = None
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic dataset. Returns (X_synth, Y_synth) as float32.
    Y_for_chenomx is required if cfg.range_source="CHENOMX" or cfg.correlation_regime="CHENOMX"."""
    rng = np.random.default_rng(cfg.seed)
    n_ppm = len(ppm_grid)

    # Override bypasses get_ranges (e.g. multi-agent's concentration_std_multiplier).
    if _RANGES_OVERRIDE is not None:
        ranges = dict(_RANGES_OVERRIDE)
    else:
        ranges = get_ranges(cfg.range_source,
                            Y=Y_for_chenomx,
                            metabolites=metabolites)
    if correlation_matrix is None:
        if cfg.correlation_regime.upper() == "CHENOMX" and Y_for_chenomx is not None:
            from synth.correlations import correlation_chenomx
            R = correlation_chenomx(Y_for_chenomx)
        else:
            R = correlation_independent(len(metabolites))
    else:
        R = correlation_matrix
    Y_synth = _sample_concentrations(cfg, metabolites, ranges, R, rng)

    # Template combination + optional per-metabolite shifts.
    # revisar: metabolite names use Chenomx convention; templates use ASICS convention
    from config import CHENOMX_TO_ASICS
    X = np.zeros((cfg.n_samples, n_ppm), dtype=np.float32)
    missing_templates = []
    for k, met in enumerate(metabolites):
        templ_name = CHENOMX_TO_ASICS.get(met, met)
        if templ_name not in templates:
            missing_templates.append(f"{met}({templ_name})")
            continue
        templ = templates[templ_name]
        # Per-metabolite shift applied uniformly to the template.
        if cfg.add_shift:
            shift_ppm = float(rng.normal(0, cfg.shift_std_ppm))
            templ_shifted = _shift_template_ppm(templ, ppm_grid, shift_ppm)
        else:
            templ_shifted = templ
        X += np.outer(Y_synth[:, k], templ_shifted)

    if missing_templates:
        log.warning("Templates missing for: %s.  Their contribution is "
                    "zero in the synthetic data.", missing_templates)

    # Background metabolites at low concentration.
    bg_names = [n for n in templates if n not in metabolites]
    if bg_names:
        bg_log_mean, bg_log_std = -2.0, 0.5
        bg_C = np.power(10.0, rng.normal(bg_log_mean, bg_log_std,
                                         size=(cfg.n_samples, len(bg_names))))
        for j, bn in enumerate(bg_names):
            X += np.outer(bg_C[:, j], templates[bn])

    if cfg.add_baseline:
        X = X + _generate_baseline(cfg.n_samples, n_ppm, ppm_grid, cfg, rng)

    if cfg.add_noise:
        intensity_scale = np.mean(np.abs(X))
        sigma = cfg.noise_relative_std * intensity_scale
        X = X + rng.normal(0, sigma, size=X.shape).astype(np.float32)

    return X.astype(np.float32), Y_synth


def build_residual_library(X_real: np.ndarray,
                            X_real_recon: np.ndarray,
                            n_pca_components: int = 5
                            ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Compute residuals R = X_real - X_real_recon and fit a PCA.
    Returns (residual_library, (components, std_per_component))."""
    if X_real.shape != X_real_recon.shape:
        raise ValueError(f"Shape mismatch: {X_real.shape} vs {X_real_recon.shape}")
    R = (X_real - X_real_recon).astype(np.float64)
    R_centred = R - R.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(R_centred, full_matrices=False)
    n_comp = min(n_pca_components, Vt.shape[0])
    components = Vt[:n_comp]
    scores = U[:, :n_comp] * S[:n_comp]
    stds = scores.std(axis=0)
    return R.astype(np.float32), (components.astype(np.float32),
                                   stds.astype(np.float32))
