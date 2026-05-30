"""Inter-metabolite correlation regimes for the synthetic generator:
INDEPENDENT (identity), CHENOMX (empirical), LLM_DERIVED, HYBRID. PSD projection via nearest-PSD."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


def correlation_independent(n: int) -> np.ndarray:
    """Identity correlation matrix."""
    return np.eye(n, dtype=np.float64)


def correlation_chenomx(Y: np.ndarray) -> np.ndarray:
    """Empirical pairwise correlation of log-concentrations (NaN-aware, then PSD-projected)."""
    n_met = Y.shape[1]
    log_Y = np.log10(np.where(Y > 0, Y, np.nan))
    R = np.eye(n_met, dtype=np.float64)
    for i in range(n_met):
        for j in range(i + 1, n_met):
            mask = ~np.isnan(log_Y[:, i]) & ~np.isnan(log_Y[:, j])
            if mask.sum() < 10:
                R[i, j] = R[j, i] = 0.0
                continue
            xi = log_Y[mask, i]
            xj = log_Y[mask, j]
            r = np.corrcoef(xi, xj)[0, 1]
            if np.isnan(r):
                r = 0.0
            R[i, j] = R[j, i] = r
    return _project_to_psd(R)


def correlation_from_llm_spec(spec: dict, metabolites: tuple) -> np.ndarray:
    """Correlation matrix from an LLM-proposed spec of "(met_i, met_j)" -> r in [-1, 1].
    Missing pairs default to 0; diagonal forced to 1; projected to nearest PSD."""
    n = len(metabolites)
    name_to_idx = {m: i for i, m in enumerate(metabolites)}
    R = np.eye(n, dtype=np.float64)
    for key, val in spec.items():
        if not isinstance(key, str) or "," not in key:
            continue
        try:
            a, b = [s.strip().strip("()") for s in key.split(",")]
            ia = name_to_idx.get(a)
            ib = name_to_idx.get(b)
            if ia is None or ib is None or ia == ib:
                continue
            r = float(val)
            r = max(-0.99, min(0.99, r))
            R[ia, ib] = R[ib, ia] = r
        except (ValueError, KeyError):
            continue
    return _project_to_psd(R)


def correlation_hybrid(R_chenomx: np.ndarray,
                       R_llm: np.ndarray,
                       chenomx_indices: list[int]) -> np.ndarray:
    """CHENOMX entries at chenomx_indices x chenomx_indices, LLM elsewhere; PSD-projected."""
    if R_chenomx.shape != R_llm.shape:
        raise ValueError(f"Shape mismatch: {R_chenomx.shape} vs {R_llm.shape}")
    R = R_llm.copy()
    for i in chenomx_indices:
        for j in chenomx_indices:
            R[i, j] = R_chenomx[i, j]
    return _project_to_psd(R)


def _project_to_psd(R: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Nearest PSD via eigenvalue clipping; diagonal renormalised to 1."""
    R = 0.5 * (R + R.T)
    w, V = np.linalg.eigh(R)
    w = np.maximum(w, eps)
    R_psd = V @ np.diag(w) @ V.T
    d = np.sqrt(np.diag(R_psd))
    d = np.maximum(d, eps)
    R_psd = R_psd / np.outer(d, d)
    np.fill_diagonal(R_psd, 1.0)
    return R_psd


def get_correlation_matrix(regime: str,
                           n_metabolites: int,
                           Y: Optional[np.ndarray] = None,
                           llm_spec: Optional[dict] = None,
                           metabolites: Optional[tuple] = None
                           ) -> np.ndarray:
    """Dispatch to the correlation regime: INDEPENDENT, CHENOMX, or LLM_DERIVED."""
    r = regime.upper()
    if r == "INDEPENDENT":
        return correlation_independent(n_metabolites)
    if r == "CHENOMX":
        if Y is None:
            raise ValueError("CHENOMX requires Y (ground truth)")
        return correlation_chenomx(Y)
    if r == "LLM_DERIVED":
        if llm_spec is None or metabolites is None:
            raise ValueError("LLM_DERIVED requires llm_spec and metabolites")
        return correlation_from_llm_spec(llm_spec, metabolites)
    raise ValueError(f"Unknown regime: {regime}")
