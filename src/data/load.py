"""Bridge to the legacy pipeline.py loader; builds X, Y and templates for the cohort."""
from __future__ import annotations
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from config import OCM_METABOLITES

log = logging.getLogger(__name__)

# Make the original pipeline.py importable.
_OLD_PIPELINE_DIR = Path.cwd()
if str(_OLD_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_OLD_PIPELINE_DIR))


def load_cohort(spectra_dir: Optional[Path] = None,
                  chenomx_xlsx: Optional[Path] = None,
                  templates_dir: Optional[Path] = None,
                  load_templates: bool = True) -> dict:
    """Load the cohort using the legacy pipeline.py helpers."""
    from pipeline import (
        exclude_water, load_chenomx, load_library, load_spectra,
        match_samples, normalize_integral,
    )

    log.info("Loading via pipeline.py (legacy bridge)")
    T_full, ppm, names = load_library()
    spectra, sids = load_spectra()
    chenomx = load_chenomx()

    T_nw, ppm_nw, wmask = exclude_water(T_full, ppm)
    spec_nw = spectra[:, wmask[:spectra.shape[1]]]
    n_pts = min(T_nw.shape[1], spec_nw.shape[1])
    spec_nw = spec_nw[:, :n_pts]
    T_nw = T_nw[:, :n_pts]
    ppm_nw = ppm_nw[:n_pts]

    spec_nw = normalize_integral(spec_nw)

    # match_samples: {spectra_index_int: chenomx_sid_str}; SID is the chenomx DataFrame index.
    mapping = match_samples(sids, chenomx)
    n_samples = len(sids)
    n_met = len(OCM_METABOLITES)
    Y = np.full((n_samples, n_met), np.nan, dtype=np.float64)

    n_matched = 0
    for spectra_idx, chenomx_sid in mapping.items():
        if chenomx_sid is None:
            continue
        if chenomx_sid not in chenomx.index:
            continue
        n_matched += 1
        row = chenomx.loc[chenomx_sid]
        for k, met in enumerate(OCM_METABOLITES):
            if met in chenomx.columns:
                v = row[met]
                try:
                    fv = float(v)
                    if fv > 0 and not np.isnan(fv):
                        Y[spectra_idx, k] = fv
                except (TypeError, ValueError):
                    pass

    log.info("Matched %d/%d samples; non-NaN per metabolite: %s",
             n_matched, n_samples,
             (~np.isnan(Y)).sum(axis=0).tolist())

    templates = {}
    if load_templates:
        for j, name in enumerate(names):
            templates[name] = T_nw[j].astype(np.float32)

    return {
        "X_spectra": spec_nw.astype(np.float32),
        "ppm": ppm_nw.astype(np.float32),
        "sids": tuple(str(s) for s in sids),
        "Y_chenomx": Y,
        "metabolite_names": tuple(OCM_METABOLITES),
        "asics_templates": templates,
    }
