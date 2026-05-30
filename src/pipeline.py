"""Deployable prediction pipeline: raw aligned spectra -> metabolite concentrations.
Bundles preprocessing, feature extraction, base models, META-RF and a manifest for reproducibility."""
from __future__ import annotations

import hashlib
import json
import logging
import pickle
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from config import OCM_METABOLITES, SEED
from data.features import to_bins300, to_roi
from data.preprocess import hash_dataset, preprocess_spectra
from eval._fit_predict import kfold_oof_predictions
from models.base_models import (
    BASE_MODEL_REGISTRY,
    get_model_factory,
    get_recommended_features,
)
from models.meta_learners import MetaStack

log = logging.getLogger(__name__)


@dataclass
class PipelineManifest:
    """Reproducibility metadata saved next to the fitted pipeline."""
    code_version: str = ""             # git short hash, set by caller
    python_version: str = ""
    training_data_hash: str = ""
    metabolite_names: tuple = OCM_METABOLITES
    feature_repr_for_meta: str = "concat"      # how meta features are formed
    base_models: tuple = ()
    n_train_samples: int = 0
    n_features_per_repr: dict = field(default_factory=dict)
    timestamp: str = ""


class NMRPipeline:
    """Train-once-deploy-many pipeline with bins300/roi features and a META-RF stacking head."""

    def __init__(self,
                  base_model_names: Optional[tuple] = None,
                  meta_kind: str = "rf",
                  meta_scheme: str = "cross",
                  metabolites: tuple = OCM_METABOLITES,
                  use_scaler: bool = True):
        self.base_model_names = (base_model_names
                                  or tuple(BASE_MODEL_REGISTRY.keys()))
        self.meta_kind = meta_kind
        self.meta_scheme = meta_scheme
        self.metabolites = metabolites
        self.use_scaler = use_scaler

        # Fitted state (set by fit())
        self._fitted_bases: list[tuple[str, str, list]] = []
        self._meta: Optional[MetaStack] = None
        self._scalers: dict = {}            # per-feature-repr scaler
        self._feature_grid: dict = {}       # bin centres / ppm grids
        self._water_excluded_ppm: Optional[np.ndarray] = None
        self.manifest = PipelineManifest()

    def fit(self,
             X_raw: np.ndarray,
             ppm: np.ndarray,
             Y: np.ndarray,
             *,
             code_version: str = "",
             oof_n_folds: int = 5,
             seed: int = SEED) -> "NMRPipeline":
        """Train all base models, generate OOF predictions, fit the meta."""
        from sklearn.preprocessing import StandardScaler

        log.info("Training pipeline on %d samples", X_raw.shape[0])
        X_clean, ppm_clean = preprocess_spectra(X_raw, ppm)
        self._water_excluded_ppm = ppm_clean

        X_bins, bin_centres = to_bins300(X_clean, ppm_clean)
        X_roi_, _ = to_roi(X_clean, ppm_clean)
        feature_dict = {"bins300": X_bins, "roi": X_roi_}
        self._feature_grid["bin_centres"] = bin_centres
        self._feature_grid["ppm_water_excluded"] = ppm_clean

        for repr_name, X in feature_dict.items():
            sc = StandardScaler()
            sc.fit(X)
            self._scalers[repr_name] = sc

        n_samples = X_clean.shape[0]
        n_met = Y.shape[1]
        n_base = len(self.base_model_names)
        oof_preds = np.zeros((n_samples, n_base, n_met), dtype=np.float32)

        for j, name in enumerate(self.base_model_names):
            factory = get_model_factory(name)
            repr_name = get_recommended_features(name)
            X_repr = feature_dict[repr_name]
            X_repr_s = self._scalers[repr_name].transform(X_repr)
            log.info("  base[%d/%d]: %s (%s)", j + 1, n_base, name, repr_name)
            Y_oof_j = kfold_oof_predictions(
                X_repr_s, Y, factory,
                n_folds=oof_n_folds, seed=seed, use_scaler=False)
            oof_preds[:, j, :] = Y_oof_j

            # Fit each base model on the FULL training set (per-metabolite, ignoring NaN rows).
            fitted_per_met: list = []
            for k in range(n_met):
                mk = ~np.isnan(Y[:, k])
                if mk.sum() < 10:
                    fitted_per_met.append(None)
                    continue
                m = factory()
                try:
                    m.fit(X_repr_s[mk], Y[mk, k])
                    fitted_per_met.append(m)
                except Exception as e:
                    log.warning("Fit failed (base=%s, met=%d): %s",
                                 name, k, e)
                    fitted_per_met.append(None)
            self._fitted_bases.append((name, repr_name, fitted_per_met))

        log.info("Fitting meta-learner: %s / %s", self.meta_kind, self.meta_scheme)
        self._meta = MetaStack(meta_kind=self.meta_kind,
                                scheme=self.meta_scheme)
        self._meta.fit(oof_preds, Y)

        self.manifest = PipelineManifest(
            code_version=code_version,
            python_version=sys.version.split()[0],
            training_data_hash=hash_dataset(X_clean, Y, ppm_clean),
            metabolite_names=tuple(self.metabolites),
            feature_repr_for_meta=self.meta_scheme,
            base_models=tuple(self.base_model_names),
            n_train_samples=int(n_samples),
            n_features_per_repr={k: int(v.shape[1])
                                 for k, v in feature_dict.items()},
            timestamp=str(np.datetime64("now")),
        )
        return self

    def predict(self,
                 X_raw: np.ndarray,
                 ppm: np.ndarray) -> np.ndarray:
        """Predict on new spectra. Returns (n_samples, n_metabolites)."""
        if self._meta is None:
            raise RuntimeError("NMRPipeline not fitted")

        X_clean, ppm_clean = preprocess_spectra(X_raw, ppm)
        # Sanity: same ppm grid as training (length tolerance).
        if (self._water_excluded_ppm is not None and
                ppm_clean.shape != self._water_excluded_ppm.shape):
            log.warning(
                "ppm grid mismatch: predict has %d points, train had %d. "
                "Predictions may be unreliable; reproject upstream.",
                ppm_clean.shape[0], self._water_excluded_ppm.shape[0])

        X_bins, _ = to_bins300(X_clean, ppm_clean)
        X_roi_, _ = to_roi(X_clean, ppm_clean)
        feature_dict = {"bins300": X_bins, "roi": X_roi_}

        n = X_raw.shape[0]
        n_met = len(self.metabolites)
        n_base = len(self._fitted_bases)
        base_preds = np.zeros((n, n_base, n_met), dtype=np.float32)

        for j, (name, repr_name, fitted_list) in enumerate(self._fitted_bases):
            X_repr = feature_dict[repr_name]
            X_repr_s = self._scalers[repr_name].transform(X_repr)
            for k, m in enumerate(fitted_list):
                if m is None:
                    continue
                try:
                    base_preds[:, j, k] = np.maximum(
                        m.predict(X_repr_s), 0).astype(np.float32)
                except Exception as e:
                    log.debug("Predict failed (base=%s, met=%d): %s",
                              name, k, e)

        return self._meta.predict(base_preds)

    def save(self, path: str | Path):
        """Pickle the fitted pipeline; write manifest as JSON sidecar."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)
        manifest_path = path.with_suffix(".manifest.json")
        with manifest_path.open("w", encoding="utf-8") as f:
            d = asdict(self.manifest)
            d["metabolite_names"] = list(d["metabolite_names"])
            d["base_models"] = list(d["base_models"])
            json.dump(d, f, indent=2)
        log.info("Saved pipeline to %s + manifest", path)

    @staticmethod
    def load(path: str | Path) -> "NMRPipeline":
        """Load a pickled pipeline."""
        with Path(path).open("rb") as f:
            pipe = pickle.load(f)
        if not isinstance(pipe, NMRPipeline):
            raise TypeError(f"File at {path} did not unpickle to NMRPipeline")
        return pipe


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s")
    parser = argparse.ArgumentParser(
        description="Run NMRPipeline.predict on aligned spectra.")
    parser.add_argument("--pipeline", required=True, type=Path,
                         help="Path to fitted .pkl pipeline")
    parser.add_argument("--spectra", required=True, type=Path,
                         help="NPZ file with X (n,p) and ppm (p,)")
    parser.add_argument("--out", required=True, type=Path,
                         help="CSV output path for predictions")
    args = parser.parse_args()

    pipe = NMRPipeline.load(args.pipeline)
    d = np.load(args.spectra, allow_pickle=True)
    X = d["X"].astype(np.float32)
    ppm = d["ppm"].astype(np.float32)
    sids = [str(s) for s in d.get("sids", np.arange(X.shape[0]))]
    Y_pred = pipe.predict(X, ppm)
    df = pd.DataFrame(Y_pred, columns=list(pipe.metabolites))
    df.insert(0, "sid", sids)
    df.to_csv(args.out, index=False)
    log.info("Wrote %d predictions to %s", len(df), args.out)
