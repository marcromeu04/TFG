"""Stacking meta-learners on top of base model OOF predictions.
Two axes: meta architecture {ridge, rf, xgb} and feature scheme {per-metabolite, cross-metabolite}."""
from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from config import SEED

log = logging.getLogger(__name__)


try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def make_meta_ridge() -> Ridge:
    return Ridge(alpha=1.0, random_state=SEED)


def make_meta_rf() -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=100,
                                  max_depth=6,
                                  min_samples_leaf=5,
                                  random_state=SEED,
                                  n_jobs=1)


def make_meta_xgb():
    if not HAS_XGBOOST:
        raise ImportError("xgboost not installed")
    return XGBRegressor(n_estimators=200,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=SEED,
                        n_jobs=1,
                        verbosity=0)


META_REGISTRY: dict[str, Callable] = {
    "ridge": make_meta_ridge,
    "rf":    make_meta_rf,
}
if HAS_XGBOOST:
    META_REGISTRY["xgb"] = make_meta_xgb


def shape_meta_features(base_preds: np.ndarray,
                        scheme: str = "cross") -> np.ndarray:
    """Reshape base OOF predictions for the meta. scheme is "per" or "cross"."""
    if scheme == "cross":
        n, n_base, n_met = base_preds.shape
        return base_preds.reshape(n, n_base * n_met)
    if scheme == "per":
        return base_preds.transpose(0, 2, 1)        # (n, n_met, n_base)
    raise ValueError(f"Unknown scheme: {scheme}")


@dataclass
class MetaStack:
    """One concrete (meta_kind, scheme) stacking model. One meta estimator per metabolite."""
    meta_kind: str
    scheme: str = "cross"
    metas_: list = None    # list of fitted estimators, one per metabolite

    def fit(self,
            base_preds_oof: np.ndarray,
            Y: np.ndarray) -> "MetaStack":
        """Fit one meta-estimator per metabolite. base_preds_oof is (n, n_base, n_met)."""
        if self.meta_kind not in META_REGISTRY:
            raise ValueError(f"Unknown meta_kind: {self.meta_kind}; "
                             f"available: {list(META_REGISTRY.keys())}")
        n_train, n_base, n_met = base_preds_oof.shape
        feats_full = shape_meta_features(base_preds_oof, scheme=self.scheme)

        self.metas_ = []
        for k in range(n_met):
            mk = ~np.isnan(Y[:, k])
            if mk.sum() < 10:
                self.metas_.append(None)
                continue

            if self.scheme == "cross":
                X_meta_k = feats_full[mk]                  # (n_valid, n_base * n_met)
            else:
                X_meta_k = feats_full[mk, k, :]            # (n_valid, n_base)

            try:
                m = META_REGISTRY[self.meta_kind]()
                m.fit(X_meta_k, Y[mk, k])
                self.metas_.append(m)
            except Exception as e:
                log.warning("meta fit failed for met=%d: %s", k, e)
                self.metas_.append(None)
        return self

    def predict(self, base_preds_test: np.ndarray) -> np.ndarray:
        """Predict on new base predictions. Metabolites without a fitted meta return 0."""
        if self.metas_ is None:
            raise RuntimeError("MetaStack not fitted")
        n_test, n_base, n_met = base_preds_test.shape
        feats_full = shape_meta_features(base_preds_test, scheme=self.scheme)
        out = np.zeros((n_test, n_met), dtype=np.float32)
        for k, meta in enumerate(self.metas_):
            if meta is None:
                continue
            if self.scheme == "cross":
                X_meta_k = feats_full
            else:
                X_meta_k = feats_full[:, k, :]
            try:
                out[:, k] = np.maximum(meta.predict(X_meta_k), 0).astype(np.float32)
            except Exception as e:
                log.debug("meta predict failed for met=%d: %s", k, e)
        return out


def all_meta_variants(include_xgb: bool = HAS_XGBOOST
                       ) -> list[MetaStack]:
    """Return one MetaStack per (meta_kind, scheme) combination."""
    variants = []
    for kind in ("ridge", "rf"):
        for scheme in ("per", "cross"):
            variants.append(MetaStack(meta_kind=kind, scheme=scheme))
    if include_xgb:
        for scheme in ("per", "cross"):
            variants.append(MetaStack(meta_kind="xgb", scheme=scheme))
    return variants
