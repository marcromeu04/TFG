"""Model layer tests: base model factories, augmentation, ensembles, MetaStack."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from models.augmentation import apply_augmentation
from models.base_models import (
    BASE_MODEL_REGISTRY,
    get_model_factory,
    list_models,
)
from models.ensembles import (
    average_all,
    average_topk,
    bagging_predict,
    weighted_blend_ccc,
)
from models.meta_learners import MetaStack, all_meta_variants


@pytest.fixture
def small_dataset():
    """Reasonable size for fast model tests"""
    np.random.seed(42)
    n, p, n_met = 80, 30, 4
    X = np.random.randn(n, p).astype(np.float32)
    W = np.random.randn(p, n_met).astype(np.float32) * 0.5
    Y = np.maximum(X @ W + np.random.randn(n, n_met) * 0.1 + 1.0,
                    0.01).astype(np.float32)
    mask = np.random.rand(n, n_met) < 0.05
    Y[mask] = np.nan
    return X, Y


def test_all_factories_callable():
    for name in list_models():
        factory = get_model_factory(name)
        m = factory()
        assert hasattr(m, "fit") and hasattr(m, "predict"), \
            f"{name} estimator lacks fit/predict"


def test_factories_can_train_and_predict(small_dataset):
    X, Y = small_dataset
    y = Y[:, 0]
    mask = ~np.isnan(y)
    for name in list_models():
        factory = get_model_factory(name)
        # GPR skipped in tests (cubic in n).
        if name == "gpr":
            continue
        m = factory()
        m.fit(X[mask], y[mask])
        preds = m.predict(X[mask])
        assert preds.shape[0] == mask.sum(), f"{name} pred shape wrong"


def test_augmentation_target_size(small_dataset):
    X, Y = small_dataset
    for strategy in ("baseline", "bootstrap", "jitter", "mixup"):
        X_aug, Y_aug = apply_augmentation(strategy, X, Y,
                                            target_size=200, seed=42)
        if strategy == "baseline":
            assert X_aug.shape[0] == X.shape[0]
        else:
            assert X_aug.shape[0] == 200, f"{strategy} target_size failed"
        assert Y_aug.shape[0] == X_aug.shape[0]


def test_augmentation_preserves_feature_dim(small_dataset):
    X, Y = small_dataset
    for strategy in ("bootstrap", "jitter", "mixup"):
        X_aug, _ = apply_augmentation(strategy, X, Y,
                                        target_size=100, seed=42)
        assert X_aug.shape[1] == X.shape[1]


def test_average_topk(small_dataset):
    X, Y = small_dataset
    n, n_met = X.shape[0], Y.shape[1]
    n_base = 5
    base_preds = np.random.rand(n, n_base, n_met).astype(np.float32)
    base_scores = np.random.rand(n_base, n_met)
    out = average_topk(base_preds, base_scores, k=3)
    assert out.shape == (n, n_met)
    out_all = average_all(base_preds)
    assert out_all.shape == (n, n_met)


def test_bagging_runs(small_dataset):
    X, Y = small_dataset
    from models.base_models import make_ridge
    preds = bagging_predict(make_ridge, X[:60], Y[:60], X[60:],
                              n_estimators=5, seed=42)
    assert preds.shape == (X.shape[0] - 60, Y.shape[1])


def test_weighted_blend_ccc(small_dataset):
    X, Y = small_dataset
    n_train, n_test = 60, 20
    n_base = 4
    n_met = Y.shape[1]
    oof = np.random.rand(n_train, n_base, n_met).astype(np.float32)
    test = np.random.rand(n_test, n_base, n_met).astype(np.float32)
    blend, w = weighted_blend_ccc(oof, Y[:n_train], test,
                                    n_random=10, seed=42)
    assert blend.shape == (n_test, n_met)
    assert w.shape == (n_base, n_met)
    assert np.all(w >= 0)
    np.testing.assert_allclose(w.sum(axis=0), 1.0, atol=1e-4)


def test_metastack_fit_predict(small_dataset):
    X, Y = small_dataset
    n_train, n_test = 60, 20
    n_base = 5
    n_met = Y.shape[1]
    oof = np.random.rand(n_train, n_base, n_met).astype(np.float32)
    test = np.random.rand(n_test, n_base, n_met).astype(np.float32)

    for kind in ("ridge", "rf"):
        for scheme in ("per", "cross"):
            ms = MetaStack(meta_kind=kind, scheme=scheme)
            ms.fit(oof, Y[:n_train])
            pred = ms.predict(test)
            assert pred.shape == (n_test, n_met), \
                f"{kind}/{scheme}: shape {pred.shape}"
            assert np.all(pred >= 0)


def test_all_meta_variants():
    variants = all_meta_variants()
    # ridge x {per, cross} + rf x {per, cross} (4 without xgboost).
    assert len(variants) >= 4
    kinds = {v.meta_kind for v in variants}
    assert "ridge" in kinds and "rf" in kinds
