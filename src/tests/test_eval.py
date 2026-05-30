"""Tests for the evaluation framework."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval._fit_predict import fit_predict, kfold_oof_predictions
from eval.cv_simple import cv_simple
from eval.metrics import (
    aggregate_per_metabolite,
    ccc,
    compute_metrics,
    mape_pct,
    pearson_log,
    spearman,
)
from eval.significance import _holm_bonferroni, pairwise_wilcoxon
from models.base_models import make_ridge


class TestMetrics:
    def test_perfect_prediction_spearman_one(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        assert spearman(y, y) == pytest.approx(1.0)

    def test_perfect_prediction_pearson_log_one(self):
        y = np.array([0.1, 1.0, 10.0, 100.0, 1000.0])
        assert pearson_log(y, y) == pytest.approx(1.0)

    def test_perfect_prediction_ccc_one(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        assert ccc(y, y) == pytest.approx(1.0, abs=1e-3)

    def test_perfect_prediction_mape_zero(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        assert mape_pct(y, y) == pytest.approx(0.0)

    def test_random_prediction_low_correlation(self):
        rng = np.random.RandomState(42)
        y_true = rng.rand(50) + 0.1
        y_pred = rng.rand(50) + 0.1
        s = spearman(y_true, y_pred)
        assert -0.5 < s < 0.5

    def test_metrics_handle_nan(self):
        # MIN_N=5 valid pairs required after NaN filtering.
        y_true = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0])
        y_pred = np.array([1.1, 1.9, 3.0, np.nan, 5.1, 5.9, 7.0, 7.9])
        m = compute_metrics(y_true, y_pred)
        assert m["n"] == 6
        assert not np.isnan(m["spearman"])

    def test_metrics_too_few_points(self):
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([1.5, 1.5])
        m = compute_metrics(y_true, y_pred)
        assert np.isnan(m["spearman"])

    def test_negative_excluded(self):
        # Concentrations must be positive; negatives are excluded.
        y_true = np.array([-1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        y_pred = np.array([1.5, 2.0, 3.1, 4.1, 4.9, 6.0])
        m = compute_metrics(y_true, y_pred)
        assert m["n"] == 5

    def test_aggregate_per_metabolite(self):
        rng = np.random.RandomState(0)
        Y_true = rng.rand(50, 4) + 0.1
        Y_pred = Y_true + rng.normal(0, 0.05, size=Y_true.shape)
        agg = aggregate_per_metabolite(Y_true, Y_pred)
        assert agg["n_metabolites_scored"] == 4
        assert agg["spearman_mean"] > 0.7


class TestFitPredict:
    def setup_method(self):
        rng = np.random.RandomState(42)
        self.X_tr = rng.randn(60, 20).astype(np.float32)
        self.X_te = rng.randn(20, 20).astype(np.float32)
        W = rng.randn(20, 4).astype(np.float32) * 0.5
        self.Y_tr = np.maximum(self.X_tr @ W + 1.0, 0.01).astype(np.float32)

    def test_basic_shape(self):
        Yp = fit_predict(self.X_tr, self.X_te, self.Y_tr, make_ridge)
        assert Yp.shape == (20, 4)

    def test_predictions_non_negative(self):
        Yp = fit_predict(self.X_tr, self.X_te, self.Y_tr, make_ridge,
                          non_negative=True)
        assert np.all(Yp >= 0)

    def test_with_nans_in_y(self):
        Y_tr_nan = self.Y_tr.copy()
        Y_tr_nan[:5, 0] = np.nan
        Y_tr_nan[:3, 2] = np.nan
        Yp = fit_predict(self.X_tr, self.X_te, Y_tr_nan, make_ridge)
        assert Yp.shape == (20, 4)
        assert not np.all(Yp == 0)

    def test_kfold_oof_shape(self):
        Y_oof = kfold_oof_predictions(self.X_tr, self.Y_tr,
                                       make_ridge, n_folds=5)
        assert Y_oof.shape == self.Y_tr.shape


class TestCVSimple:
    def test_runs_end_to_end(self):
        rng = np.random.RandomState(0)
        X = rng.randn(80, 15).astype(np.float32)
        W = rng.randn(15, 3).astype(np.float32) * 0.3
        Y = np.maximum(X @ W + 1.0, 0.01).astype(np.float32)
        Y_oof, summ = cv_simple(X, Y, make_ridge, n_folds=5)
        assert Y_oof.shape == Y.shape
        assert summ["n_metabolites_scored"] == 3
        assert summ["spearman_mean"] > 0.3


class TestHolmBonferroni:
    def test_single_p(self):
        p = np.array([0.04])
        adj = _holm_bonferroni(p)
        assert adj[0] == pytest.approx(0.04)

    def test_no_corrections_when_all_large(self):
        p = np.array([0.5, 0.6, 0.7])
        adj = _holm_bonferroni(p)
        assert adj[0] >= 0.5

    def test_corrections_applied(self):
        # Three p-values: 0.01, 0.04, 0.06 -> Holm: 0.03, 0.08, 0.08 (monotonic).
        p = np.array([0.01, 0.04, 0.06])
        adj = _holm_bonferroni(p)
        assert adj[0] == pytest.approx(0.03, abs=1e-6)
        sorted_adj = np.sort(adj)
        assert sorted_adj[0] <= sorted_adj[1] <= sorted_adj[2]

    def test_capped_at_one(self):
        p = np.array([0.4, 0.5, 0.6])
        adj = _holm_bonferroni(p)
        assert np.all(adj <= 1.0)
