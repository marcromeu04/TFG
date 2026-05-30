"""Data layer tests: preprocess_spectra, feature builders, and active_rois water filtering."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from data.features import to_bins300, to_fullspec, to_roi
from data.metabolite_rois import ROIS, active_rois
from data.preprocess import hash_dataset, preprocess_spectra


@pytest.fixture
def fake_spectra():
    """Tiny synthetic dataset for unit tests."""
    np.random.seed(42)
    n, p = 20, 500
    ppm = np.linspace(0, 10, p).astype(np.float32)
    X = np.random.randn(n, p).astype(np.float32) + 5.0  # positive baseline
    Y = np.abs(np.random.randn(n, 12).astype(np.float32))
    return X, ppm, Y


def test_preprocess_excludes_water(fake_spectra):
    X, ppm, _ = fake_spectra
    X_clean, ppm_clean = preprocess_spectra(X, ppm)
    assert np.all((ppm_clean < 4.65) | (ppm_clean > 4.95))
    assert ppm_clean.shape[0] < ppm.shape[0]
    assert X_clean.shape[0] == X.shape[0]


def test_preprocess_integral_normalisation(fake_spectra):
    X, ppm, _ = fake_spectra
    X_clean, _ = preprocess_spectra(X, ppm, normalize=True)
    row_sums = np.sum(np.abs(X_clean), axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)


def test_preprocess_disable_normalisation(fake_spectra):
    X, ppm, _ = fake_spectra
    X_clean, _ = preprocess_spectra(X, ppm, normalize=False)
    row_sums = np.sum(np.abs(X_clean), axis=1)
    assert np.all(row_sums > 1.5)


def test_hash_dataset_stable(fake_spectra):
    X, ppm, Y = fake_spectra
    h1 = hash_dataset(X, Y, ppm)
    h2 = hash_dataset(X, Y, ppm)
    assert h1 == h2
    Y2 = Y * 2
    h3 = hash_dataset(X, Y2, ppm)
    assert h1 != h3


def test_hash_dataset_nan_safe(fake_spectra):
    X, ppm, Y = fake_spectra
    Y_with_nan = Y.copy()
    Y_with_nan[0, 0] = np.nan
    h = hash_dataset(X, Y_with_nan, ppm)
    assert isinstance(h, str) and len(h) == 32


def test_to_bins300_shape(fake_spectra):
    X, ppm, _ = fake_spectra
    X_clean, ppm_clean = preprocess_spectra(X, ppm)
    Xb, centres = to_bins300(X_clean, ppm_clean, n_bins=100)
    assert Xb.shape == (X.shape[0], 100)
    assert centres.shape == (100,)


def test_to_roi_features(fake_spectra):
    X, ppm, _ = fake_spectra
    X_clean, ppm_clean = preprocess_spectra(X, ppm)
    F, meta = to_roi(X_clean, ppm_clean)
    n_active = len(active_rois(4.65, 4.95))
    assert F.shape == (X.shape[0], 3 * n_active)
    assert len(meta) == 3 * n_active


def test_active_rois_exclusive():
    """No active ROI should sit inside the water region."""
    rois = active_rois(4.65, 4.95)
    for met, lo, hi, _ in rois:
        mid = 0.5 * (lo + hi)
        assert not (4.65 <= mid <= 4.95), \
            f"ROI for {met} at {mid:.2f} ppm is in water region"


def test_to_fullspec_identity(fake_spectra):
    X, ppm, _ = fake_spectra
    Xc, ppmc = preprocess_spectra(X, ppm)
    Xf, ppmf = to_fullspec(Xc, ppmc)
    assert Xf.shape == Xc.shape
    np.testing.assert_array_equal(Xf, Xc.astype(np.float32))
