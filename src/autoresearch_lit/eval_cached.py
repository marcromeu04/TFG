"""Cached evaluator for the auto-research agent (final version).

evaluate_spec(spec, model='ridge_bins', n_synth=5000, seed=42) -> metrics dict

Supported models (10 total):
    - ridge_bins, pls_bins, lasso_bins, enet_bins (bins300 features, 4)
    - rf_roi, knn_roi, svr_roi, gpr_roi, xgb_roi (ROI features, 5)
    - ensemble_top3 (mean of ridge_bins + pls_bins + rf_roi)

The spec is a dict with keys: ranges, correlations, baseline_kind, noise_shift.
"""
from __future__ import annotations
import hashlib, json, sys, logging
from pathlib import Path
import numpy as np

sys.path.insert(0, '.')

from data import load_cohort, preprocess_spectra, to_bins300, to_roi
from data.preprocess import _normalize_integral
from eval._fit_predict import fit_predict
from eval.metrics import compute_metrics
from models.base_models import (make_ridge, make_pls, make_lasso, make_elasticnet,
                                  make_rf, make_knn, make_svr, make_gpr, make_xgb)
from synth import (GeneratorConfig, generate_synthetic, build_residual_library,
                    use_ranges_override)
from synth.llm_spec_loader import parse_llm_spec
from config import CHENOMX_TO_ASICS

log = logging.getLogger("eval_cached")

_CACHE_DATA = {}
_CACHE_RESULTS = {}

def _load_data():
    if "loaded" in _CACHE_DATA:
        return _CACHE_DATA
    log.info("Loading cohort data...")
    d = load_cohort()
    Xc, ppmc = preprocess_spectra(d['X_spectra'], d['ppm'],
                                    exclude_water=False, normalize=False)
    Xb, _ = to_bins300(Xc, ppmc)
    Xroi, _ = to_roi(Xc, ppmc)
    Y = d['Y_chenomx']
    templates = d['asics_templates']
    metabolites = d['metabolite_names']

    X_recon = np.zeros_like(Xc)
    for k, met in enumerate(metabolites):
        asics = CHENOMX_TO_ASICS.get(met, met)
        if asics not in templates: continue
        c = Y[:, k]
        med = np.nanmedian(c) if not np.all(np.isnan(c)) else 0.0
        X_recon = X_recon + np.outer(np.where(np.isnan(c), med, c), templates[asics])
    RES_LIB, RES_PCA = build_residual_library(Xc, X_recon, n_pca_components=5)

    _CACHE_DATA.update({
        "loaded": True, "Xc": Xc, "ppmc": ppmc,
        "Xb": Xb, "Xroi": Xroi,
        "Y": Y, "templates": templates, "metabolites": metabolites,
        "RES_LIB": RES_LIB, "RES_PCA": RES_PCA,
    })
    log.info("Data loaded.")
    return _CACHE_DATA

MODELS = {
    "ridge_bins":  ("bins", make_ridge),
    "pls_bins":    ("bins", make_pls),
    "lasso_bins":  ("bins", make_lasso),
    "enet_bins":   ("bins", make_elasticnet),
    "rf_roi":      ("roi",  make_rf),
    "knn_roi":     ("roi",  make_knn),
    "svr_roi":     ("roi",  make_svr),
    "gpr_roi":     ("roi",  make_gpr),
    "xgb_roi":     ("roi",  make_xgb),
}
ALLOWED_MODELS = list(MODELS.keys()) + ["ensemble_top3"]

def _spec_hash(spec, model, seed, n_synth):
    canonical = {
        "ranges": spec.get("ranges", {}),
        "correlations": spec.get("correlations", {}),
        "baseline_kind": spec.get("baseline_kind", "polynomial"),
        "noise_shift": spec.get("noise_shift", {}),
        "model": model, "seed": seed, "n_synth": n_synth,
    }
    blob = json.dumps(canonical, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]

def _predict_one_model(model_name, X_synth_b, X_synth_roi, Xb, Xroi, Ys):
    feat, factory = MODELS[model_name]
    if feat == "bins":
        return fit_predict(X_synth_b, Xb, Ys, factory)
    return fit_predict(X_synth_roi, Xroi, Ys, factory)

def evaluate_spec(spec: dict, model: str = "ridge_bins",
                   n_synth: int = 5000, seed: int = 42) -> dict:
    """Evaluate spec with selected model."""
    if model not in ALLOWED_MODELS:
        return {"error": f"unknown model: {model}. Allowed: {ALLOWED_MODELS}",
                 "from_cache": False}
    key = _spec_hash(spec, model, seed, n_synth)
    if key in _CACHE_RESULTS:
        cached = dict(_CACHE_RESULTS[key])
        cached["from_cache"] = True
        return cached

    d = _load_data()
    metabolites = d["metabolites"]
    try:
        cfg, ranges, R = parse_llm_spec(spec, metabolites=metabolites,
                                          n_samples=n_synth, seed=seed)
    except Exception as e:
        return {"error": f"parse_llm_spec failed: {e}", "from_cache": False}
    cfg.residual_library = d["RES_LIB"]
    cfg.residual_pca = d["RES_PCA"]
    try:
        with use_ranges_override(ranges):
            Xs, Ys = generate_synthetic(cfg, d["templates"], d["ppmc"],
                                          metabolites=metabolites,
                                          Y_for_chenomx=d["Y"],
                                          correlation_matrix=R)
        Xs = _normalize_integral(Xs.astype(np.float64)).astype(np.float32)
        Xsb, _ = to_bins300(Xs, d["ppmc"])
        Xsroi, _ = to_roi(Xs, d["ppmc"])
    except Exception as e:
        return {"error": f"generate failed: {e}", "from_cache": False}

    try:
        if model == "ensemble_top3":
            preds = []
            for m_name in ["ridge_bins", "pls_bins", "rf_roi"]:
                Y_pred = _predict_one_model(m_name, Xsb, Xsroi,
                                              d["Xb"], d["Xroi"], Ys)
                preds.append(Y_pred)
            Y_pred = np.mean(np.stack(preds, axis=0), axis=0)
        else:
            Y_pred = _predict_one_model(model, Xsb, Xsroi,
                                          d["Xb"], d["Xroi"], Ys)
    except Exception as e:
        return {"error": f"predict failed: {e}", "from_cache": False}

    per_met = {}
    for k, met in enumerate(metabolites):
        m = compute_metrics(d["Y"][:, k], Y_pred[:, k])
        per_met[met] = {
            "spearman": float(m["spearman"]) if not np.isnan(m["spearman"]) else None,
            "pearson_log": float(m["pearson_log"]) if not np.isnan(m["pearson_log"]) else None,
            "n": int(m["n"]),
            "mape_pct": float(m["mape_pct"]) if not np.isnan(m["mape_pct"]) else None,
        }
    valid_rhos = [v["spearman"] for v in per_met.values() if v["spearman"] is not None]
    n_valid = len(valid_rhos)
    mean_rho = float(np.mean(valid_rhos)) if valid_rhos else float("nan")

    result = {
        "spearman_mean_valid": mean_rho,
        "n_metabolites_valid": n_valid,
        "model_used": model,
        "per_metabolite": per_met,
        "from_cache": False,
    }
    _CACHE_RESULTS[key] = dict(result)
    return result

if __name__ == "__main__":
    import time
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')
    HMDB = {
        "Lactate":(1.5, 0.6), "Histidine":(0.075,0.020), "Cysteine":(0.060,0.020),
        "Glucose":(5.0, 1.0), "Glycine":(0.250,0.060), "Betaine":(0.040,0.015),
        "Pyruvate":(0.080,0.030), "Threonine":(0.140,0.040), "Serine":(0.120,0.030),
        "Choline":(0.010,0.005), "Creatine":(0.045,0.015), "Creatinine":(0.080,0.020),
    }
    spec = {
        "ranges": {k: [float(np.log10(m)), max(float(s/(m*np.log(10))), 0.05)]
                    for k, (m, s) in HMDB.items()},
        "correlations": {},
        "baseline_kind": "polynomial",
        "noise_shift": {"noise_amp": 0.005, "shift_amp": 0.001},
    }
    print(f"Allowed models ({len(ALLOWED_MODELS)}): {ALLOWED_MODELS}")
    print()
    for m in ["ridge_bins", "rf_roi", "ensemble_top3"]:
        t0 = time.time()
        r = evaluate_spec(spec, model=m)
        if "error" in r:
            print(f"{m:<18}: ERROR {r['error']}")
        else:
            print(f"{m:<18}: {time.time()-t0:>6.1f}s, "
                  f"n_valid={r['n_metabolites_valid']}, "
                  f"mean={r['spearman_mean_valid']:>+.3f}, "
                  f"cache={r['from_cache']}")
