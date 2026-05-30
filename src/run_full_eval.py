"""Master orchestrator for the full evaluation campaign.

Runs pretests, direct supervision (quick CV, bootstrap-OOB, learning curve, augmentation,
pairwise Wilcoxon), the optional LLM-spec experiment, and downstream figures/tables/reports.
Each major step writes a CSV under results/eval/, so --resume skips already-done steps.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    LC_SIZES,
    LOGS,
    N_BOOTSTRAPS_OOB,
    N_FOLDS,
    N_JOBS_OUTER,
    N_REPEATS,
    OCM_METABOLITES,
    RESULTS_EVAL,
    SEED,
    assert_data_present,
)
from data import load_cohort, preprocess_spectra, to_bins300, to_roi
from eval import (
    augmentation_eval,
    bootstrap_oob,
    cv_simple,
    pairwise_wilcoxon,
    subsampling_lc,
)
from eval._fit_predict import kfold_oof_predictions
from eval.metrics import aggregate_per_metabolite, compute_metrics
from models.base_models import (
    BASE_MODEL_REGISTRY,
    get_model_factory,
    get_recommended_features,
    list_models,
)
from models.meta_learners import all_meta_variants

log = logging.getLogger(__name__)


def _setup_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, mode="a", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        handlers=handlers,
        force=True,
    )


def _build_feature_dict(X_clean: np.ndarray, ppm_clean: np.ndarray
                          ) -> dict[str, np.ndarray]:
    X_bins, _ = to_bins300(X_clean, ppm_clean)
    X_roi_, _ = to_roi(X_clean, ppm_clean)
    return {"bins300": X_bins, "roi": X_roi_}


def _model_factories_keyed_by_name() -> dict[str, callable]:
    """Return a dict {model_name: factory} for all registered base models."""
    return {n: get_model_factory(n) for n in list_models()}


def _evaluate_base_models_quick(X_dict: dict, Y: np.ndarray) -> pd.DataFrame:
    """Quick 5-fold CV on each base model with its recommended features."""
    rows = []
    for name in list_models():
        factory = get_model_factory(name)
        repr_name = get_recommended_features(name)
        log.info("  quick CV: %s on %s features", name, repr_name)
        _, summary = cv_simple(X_dict[repr_name], Y, factory)
        rows.append({"model": name, "feature_repr": repr_name, **summary})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-pretests", action="store_true")
    parser.add_argument("--skip-bootstrap-oob", action="store_true")
    parser.add_argument("--skip-lc", action="store_true")
    parser.add_argument("--skip-augmentation", action="store_true")
    parser.add_argument("--enable-llm-spec", action="store_true")
    parser.add_argument("--resume", action="store_true",
                          help="Skip steps with existing CSVs")
    parser.add_argument("--quick", action="store_true",
                          help="Small B (=20) for fast smoke tests")
    parser.add_argument("--n-jobs", type=int, default=N_JOBS_OUTER)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    log_path = LOGS / f"run_full_eval_{int(time.time())}.log"
    _setup_logging(log_path)

    log.info("run_full_eval starting | log: %s", log_path)

    assert_data_present()
    log.info("Loading cohort")
    data = load_cohort()
    X_raw = data["X_spectra"]
    Y = data["Y_chenomx"]
    ppm = data["ppm"]
    sids = data["sids"]
    metabolites = data["metabolite_names"]
    log.info("Loaded: X=%s, Y=%s, n_metabolites=%d",
             X_raw.shape, Y.shape, len(metabolites))

    X_clean, ppm_clean = preprocess_spectra(X_raw, ppm)
    X_dict = _build_feature_dict(X_clean, ppm_clean)
    log.info("Feature reprs: bins300=%s, roi=%s",
             X_dict["bins300"].shape, X_dict["roi"].shape)

    if not args.skip_pretests:
        log.info("Pretests A-D")
        from pretests import pretest_a, pretest_b, pretest_c, pretest_d
        for mod in (pretest_a, pretest_b, pretest_c, pretest_d):
            try:
                mod.__name__  # noqa
                log.info(" Running %s", mod.__name__)
                fn_name = f"run_{mod.__name__.split('.')[-1].split('_')[1]}"
                fn = getattr(mod, fn_name, None)
                if fn:
                    fn(load_dict=data, seed=args.seed)
            except Exception as e:
                log.error("Pretest %s failed: %s", mod.__name__, e)

    log.info("Direct supervision")

    quick_csv = RESULTS_EVAL / "base_models_quick.csv"
    if args.resume and quick_csv.exists():
        log.info("Skipping quick CV (resume): %s exists", quick_csv)
    else:
        df_quick = _evaluate_base_models_quick(X_dict, Y)
        df_quick.to_csv(quick_csv, index=False)
        log.info("Saved %s\n%s", quick_csv,
                 df_quick.sort_values("spearman_mean",
                                       ascending=False).to_string(index=False))

    oob_csv = RESULTS_EVAL / "oob_master.csv"
    n_boot = 20 if args.quick else N_BOOTSTRAPS_OOB
    if args.skip_bootstrap_oob or (args.resume and oob_csv.exists()):
        log.info("Skipping bootstrap-OOB")
    else:
        log.info("Bootstrap-OOB B=%d on %d cores", n_boot, args.n_jobs)
        # ROI features for the main protocol (best for non-linear models).
        base_factories: dict[str, callable] = {}
        for name in list_models():
            base_factories[name] = get_model_factory(name)

        df_master, aggregated, per_boot = bootstrap_oob(
            X_dict["roi"], Y, base_factories,
            n_bootstraps=n_boot, n_jobs=args.n_jobs, seed=args.seed,
            return_per_bootstrap=True)
        df_master.to_csv(oob_csv, index=False)
        log.info("Saved %s", oob_csv)

        # Per-bootstrap (long form) for downstream Wilcoxon.
        long_rows = []
        for name, recs in per_boot.items():
            for r in recs:
                long_rows.append({"model": name, **r})
        pd.DataFrame(long_rows).to_csv(
            RESULTS_EVAL / "oob_per_bootstrap.csv", index=False)

        np.savez(RESULTS_EVAL / "oof_predictions.npz",
                 Y=Y,
                 sids=np.array(sids),
                 **{f"pred_{name}": pred
                    for name, pred in aggregated.items()})

        log.info("Pairwise Wilcoxon (Bonferroni-Holm)")
        df_wil = pairwise_wilcoxon(per_boot, metric="spearman_mean")
        df_wil.to_csv(RESULTS_EVAL / "wilcoxon.csv", index=False)
        log.info("Saved %s", RESULTS_EVAL / "wilcoxon.csv")

    lc_csv = RESULTS_EVAL / "lc_master.csv"
    if args.skip_lc or (args.resume and lc_csv.exists()):
        log.info("Skipping LC")
    else:
        log.info("Subsampling LC on sizes %s", LC_SIZES)
        # A few representative models for the LC.
        lc_factories = {n: get_model_factory(n)
                        for n in ("ridge", "rf") if n in BASE_MODEL_REGISTRY}
        df_lc = subsampling_lc(
            X_dict["roi"], Y, lc_factories,
            sizes=LC_SIZES, n_per_size=10 if args.quick else 20,
            n_jobs=args.n_jobs, seed=args.seed)
        df_lc.to_csv(lc_csv, index=False)
        log.info("Saved %s", lc_csv)

    aug_csv = RESULTS_EVAL / "augmentation_master.csv"
    if args.skip_augmentation or (args.resume and aug_csv.exists()):
        log.info("Skipping augmentation comparison")
    else:
        log.info("Augmentation comparison")
        aug_factories = {n: get_model_factory(n)
                         for n in ("ridge", "rf", "knn")
                         if n in BASE_MODEL_REGISTRY}
        df_aug = augmentation_eval(
            X_dict["roi"], Y, aug_factories,
            n_jobs=args.n_jobs, seed=args.seed,
            n_repeats=2 if args.quick else N_REPEATS)
        df_aug.to_csv(aug_csv, index=False)
        log.info("Saved %s", aug_csv)

    if args.enable_llm_spec:
        log.info("LLM-spec experiment")
        try:
            from llm_spec_experiment.generate_spec import generate_spec
            from llm_spec_experiment.run_comparison import run_comparison
            generate_spec(metabolites=metabolites, n_samples=3, temperature=0.2)
            run_comparison(seed=args.seed)
        except Exception as e:
            log.error("LLM-spec experiment failed: %s", e)

    log.info("Figures + tables")
    try:
        from reports.make_figures import make_all as make_figures_all
        make_figures_all()
    except Exception as e:
        log.error("Figures failed: %s", e)
    try:
        from reports.make_tables import make_all as make_tables_all
        make_tables_all()
    except Exception as e:
        log.error("Tables failed: %s", e)

    log.info("Per-metabolite report")
    try:
        oof_npz = RESULTS_EVAL / "oof_predictions.npz"
        if oof_npz.exists():
            from reports.per_metabolite_report import make_report
            d = np.load(oof_npz, allow_pickle=True)
            best_key = "pred_rf" if "pred_rf" in d.files else None
            if best_key is None and len(d.files) > 1:
                pred_keys = [k for k in d.files if k.startswith("pred_")]
                if pred_keys:
                    best_key = pred_keys[0]
            if best_key is not None:
                make_report(d["Y"], d[best_key])
    except Exception as e:
        log.error("Per-metabolite report failed: %s", e)

    log.info("run_full_eval done")


if __name__ == "__main__":
    main()
