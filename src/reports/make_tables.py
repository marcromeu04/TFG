"""Generate all tables from the CSVs in results/. Saved as CSV + Markdown."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config import (
    RESULTS_EVAL,
    RESULTS_LLM_SPEC,
    RESULTS_PRETESTS,
    RESULTS_REPORTS,
)

log = logging.getLogger(__name__)

OUT = RESULTS_REPORTS / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def _save(df, name):
    df.to_csv(OUT / f"{name}.csv", index=False)
    (OUT / f"{name}.md").write_text(
        df.to_markdown(index=False, floatfmt=".3f"), encoding="utf-8")
    log.info("saved %s", name)


def table_pretest_a():
    df = pd.read_csv(RESULTS_PRETESTS / "A" / "master.csv")
    cols = ["model", "range_regime", "real_spearman", "synth_spearman", "gap"]
    _save(df[cols], "pretest_a")


def table_pretest_b():
    df = pd.read_csv(RESULTS_PRETESTS / "B" / "master.csv")
    cols = ["ablation", "spearman_synth_real",
            "delta_spearman_vs_default", "mape_synth_real"]
    _save(df[cols], "pretest_b")


def table_pretest_c():
    df = pd.read_csv(RESULTS_PRETESTS / "C" / "master.csv")
    cols = ["regime", "spearman_mean", "pearson_log_mean",
            "ccc_mean", "mape_pct_mean"]
    _save(df[cols], "pretest_c")


def table_pretest_d():
    df = pd.read_csv(RESULTS_PRETESTS / "D" / "master.csv")
    cols = ["spec", "spearman_mean", "pearson_log_mean",
            "ccc_mean", "mape_pct_mean"]
    _save(df[cols], "pretest_d")


def table_best_per_family():
    df = pd.read_csv(RESULTS_EVAL / "oob_master.csv")
    cols = ["model", "spearman_mean", "spearman_ci_lo", "spearman_ci_hi",
            "pearson_log_mean", "ccc_mean", "mape_pct_mean"]
    _save(df[cols], "oob_master")


def table_per_metabolite():
    df = pd.read_csv(RESULTS_EVAL / "per_metabolite.csv")
    _save(df, "per_metabolite")


def table_wilcoxon():
    df = pd.read_csv(RESULTS_EVAL / "wilcoxon.csv")
    _save(df, "wilcoxon")


def table_augmentation():
    df = pd.read_csv(RESULTS_EVAL / "augmentation_master.csv")
    cols = ["strategy", "model", "spearman_mean", "pearson_log_mean",
            "mape_pct_mean"]
    _save(df[cols], "augmentation")


def table_lc():
    df = pd.read_csv(RESULTS_EVAL / "lc_master.csv")
    _save(df, "learning_curve")


def table_llm_spec_ablation():
    df = pd.read_csv(RESULTS_LLM_SPEC / "ablation.csv")
    cols = ["ablation", "spearman_mean",
            "delta_spearman_vs_full", "pearson_log_mean", "mape_pct_mean"]
    _save(df[cols], "llm_spec_ablation")


def make_all():
    table_pretest_a()
    table_pretest_b()
    table_pretest_c()
    table_pretest_d()
    table_best_per_family()
    table_per_metabolite()
    table_wilcoxon()
    table_augmentation()
    table_lc()
    table_llm_spec_ablation()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s")
    make_all()
