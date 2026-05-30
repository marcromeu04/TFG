"""Single source of truth for all configuration. Modify paths here to relocate data/results."""
from __future__ import annotations

import os
from pathlib import Path

SEED = 42

N_FOLDS = 5                # outer folds for CV
N_REPEATS = 3              # repetitions of the outer CV (5x3 = 15 outer fits)
N_FOLDS_INNER = 3          # inner folds for nested CV (hyperparameter selection)
N_BOOTSTRAPS_OOB = 100     # bootstrap-OOB B for main evaluation
N_BOOTSTRAPS_LC = 20       # subsampling repetitions per learning-curve point
LC_SIZES = (50, 100, 150, 200, 253)   # cohort sizes evaluated on the curve

# Bootstrap-OOB CIs are 2.5/97.5 percentiles of B replicates.
BOOT_CI_LO_PCT = 2.5
BOOT_CI_HI_PCT = 97.5

# Optuna trials per inner fold. Slow models (GPR, XGBoost on small data, SVR-RBF)
# get fewer trials to keep nested CV inside compute budget.
N_OPTUNA_TRIALS_FAST = 30
N_OPTUNA_TRIALS_SLOW = 15

TARGET_AUG_SIZE = 2500     # target size after augmentation (10x cohort)
JITTER_SCALE = 0.01        # 1% of feature std
MIXUP_ALPHA = 0.4          # Beta(alpha, alpha) parameter

# 36 of the 48 cores; rest reserved for system + memory headroom.
N_JOBS_OUTER = 36          # parallel workers for top-level fan-out
N_JOBS_INNER = 1           # workers within a single fit (avoid oversubscription)

ROOT = Path(os.environ.get("NMR_DATA_ROOT", str(Path.cwd()))).resolve()

# Inputs (read-only)
INPUT_SPECTRA_DIR = ROOT / "results" / "R_outputs"          # ASICS-aligned
INPUT_CHENOMX_XLSX = Path.home() / "concentrations_clean.xlsx"
INPUT_TEMPLATES_DIR = ROOT / "data" / "asics_templates"      # 193 templates

# Outputs
RESULTS_ROOT = ROOT / "results"
RESULTS_EVAL = RESULTS_ROOT / "eval"
RESULTS_PRETESTS = RESULTS_ROOT / "pretests"
RESULTS_LLM_SPEC = RESULTS_ROOT / "llm_spec"
RESULTS_MULTIAGENT = RESULTS_ROOT / "multi_agent"
RESULTS_REPORTS = RESULTS_ROOT / "reports"
LOGS = RESULTS_ROOT / "logs"

for _p in (RESULTS_EVAL, RESULTS_PRETESTS, RESULTS_LLM_SPEC,
           RESULTS_MULTIAGENT, RESULTS_REPORTS, LOGS):
    _p.mkdir(parents=True, exist_ok=True)

PPM_MIN, PPM_MAX = 0.0, 10.0
WATER_MIN, WATER_MAX = 4.65, 4.95   # excluded
N_BINS_DEFAULT = 300                 # for the bins300 representation

# OCM panel: the 12 metabolites with Chenomx ground truth.
OCM_METABOLITES = (
    "Lactate",
    "Histidine",
    "Cysteine",
    "Glucose",
    "Glycine",
    "Betaine",
    "Pyruvate",
    "Threonine",
    "Serine",
    "Choline",
    "Creatine",
    "Creatinine",
)

# Canonical mapping Chenomx column name -> ASICS template name.
CHENOMX_TO_ASICS = {
    "Lactate":     "Lactate",
    "Histidine":   "L-Histidine",
    "Cysteine":    "L-Cysteine",
    "Glucose":     "D-Glucose",
    "Glycine":     "L-Glycine",
    "Betaine":     "Betaine",
    "Pyruvate":    "Pyruvic-Acid",
    "Threonine":   "L-Threonine",
    "Serine":      "L-Serine",
    "Choline":     "CholineChloride",
    "Creatine":    "Creatine",
    "Creatinine":  "Creatinine",
}

# Provider keys are read from environment, never hardcoded.
GROQ_API_KEY_ENV = "GROQ_API_KEY"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"

GROQ_MODEL_LARGE = "llama-3.3-70b-versatile"
GROQ_MODEL_SMALL = "llama-3.1-8b-instant"

# OpenRouter route used only for critical validation steps (budget is small).
OPENROUTER_MODEL_REASONING = "anthropic/claude-3.5-sonnet"

# Tokens that should never appear in a prompt sent to LLMs.
PRIVACY_BLACKLIST_PATTERNS = (
    r"sample[_\s]\d+",
    r"patient[_\s]\d+",
    r"\bsid\b",
    r"PID[:\s]\d+",
)


def assert_data_present():
    """Raise if essential input files are missing."""
    missing = []
    if not INPUT_SPECTRA_DIR.exists():
        missing.append(str(INPUT_SPECTRA_DIR))
    if not INPUT_CHENOMX_XLSX.exists():
        missing.append(str(INPUT_CHENOMX_XLSX))
    if missing:
        raise FileNotFoundError(
            "Missing required input data:\n  - " + "\n  - ".join(missing) +
            "\nReview config.py and ensure paths are correct."
        )
