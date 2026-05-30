# Architecture

Layered codebase. Higher layers may import from lower ones, not the
other way around.

```
config.py              constants + paths

data/                  load + preprocess + features
synth/                 generator + ranges + correlations + llm_spec_loader

models/                base models + ensembles + stacking
eval/                  CV, OOB, learning curve, augmentation, significance

multi_agent/           agents + prompts + orchestrator + privacy filter

pretests/              synthetic ablations
llm_spec_experiment/   LLM-as-spec experiment

reports/               figure + table scripts
pipeline.py            inference
run_full_eval.py       orchestrator
```

## Data flow

```
raw NMR spectra (ASICS-aligned)
        |
        v
   data/load -> preprocess -> features
        |
        +--> direct supervision:  models/  +  eval/
        |
        +--> synthetic supervision: synth/  ->  models/ on synth  ->  eval/ on real
                                                                                |
                                                                                v
                                                                          reports/
```

## Design notes

- `config.py` is the only place that knows paths and the global seed.
- The generator is built as independently togglable components
  (concentrations / correlations / shifts / baseline / noise), which is
  what makes the ablation pretests possible without code duplication.
- Evaluation routines handle per-target NaN; the ground truth has
  missing values for a few metabolites and we drop only the missing
  rows for that target.
- Bootstrap-OOB is the headline protocol — no train-test overlap, and
  averaging ~37 OOB predictions per sample is much stabler than 5-fold
  CV at small n.
- All LLM calls go through `multi_agent.privacy_filter.validate_prompt`
  before leaving the process. It blocks sample/patient IDs and anything
  that looks like a raw spectrum.
- `run_full_eval.py` writes a master CSV per step and re-uses existing
  ones with `--resume`. Long campaigns survive crashes.

## Synthetic generator: lineage

Additive linear model:

    X = sum_m c_m * T_m(shifted) + baseline + noise

with log-normal concentrations (optional inter-metabolite correlation),
ASICS templates, per-metabolite Gaussian peak shifts, and one of three
baselines (polynomial / empirical PCA / empirical resample). The
empirical baselines are sampled from a residual library built from real
spectra; the non-linear regime is out of scope here.

What is new here:

- modular ablation interface;
- LLM spec loader (`synth/llm_spec_loader.py`) — a one-shot config
  synthesiser, not a tuning loop;
- privacy filter on every LLM call that touches the generator.

`pipeline.py` does NOT call the generator at inference. It serves the
META-RF stack trained on real Chenomx-supervised data.

## Compute

Reference numbers on 48 cores / 1 TB RAM, `n_jobs = 36`:

- pretests A–D: ~25 min
- 5-fold CV across 9 models: ~5 min
- bootstrap-OOB (B=100): ~60 min
- subsampling LC (5 sizes x 20): ~30 min
- augmentation comparison: ~15 min
- nested CV with Optuna: ~5 h (slowest)
- multi-agent v1 + v2 (x3 runs): ~30 min
- LLM-spec experiment: ~15 min
- figures + tables: ~5 min

~7–8 h total.
