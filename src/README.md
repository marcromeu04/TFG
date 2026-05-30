# nmr-quant

Predicts concentrations of 12 serum metabolites from 1H-NMR spectra.
Two training paths:

- direct supervision on Chenomx ground truth (9 base models + stacking)
- synthetic supervision (linear additive generator + ASICS templates)

Plus an LLM-driven pipeline that proposes generator specs and a small
multi-agent diagnostic loop. The synthetic path is experimental and
the deployable inference (`pipeline.py`) uses the direct-supervision
META-RF stack.

## Layout

```
config.py        constants + paths
data/            spectrum loading + preprocessing + features
synth/           generator, ranges, correlations, llm_spec_loader
models/          base models, ensembles, meta-learners, augmentation
eval/            CV, bootstrap-OOB, learning curve, significance
multi_agent/     agents + orchestrator + privacy filter
pretests/        synthetic ablations
llm_spec_experiment/
reports/         figure + table scripts
tests/
pipeline.py
run_full_eval.py
```

See `ARCHITECTURE.md` for the data flow.

## Run

```bash
pip install -r requirements.txt
export NMR_DATA_ROOT=/path/to/data
export GROQ_API_KEY=...           # only for LLM paths
export OPENROUTER_API_KEY=...     # fallback only

pytest tests/ -v
python run_full_eval.py           # full campaign (~7–8 h on 48 cores)
python run_full_eval.py --quick   # smoke
```

Random seed in `config.SEED`. Outputs land in `$NMR_DATA_ROOT/results/`.

## Notes

Code released for academic use.
