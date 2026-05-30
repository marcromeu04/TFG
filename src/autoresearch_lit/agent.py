"""Literature-driven auto-research agent.
Closed loop: reflect -> PubMed search -> read abstracts -> propose spec (with PMID citations)
-> evaluate -> log -> iterate. Constraints in the system prompt require PMID citations and
counter-evidence searches. Outputs per-run JSONL audit log and results JSON."""
from __future__ import annotations
import os, sys, json, time, logging, argparse
from pathlib import Path
from datetime import datetime
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, './autoresearch_lit')

from pubmed_tools import pubmed_search, fetch_abstract
from eval_cached import evaluate_spec, ALLOWED_MODELS

from groq import Groq

def setup_logging(run_id: int, log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = '%(asctime)s [%(levelname)s] [run%(run)d] %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(log_dir / f"agent_run{run_id}.log"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return logging.LoggerAdapter(logging.getLogger("agent"), {"run": run_id})

# HMDB-anchored initial concentrations (mM mean, mM std).
HMDB_MM = {
    "Lactate":(1.5, 0.6), "Histidine":(0.075,0.020), "Cysteine":(0.060,0.020),
    "Glucose":(5.0, 1.0), "Glycine":(0.250,0.060), "Betaine":(0.040,0.015),
    "Pyruvate":(0.080,0.030), "Threonine":(0.140,0.040), "Serine":(0.120,0.030),
    "Choline":(0.010,0.005), "Creatine":(0.045,0.015), "Creatinine":(0.080,0.020),
}
def initial_spec():
    return {
        "ranges": {k: [float(np.log10(m)),
                        max(float(s/(m*np.log(10))), 0.05)]
                    for k, (m, s) in HMDB_MM.items()},
        "correlations": {},
        "baseline_kind": "polynomial",
        "noise_shift": {"noise_amp": 0.005, "shift_amp": 0.001},
    }

SYSTEM_PROMPT = """You are an automated research scientist working on
quantitative 1H-NMR metabolomics in human serum.

CONTEXT
=======
You have access to:
  - The cohort (N=253 sera, 12 metabolites quantified by Chenomx).
  - A synthetic data generator that takes a spec (ranges, correlations,
    baseline, noise) and produces a synthetic training set.
  - A predictive pipeline that trains one of 10 supervised models on
    synthetic data and evaluates Spearman per-metabolite on the real cohort.

YOUR GOAL
=========
Maximize the per-metabolite Spearman correlation on the real cohort.
Both the mean Spearman across valid metabolites AND the number of valid
metabolites (n_valid out of 12) matter.

YOUR METHOD
===========
You must operate as a literature-driven scientist:

1. EVERY change you propose MUST be justified by at least one citation
   (PMID) from PubMed. Do not propose changes from memory alone.

2. At iteration t > 0 you MUST start by REFLECTING on the previous
   iteration's outcome:
     - Did the result confirm the literature-derived hypothesis?
     - If not, why might it have failed?
   Then act on that reflection.

3. You MUST search for COUNTER-evidence at least once per run.
   E.g. "studies that found NO correlation between X and Y in serum".
   This avoids confirmation bias.

4. Diversity matters: avoid repeating the same query verbatim.

TOOLS
=====
You have three tools:
  pubmed_search(query, max_results=10): returns list of {pmid, title, year, journal}
  fetch_abstract(pmid): returns full abstract for one PMID
  evaluate_spec(spec, model): runs the synth->real pipeline, returns
    {spearman_mean_valid, n_metabolites_valid, per_metabolite, model_used}

The model argument can be one of:
  ridge_bins, pls_bins, lasso_bins, enet_bins,
  rf_roi, knn_roi, svr_roi, gpr_roi, xgb_roi,
  ensemble_top3.

The spec argument has keys:
  ranges: dict of metabolite -> [log10_mean, log10_std]
  correlations: dict of "metA__metB" -> correlation in [-1, 1]
  baseline_kind: "polynomial" | "spline" | "empirical_residual"
  noise_shift: {"noise_amp": float, "shift_amp": float}

OUTPUT FORMAT
=============
Always reason out loud, then call tools as needed. When you propose
a final spec for an iteration, call evaluate_spec. The system will
record everything.

Iterate up to 5 times. Try to converge on a stable, well-justified spec.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "pubmed_search",
            "description": "Search PubMed for relevant papers. Returns list of {pmid, title, year, journal}.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "PubMed query string"},
                    "max_results": {"type": "integer", "default": 10,
                                     "description": "Max number of results"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_abstract",
            "description": "Fetch the full abstract for a single PMID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pmid": {"type": "string", "description": "PubMed ID"},
                },
                "required": ["pmid"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_spec",
            "description": "Evaluate a generator spec on the cohort. "
                            "Returns per-metabolite Spearman + summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "spec": {
                        "type": "object",
                        "description": "Generator spec (ranges, correlations, baseline_kind, noise_shift)",
                    },
                    "model": {
                        "type": "string",
                        "description": "Predictive model to use",
                        "enum": ALLOWED_MODELS,
                    },
                },
                "required": ["spec", "model"],
            },
        },
    },
]

def dispatch_tool(name: str, args: dict, log) -> dict:
    """Execute a tool call and return the result as a dict."""
    try:
        if name == "pubmed_search":
            query = args.get("query", "")
            max_results = int(args.get("max_results", 10))
            log.info(f"TOOL pubmed_search: query={query!r}, max={max_results}")
            res = pubmed_search(query, max_results=max_results)
            return {"results": res, "n_results": len(res)}
        elif name == "fetch_abstract":
            pmid = str(args.get("pmid", ""))
            log.info(f"TOOL fetch_abstract: pmid={pmid}")
            res = fetch_abstract(pmid)
            if "abstract" in res and len(res["abstract"]) > 2000:
                res["abstract"] = res["abstract"][:2000] + "...[truncated]"
            return res
        elif name == "evaluate_spec":
            spec = args.get("spec", {})
            model = args.get("model", "ridge_bins")
            log.info(f"TOOL evaluate_spec: model={model}, "
                      f"ranges={list(spec.get('ranges',{}).keys())[:3]}...")
            t0 = time.time()
            res = evaluate_spec(spec, model=model)
            log.info(f"  -> mean={res.get('spearman_mean_valid')}, "
                      f"n_valid={res.get('n_metabolites_valid')}, "
                      f"cache={res.get('from_cache')} ({time.time()-t0:.1f}s)")
            return res
        else:
            return {"error": f"unknown tool: {name}"}
    except Exception as e:
        log.error(f"TOOL {name} failed: {e}")
        return {"error": str(e)}

def run_agent(run_id: int,
               max_iterations: int = 5,
               max_tool_calls_per_turn: int = 30,
               results_dir: Path = Path("./results/autoresearch_lit"),
               log_dir: Path = Path("./results/autoresearch_lit/logs"),
               temperature: float = 0.4,
               model: str = "llama-3.3-70b-versatile"):
    log = setup_logging(run_id, log_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    audit_path = log_dir / f"agent_run{run_id}.jsonl"

    log.info(f"Starting agent run {run_id}")
    log.info(f"Max iterations: {max_iterations}")
    log.info(f"LLM: {model}, temperature: {temperature}")

    init_spec = initial_spec()
    log.info("Evaluating initial HMDB-anchored spec...")
    init_metrics = evaluate_spec(init_spec, model="ridge_bins")
    log.info(f"INITIAL: mean={init_metrics['spearman_mean_valid']:.3f}, "
              f"n_valid={init_metrics['n_metabolites_valid']}")

    initial_summary = (f"Initial spec (HMDB-anchored): "
                        f"Spearman mean={init_metrics['spearman_mean_valid']:.3f}, "
                        f"n_valid={init_metrics['n_metabolites_valid']}/12. "
                        f"Per-metabolite: " +
                        ", ".join(f"{m}={v['spearman']:+.2f}" if v['spearman'] is not None else f"{m}=NaN"
                                    for m, v in init_metrics['per_metabolite'].items()))

    client = Groq()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"""You start with the HMDB-anchored
initial spec. Your initial evaluation:

{initial_summary}

Iterate up to {max_iterations} times. In each iteration:
  1. (If t > 0) Reflect on the previous outcome.
  2. Search PubMed for relevant literature (and at least once per run, counter-evidence).
  3. Fetch abstracts you find promising.
  4. Propose a refined spec, with PMIDs justifying each change.
  5. Call evaluate_spec to test it.

Begin iteration 1."""}
    ]

    trajectory = [{
        "iteration": 0,
        "kind": "initial",
        "spec": init_spec,
        "metrics": init_metrics,
        "timestamp": time.time(),
    }]
    audit_records = []
    pmids_cited_global = set()
    queries_global = []

    iteration = 0
    converged = False

    for iteration in range(1, max_iterations + 1):
        log.info(f"Iteration {iteration}")
        n_tool_calls = 0

        while True:
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=temperature,
                    max_tokens=4000,
                )
            except Exception as e:
                log.error(f"Groq call failed: {e}")
                time.sleep(5)
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        tools=TOOLS,
                        tool_choice="auto",
                        temperature=temperature,
                        max_tokens=4000,
                    )
                except Exception as e2:
                    log.error(f"Groq retry failed: {e2}")
                    break

            msg = resp.choices[0].message
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                      "function": {"name": tc.function.name,
                                    "arguments": tc.function.arguments}}
                    for tc in (msg.tool_calls or [])
                ] if msg.tool_calls else None,
            })

            audit_records.append({
                "iteration": iteration,
                "timestamp": time.time(),
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {"name": tc.function.name, "args": tc.function.arguments}
                    for tc in (msg.tool_calls or [])
                ],
            })

            if not msg.tool_calls:
                # Assistant ended turn without tool calls; force continuation if budget remains.
                log.info(f"Iter {iteration}: assistant ended turn without tools.")
                if iteration < max_iterations:
                    messages.append({
                        "role": "user",
                        "content": (
                            "Do not stop yet. You have more iterations available. "
                            "Continue: search PubMed for additional literature "
                            "(consider angles you have not explored yet, like noise/baseline, "
                            "metabolite-specific anomalies, or different correlation structures), "
                            "fetch new abstracts, propose a substantively different spec or "
                            "try a different model (e.g. ensemble_top3, rf_roi), "
                            "and call evaluate_spec. Do not respond with just text - "
                            "you MUST call tools."
                        )
                    })
                    audit_records.append({
                        "iteration": iteration, "timestamp": time.time(),
                        "role": "user_forced_continue",
                        "content": "Forced continuation",
                    })
                    continue
                else:
                    break

            for tc in msg.tool_calls:
                n_tool_calls += 1
                if n_tool_calls > max_tool_calls_per_turn:
                    log.warning(f"Iter {iteration}: tool call cap reached.")
                    break
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError as e:
                    log.warning(f"Bad JSON args for {tc.function.name}: {e}")
                    args = {}

                if tc.function.name == "pubmed_search":
                    queries_global.append(args.get("query", ""))
                if tc.function.name == "fetch_abstract":
                    pmid = str(args.get("pmid", ""))
                    if pmid: pmids_cited_global.add(pmid)

                result = dispatch_tool(tc.function.name, args, log)

                if tc.function.name == "evaluate_spec" and "error" not in result:
                    trajectory.append({
                        "iteration": iteration,
                        "kind": "evaluation",
                        "spec": args.get("spec", {}),
                        "model": args.get("model", "ridge_bins"),
                        "metrics": result,
                        "timestamp": time.time(),
                    })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, default=str)[:30000],
                })
                audit_records.append({
                    "iteration": iteration,
                    "timestamp": time.time(),
                    "role": "tool",
                    "tool": tc.function.name,
                    "args": args,
                    "result_summary": (
                        {k: v for k, v in result.items() if k != "per_metabolite"}
                        if isinstance(result, dict) else str(result)[:500]
                    ),
                })

            if n_tool_calls > max_tool_calls_per_turn:
                break

        evals_this_iter = [t for t in trajectory
                            if t["iteration"] == iteration and t["kind"] == "evaluation"]
        if not evals_this_iter:
            log.warning(f"Iter {iteration}: no evaluations performed.")
        else:
            best_this = max(evals_this_iter,
                             key=lambda t: t["metrics"].get("spearman_mean_valid", -1))
            log.info(f"Iter {iteration} best: "
                      f"mean={best_this['metrics']['spearman_mean_valid']:.3f}, "
                      f"n={best_this['metrics']['n_metabolites_valid']}, "
                      f"model={best_this.get('model')}")

        if iteration < max_iterations:
            messages.append({
                "role": "user",
                "content": (f"Now begin iteration {iteration + 1}. "
                            "Remember: reflect on the previous result, "
                            "search literature (with counter-evidence at some point), "
                            "and propose a justified change. "
                            "Cite PMIDs explicitly.")
            })

    with audit_path.open("w") as f:
        for r in audit_records:
            f.write(json.dumps(r, default=str) + "\n")
    log.info(f"Audit log saved to {audit_path}")

    evals = [t for t in trajectory if t["kind"] == "evaluation"]
    if evals:
        best = max(evals, key=lambda t: t["metrics"].get("spearman_mean_valid", -1))
    else:
        best = trajectory[0]

    summary = {
        "run_id": run_id,
        "model": model, "temperature": temperature,
        "n_iterations_completed": iteration,
        "n_evaluations": len(evals),
        "n_searches": len(queries_global),
        "n_unique_queries": len(set(queries_global)),
        "n_pmids_fetched": len(pmids_cited_global),
        "pmids_fetched": sorted(pmids_cited_global),
        "queries": queries_global,
        "initial": {
            "spearman_mean": init_metrics["spearman_mean_valid"],
            "n_valid": init_metrics["n_metabolites_valid"],
        },
        "best": {
            "iteration": best["iteration"],
            "spearman_mean": best["metrics"].get("spearman_mean_valid"),
            "n_valid": best["metrics"].get("n_metabolites_valid"),
            "model_used": best.get("model"),
            "spec": best.get("spec"),
        },
        "trajectory": [
            {"iter": t["iteration"], "kind": t["kind"],
              "model": t.get("model"),
              "spearman_mean": t["metrics"].get("spearman_mean_valid"),
              "n_valid": t["metrics"].get("n_metabolites_valid")}
            for t in trajectory
        ],
        "improvement_best": (
            best["metrics"].get("spearman_mean_valid", 0)
            - init_metrics["spearman_mean_valid"]
        ),
    }
    out_path = results_dir / f"agent_run{run_id}.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    log.info(f"Summary saved to {out_path}")
    log.info(f"BEST: mean={summary['best']['spearman_mean']}, "
              f"n_valid={summary['best']['n_valid']}, "
              f"PMIDs cited={summary['n_pmids_fetched']}, "
              f"unique queries={summary['n_unique_queries']}")
    log.info(f"Run {run_id} done")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, required=True)
    parser.add_argument("--max_iter", type=int, default=5)
    parser.add_argument("--temp", type=float, default=0.4)
    parser.add_argument("--model", type=str, default="llama-3.3-70b-versatile")
    args = parser.parse_args()
    run_agent(run_id=args.run_id,
               max_iterations=args.max_iter,
               temperature=args.temp,
               model=args.model)
