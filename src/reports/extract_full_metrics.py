"""Re-evaluate best specs for each T=0.4 run and extract the full per-metabolite metric set
(Pearson_log, MAPE_median, MAPE_max). Saves T04_full_metrics.json for downstream figures/tables."""
import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, "./autoresearch_lit")
from eval_cached import evaluate_spec

LOGS = Path("./results/autoresearch_lit/logs")

# T=0.4 runs and their target rho (best already known).
RUNS_T04 = {
    "v2-r2":   ("agent_run2.jsonl",   0.164),
    "v2-r5":   ("agent_run5.jsonl",   0.186),
    "v2-r6":   ("agent_run6.jsonl",   0.147),
    "rep-100": ("agent_run100.jsonl", 0.139),
    "rep-101": ("agent_run101.jsonl", 0.189),
    "rep-102": ("agent_run102.jsonl", 0.177),
    "rep-103": ("agent_run103.jsonl", 0.046),
    "rep-104": ("agent_run104.jsonl", 0.169),
}


def find_best_spec(jsonl_path, target_rho, tol=0.001):
    """Search JSONL for the evaluate_spec call with rho closest to target."""
    best_spec, best_model, best_diff = None, "ridge_bins", 1e9
    for line in jsonl_path.open():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("role") != "tool":
            continue
        if r.get("tool") != "evaluate_spec":
            continue
        rs = r.get("result_summary", {})
        if not isinstance(rs, dict):
            continue
        rho = rs.get("spearman_mean_valid")
        if rho is None:
            continue
        diff = abs(rho - target_rho)
        if diff < best_diff:
            best_diff = diff
            best_spec = r.get("args", {}).get("spec")
            best_model = r.get("args", {}).get("model", "ridge_bins")
    return best_spec, best_model


def aggregate_metrics(per_met, n_min=20):
    """Aggregate per-metabolite metrics (filtering n < n_min)."""
    rhos, plogs, mapes = [], [], []
    for met, m in per_met.items():
        rho = m.get("spearman")
        plog = m.get("pearson_log")
        mape = m.get("mape_pct")
        n = m.get("n", 0)
        if rho is None or n < n_min:
            continue
        if isinstance(rho, float) and np.isnan(rho):
            continue
        rhos.append(rho)
        if plog is not None and not (isinstance(plog, float) and np.isnan(plog)):
            plogs.append(plog)
        if mape is not None and not (isinstance(mape, float) and np.isnan(mape)):
            mapes.append(mape)
    return {
        "n_valid": len(rhos),
        "spearman": float(np.mean(rhos)) if rhos else None,
        "pearson_log": float(np.mean(plogs)) if plogs else None,
        "mape_mean": float(np.mean(mapes)) if mapes else None,
        "mape_median": float(np.median(mapes)) if mapes else None,
        "mape_min": float(min(mapes)) if mapes else None,
        "mape_max": float(max(mapes)) if mapes else None,
    }


def main():
    print("Re-evaluating best specs (may take 5-10 min with cache)...")
    records = []
    for name, (logname, target_rho) in RUNS_T04.items():
        jsonl = LOGS / logname
        if not jsonl.exists():
            print(f"  {name}: NO log"); continue

        best_spec, best_model = find_best_spec(jsonl, target_rho)
        if not best_spec:
            print(f"  {name}: spec NOT found"); continue

        print(f"  {name}: re-evaluating (model={best_model})...", flush=True)
        try:
            res = evaluate_spec(best_spec, best_model)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

        per_met = res.get("per_metabolite", {})
        agg = aggregate_metrics(per_met)
        agg["name"] = name
        agg["model"] = best_model
        records.append(agg)

    print()
    print(f"{'Run':<10} {'n':<4} {'Spearman':<10} {'Pearson_log':<13} "
          f"{'MAPE_med%':<10} {'MAPE_max%'}")
    for r in records:
        sp = f"{r['spearman']:>+.3f}" if r['spearman'] else "  ---"
        pl = f"{r['pearson_log']:>+.3f}" if r['pearson_log'] else "  ---"
        mm = f"{r['mape_median']:>7.1f}" if r['mape_median'] else "  ---"
        mx = f"{r['mape_max']:>9.1e}" if r['mape_max'] else "  ---"
        print(f"{r['name']:<10} {r['n_valid']:<4} {sp:<10}  {pl:<13} "
              f"{mm:<10} {mx}")

    out = Path("./results/autoresearch_lit/T04_full_metrics.json")
    out.write_text(json.dumps(records, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
