"""Run 7 parallel literature-driven agent runs and aggregate."""
import subprocess, sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

ROOT = "."
N_RUNS = 7
TEMPS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.4, 0.4]  # vary slightly per run
MAX_ITER = 5

def launch_one(run_id, temp):
    cmd = [
        sys.executable,
        f"{ROOT}/autoresearch_lit/agent.py",
        "--run_id", str(run_id),
        "--max_iter", str(MAX_ITER),
        "--temp", str(temp),
    ]
    print(f"[{time.strftime('%H:%M:%S')}] Launching run {run_id} (T={temp})")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)
    return run_id, proc.returncode, proc.stdout[-2000:], proc.stderr[-2000:]

def main():
    t0 = time.time()
    print(f"Starting {N_RUNS} parallel runs at {time.strftime('%H:%M:%S')}")
    with ProcessPoolExecutor(max_workers=N_RUNS) as pool:
        futures = {pool.submit(launch_one, i, TEMPS[i]): i
                    for i in range(N_RUNS)}
        for fut in as_completed(futures):
            run_id, rc, out, err = fut.result()
            print(f"[{time.strftime('%H:%M:%S')}] Run {run_id} done "
                  f"(rc={rc}, elapsed {time.time()-t0:.0f}s)")
            if rc != 0:
                print(f"  STDERR (last 500): {err[-500:]}")

    elapsed = time.time() - t0
    print(f"\nAll runs done in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Aggregate
    print("\n=== Aggregating ===")
    results_dir = Path("./results/autoresearch_lit")
    summaries = []
    for i in range(N_RUNS):
        p = results_dir / f"agent_run{i}.json"
        if not p.exists():
            continue
        summaries.append(json.loads(p.read_text()))

    if summaries:
        all_path = results_dir / "all_runs_summary.json"
        all_path.write_text(json.dumps(summaries, indent=2, default=str))
        print(f"  Saved {all_path}")

        # Print compact table
        print(f"\n{'run':<4} {'init':<8} {'best':<8} {'n_valid':<8} {'iters':<6} {'queries':<8} {'pmids':<6}")
        print("-" * 60)
        for s in summaries:
            print(f"{s['run_id']:<4} "
                  f"{s['initial']['spearman_mean']:<8.3f} "
                  f"{s['best']['spearman_mean'] or 0:<8.3f} "
                  f"{s['best']['n_valid'] or 0:<8} "
                  f"{s['n_iterations_completed']:<6} "
                  f"{s['n_unique_queries']:<8} "
                  f"{s['n_pmids_fetched']:<6}")

if __name__ == "__main__":
    main()
