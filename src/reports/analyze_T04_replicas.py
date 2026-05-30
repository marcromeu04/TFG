"""Statistical analysis of the T=0.4 lit-driven replicas: descriptives, bootstrap CI,
sign/Wilcoxon tests vs naive/HMDB/chenomx baselines, and Holm-Bonferroni correction."""
import numpy as np
from scipy.stats import binomtest, wilcoxon


def analyze_replicas():
    T04_runs = {
        "v2-r2":   0.164,
        "v2-r5":   0.186,
        "v2-r6":   0.147,
        "rep-100": 0.139,
        "rep-101": 0.189,
        "rep-102": 0.177,
        "rep-103": 0.046,  # outlier (collapse)
        "rep-104": 0.169,
    }

    rhos = list(T04_runs.values())
    n = len(rhos)

    print(f"STATISTICAL ANALYSIS OF T=0.4 REPLICAS (n={n})")

    print("\nData:")
    for name, val in T04_runs.items():
        print(f"  {name:<10} ρ={val:+.3f}")

    print("\nDescriptive statistics:")
    print(f"  Mean:     {np.mean(rhos):+.4f}")
    print(f"  Median:   {np.median(rhos):+.4f}")
    print(f"  Std:      {np.std(rhos):+.4f}")
    print(f"  Min:      {min(rhos):+.4f}")
    print(f"  Max:      {max(rhos):+.4f}")
    print(f"  Range:    {max(rhos) - min(rhos):.4f}")
    print(f"  CV:       {np.std(rhos) / np.mean(rhos):.3f}")

    rng = np.random.default_rng(42)
    boot_means = [np.mean(rng.choice(rhos, len(rhos), replace=True))
                  for _ in range(2000)]
    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))
    print(f"  95% bootstrap CI (2000 resamples): "
          f"[{ci_lo:+.4f}, {ci_hi:+.4f}]")

    rhos_no = [r for r in rhos if r > 0.10]
    boot_no = [np.mean(rng.choice(rhos_no, len(rhos_no), replace=True))
               for _ in range(2000)]
    print(f"\n  Excluding outlier rep-103:")
    print(f"    Mean:    {np.mean(rhos_no):+.4f}")
    print(f"    95% CI:  [{np.percentile(boot_no, 2.5):+.4f}, "
          f"{np.percentile(boot_no, 97.5):+.4f}]")

    print("\nNon-parametric tests vs baselines (one-sided)")
    baselines = {
        "naive":               0.124,
        "HMDB initial":        0.160,
        "chenomx_empirical":   0.184,
    }
    print(f"{'Baseline':<22} {'n_above':<9} {'Sign p':<10} "
          f"{'Wilcoxon p':<12} {'Sig?'}")
    raw_pvals = []
    for name, val in baselines.items():
        n_above = sum(1 for r in rhos if r >= val)
        p_sign = binomtest(n_above, n=n, p=0.5,
                           alternative="greater").pvalue
        raw_pvals.append((name, p_sign))
        diffs = [r - val for r in rhos]
        try:
            stat, p_wil = wilcoxon(diffs, alternative="greater")
        except Exception:
            p_wil = float("nan")
        sig = "✓" if p_sign < 0.05 else ("~" if p_sign < 0.10 else "✗")
        print(f"{name:<22} {n_above}/{n}        "
              f"{p_sign:<10.4f} {p_wil:<12.4f} {sig}")

    print("\nHolm-Bonferroni correction over the 3 sign tests (alpha=0.05)")
    sorted_pvals = sorted(raw_pvals, key=lambda x: x[1])
    holm = []
    last = 0.0
    for i, (name, p) in enumerate(sorted_pvals):
        factor = len(sorted_pvals) - i
        adj = min(p * factor, 1.0)
        last = max(last, adj)
        holm.append((name, p, last))
    for name, p_raw, p_adj in holm:
        sig = "✓" if p_adj < 0.05 else "✗"
        print(f"  {name:<22} p_raw={p_raw:.4f}  p_Holm={p_adj:.4f}  {sig}")

    print("\nCONCLUSION")
    print("""
  Lit-driven T=0.4 (n=8 replicas):
    - vs naive:             p_raw=0.035 (significant);
                            p_Holm=0.105 (NS after correction).
    - vs HMDB initial:      5/8 exceed it, NS.
    - vs chenomx_empirical: 2/8 exceed it, NS.
    - Best case (rep-101, ρ=0.189) exceeds chenomx_emp pointwise.
""")


if __name__ == "__main__":
    analyze_replicas()
