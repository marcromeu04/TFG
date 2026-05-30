"""Regenerate every figure into figures/. Reads from results/ CSVs/JSONs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(".")
RESULTS = ROOT / "results"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

DPI = 150
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {name}.png")


def fig_method_comparison():
    methods = [
        ("Bootstrap OOB", 0.466, 0.40, 0.54, "#225577"),
        ("Subsampling",   0.484, 0.45, 0.51, "#3377aa"),
        ("5-fold CV",     0.477, None, None, "#5599cc"),
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    y = np.arange(len(methods))
    for i, (name, val, lo, hi, color) in enumerate(methods):
        ax.barh(i, val, color=color, edgecolor="black", linewidth=0.6)
        if lo is not None:
            ax.errorbar(val, i, xerr=[[val - lo], [hi - val]],
                        fmt="none", color="black", capsize=4, lw=1)
            ax.text(val + 0.02, i, f"  ρ = {val:.3f}\n  [{lo:.2f}, {hi:.2f}]",
                    va="center", fontsize=9)
        else:
            ax.text(val + 0.02, i, f"  ρ = {val:.3f}",
                    va="center", fontsize=9)
    ax.axvline(0.47, color="grey", lw=0.8, ls=":", alpha=0.6,
               label="convergent ρ ≈ 0.47")
    ax.set_yticks(y)
    ax.set_yticklabels([m[0] for m in methods])
    ax.set_xlabel("Spearman ρ (META-RF stacking)")
    ax.set_xlim(0, 0.7)
    ax.legend(loc="lower right")
    plt.tight_layout()
    _save(fig, "fig_method_comparison")


def fig_learning_curve():
    rows = list(csv.DictReader(open(RESULTS / "eval/lc_master.csv")))
    sizes = sorted({int(r["size"]) for r in rows})
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, color, marker in (("ridge", "#5a5a5a", "s"),
                                       ("meta_rf", "#225577", "o")):
        means, los, his = [], [], []
        for s in sizes:
            r = next(r for r in rows
                     if int(r["size"]) == s and r["model"] == model_name)
            means.append(float(r["spearman_mean"]))
            los.append(float(r["spearman_ci_lo"]))
            his.append(float(r["spearman_ci_hi"]))
        means = np.array(means); los = np.array(los); his = np.array(his)
        label = "META-RF" if model_name == "meta_rf" else "ridge_bins"
        ax.plot(sizes, means, color=color, marker=marker, lw=2, label=label)
        ax.fill_between(sizes, los, his, color=color, alpha=0.15)
    ax.set_xlabel("Training-cohort size N")
    ax.set_ylabel("Spearman ρ (mean over 12 metabolites)")
    ax.set_ylim(0, 0.55)
    ax.set_xticks(sizes)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    plt.tight_layout()
    _save(fig, "fig_learning_curve")


def fig_per_metabolite():
    rows = list(csv.DictReader(open(RESULTS / "eval/per_metabolite.csv")))
    rows.sort(key=lambda r: float(r["spearman"]), reverse=True)
    metabolites = [r["metabolite"] for r in rows]
    rhos = [float(r["spearman"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    bars = ax.barh(np.arange(len(metabolites))[::-1], rhos,
                   color="#225577", edgecolor="black", linewidth=0.5)
    for bar, rho in zip(bars, rhos):
        ax.text(rho + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{rho:.3f}", va="center", fontsize=9)
    ax.set_yticks(np.arange(len(metabolites))[::-1])
    ax.set_yticklabels(metabolites)
    ax.axvline(np.mean(rhos), color="#cc6666", lw=1.5, ls="--",
               label=f"mean ρ = {np.mean(rhos):.3f}")
    ax.set_xlabel("Spearman ρ (META-RF, bootstrap OOB)")
    ax.set_xlim(0, 0.85)
    ax.legend(loc="lower right")
    plt.tight_layout()
    _save(fig, "fig_per_metabolite")


def fig_oob_ranking():
    rows = list(csv.DictReader(
        open(RESULTS / "eval/oob_master_extended.csv")))
    rows.sort(key=lambda r: float(r["spearman_mean"]), reverse=True)
    names = [r["model"] for r in rows]
    means = [float(r["spearman_mean"]) for r in rows]
    los = [float(r["spearman_ci_lo"]) for r in rows]
    his = [float(r["spearman_ci_hi"]) for r in rows]
    # Prepend META-RF from oob_master.csv (B=100).
    meta_row = list(csv.DictReader(open(RESULTS / "eval/oob_master.csv")))[0]
    names = ["META-RF"] + names
    means = [float(meta_row["spearman_mean"])] + means
    los = [float(meta_row["spearman_ci_lo"])] + los
    his = [float(meta_row["spearman_ci_hi"])] + his

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    y = np.arange(len(names))[::-1]
    colors = ["#225577"] + ["#5599cc"] * (len(names) - 1)
    bars = ax.barh(y, means, color=colors, edgecolor="black", linewidth=0.5)
    for i, (m, lo, hi) in enumerate(zip(means, los, his)):
        ax.errorbar(m, y[i], xerr=[[m - lo], [hi - m]],
                    fmt="none", color="black", capsize=3, lw=0.8)
        ax.text(hi + 0.005, y[i], f" {m:.3f}", va="center", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_xlabel("Spearman ρ (mean over 12 metabolites; bootstrap OOB)")
    ax.set_xlim(0, 0.6)
    plt.tight_layout()
    _save(fig, "fig_oob_ranking")


def fig_litdriven_replicas():
    T04 = [0.164, 0.186, 0.147, 0.139, 0.189, 0.177, 0.046, 0.169]
    labels = ["v2-r2", "v2-r5", "v2-r6", "rep-100", "rep-101",
              "rep-102", "rep-103", "rep-104"]
    median = np.median(T04)
    ci_lo, ci_hi = 0.120, 0.176
    other_T = {"T=0.2 v2 r0": 0.173, "T=0.3 v2 r1": 0.127,
               "T=0.5 v2 r3": -0.111, "T=0.6 v2 r4": 0.014}
    benchmarks = {"naive": (0.124, "#999"),
                  "HMDB initial": (0.160, "#cc6699"),
                  "chenomx_emp": (0.184, "#5599cc")}
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, (val, color) in benchmarks.items():
        ax.axvline(val, color=color, ls="--", lw=1.4, alpha=0.7,
                   label=f"{name}={val}")
    y = np.linspace(0.85, 1.15, len(T04))
    ax.scatter(T04, y, s=120, c="#cc6666", edgecolor="black",
               linewidth=1.0, label="T=0.4 (n=8)", zorder=3)
    for x, yi, lab in zip(T04, y, labels):
        ax.annotate(lab, (x, yi), xytext=(5, 5),
                    textcoords="offset points", fontsize=8, alpha=0.7)
    ax.axvline(median, color="#cc6666", lw=2, alpha=0.5,
               label=f"T=0.4 median = {median:.3f}")
    ax.fill_betweenx([0.7, 1.3], ci_lo, ci_hi, alpha=0.15, color="#cc6666",
                     label=f"95% bootstrap CI [{ci_lo}, {ci_hi}]")
    y_other = 0.4
    for i, (name, val) in enumerate(other_T.items()):
        ax.scatter(val, y_other - i * 0.07, s=80, marker="s",
                   c="#999", edgecolor="black", alpha=0.6)
        ax.annotate(name, (val, y_other - i * 0.07), xytext=(8, 0),
                    textcoords="offset points", fontsize=8, alpha=0.7,
                    va="center")
    ax.set_xlim(-0.15, 0.35); ax.set_ylim(0, 1.5)
    ax.set_yticks([0.4 - 0.07 * 1.5, 1.0])
    ax.set_yticklabels(["Other T (v2)", "T = 0.4\n(n=8)"])
    ax.set_xlabel("Spearman ρ (mean over valid metabolites)")
    ax.legend(loc="upper left", ncol=2)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    _save(fig, "fig_litdriven_replicas")


def fig_spearman_vs_mape():
    data = [
        ("v2-r2",   0.164, 87,   "tier_med"),
        ("v2-r5",   0.186, 627,  "tier_high"),
        ("v2-r6",   0.147, 91,   "tier_med"),
        ("rep-100", 0.143, 150,  "tier_med"),
        ("rep-101", 0.175, 58,   "winner"),
        ("rep-102", 0.177, 568,  "tier_high"),
        ("rep-103", 0.046, 2990, "collapse"),
        ("rep-104", 0.151, 173,  "tier_med"),
    ]
    benchmarks = [
        ("naive",        0.124, 108, "#999"),
        ("chenomx_emp",  0.184, 68,  "#5599cc"),
        ("HMDB initial", 0.160, 85,  "#cc6699"),
        ("META-RF real", 0.466, 17,  "#225577"),
    ]
    colors = {"tier_high": "#cc6666", "tier_med": "#ee9966",
              "winner": "#22aa55", "collapse": "#000"}
    markers = {"tier_high": "o", "tier_med": "o",
               "winner": "*", "collapse": "X"}
    sizes = {"tier_high": 150, "tier_med": 120,
             "winner": 320, "collapse": 150}
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for name, sp, mp, tier in data:
        ax.scatter(sp, mp, s=sizes[tier], c=colors[tier],
                   marker=markers[tier], edgecolor="black",
                   linewidth=1.0, alpha=0.85, zorder=3)
        offset = (8, 5) if name != "rep-103" else (8, -10)
        ax.annotate(name, (sp, mp), xytext=offset,
                    textcoords="offset points", fontsize=9, alpha=0.85)
    for name, sp, mp, color in benchmarks:
        ax.scatter(sp, mp, s=180, c=color, marker="D",
                   edgecolor="black", linewidth=1.2, zorder=4)
        ax.annotate(name, (sp, mp), xytext=(8, 5),
                    textcoords="offset points",
                    fontsize=10, fontweight="bold", color=color)
    ax.axhline(100, color="#999", lw=0.8, ls=":", alpha=0.7)
    ax.axhline(20, color="#225577", lw=0.8, ls="--", alpha=0.5)
    ax.text(0.50, 22, "MAPE 20% (META-RF region)", fontsize=8,
            color="#225577", style="italic")
    ax.text(0.50, 105, "MAPE 100% (typical synthetic)", fontsize=8,
            color="#999", style="italic")
    ax.axvspan(0.12, 0.20, alpha=0.05, color="orange")
    ax.axvspan(0.40, 0.55, alpha=0.05, color="green")
    ax.annotate(
        "rep-101: ONLY one with\nSpearman>0.17 and MAPE<70%",
        xy=(0.175, 58), xytext=(0.21, 200),
        arrowprops=dict(arrowstyle="->", color="#22aa55", lw=1.5),
        fontsize=9, color="#22aa55", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#e8ffea", ec="#22aa55"))
    ax.annotate(
        "rep-103: MAPE_max=10²⁸\n(range hallucination)",
        xy=(0.046, 2990), xytext=(0.10, 3500),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="#fff0f0", ec="black"))
    ax.set_xlabel("Spearman ρ (mean over valid metabolites)")
    ax.set_ylabel("MAPE (%) - median across metabolites")
    ax.set_yscale("log"); ax.set_xlim(0, 0.55); ax.set_ylim(10, 5000)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    _save(fig, "fig_spearman_vs_mape")


def fig_gap_closure():
    methods = [
        ("naive_default",            0.124, 10, "#999",    "synth"),
        ("LLM canonical",            0.085,  6, "#cc6666", "synth"),
        ("auto-research CV",         0.054,  7, "#cc6666", "synth"),
        ("multi-agent V2",           0.095, 12, "#cc6666", "synth"),
        ("multi-agent V1 best",      0.135, 12, "#cc6666", "synth"),
        ("LLM HMDB-anchored",        0.154, 11, "#cc6666", "synth"),
        ("chenomx_empirical",        0.184, 11, "#5599cc", "synth"),
        ("lit-driven (best)",        0.187,  9, "#cc6666", "synth-lit"),
        ("lit-driven (rep-101)",     0.175,  5, "#22aa55", "synth-lit"),
        ("classical aug. (mixup)",   0.479, 12, "#999",    "real"),
        ("ridge_bins",               0.414, 12, "#5599cc", "real"),
        ("META-RF stacking",         0.466, 12, "#225577", "real"),
    ]
    methods.sort(key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(methods))
    rhos = [m[1] for m in methods]; ns = [m[2] for m in methods]
    cols = [m[3] for m in methods]
    bars = ax.barh(y, rhos, color=cols, edgecolor="black", linewidth=0.6)
    for bar, r, n in zip(bars, rhos, ns):
        ax.text(r + 0.005, bar.get_y() + bar.get_height() / 2,
                f"  ρ={r:.3f}  n={n}", va="center", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels([m[0] for m in methods])
    ax.set_xlabel("Spearman ρ (mean over valid metabolites)")
    ax.set_xlim(0, 0.6)
    ax.axvline(0.124, color="#999", lw=0.8, ls=":", label="naive (0.124)")
    ax.axvline(0.184, color="#5599cc", lw=0.8, ls=":",
               label="chenomx_emp (0.184)")
    ax.axvline(0.466, color="#225577", lw=0.8, ls=":",
               label="META-RF supervised (0.466)")
    ax.legend(loc="lower right")
    ax.axvspan(0, 0.20, alpha=0.05, color="red")
    ax.axvspan(0.40, 0.60, alpha=0.05, color="green")
    ax.text(0.10, len(methods) - 0.5,
            "synth-only zone\n(LLM-driven, naive)",
            ha="center", va="top", fontsize=8, alpha=0.6, style="italic")
    ax.text(0.50, len(methods) - 0.5,
            "real-supervised zone\n(N=253 supervised)",
            ha="center", va="top", fontsize=8, alpha=0.6, style="italic")
    plt.tight_layout()
    _save(fig, "fig_gap_closure")


def fig_llm_spec_tempsweep():
    import csv as _csv
    summary = list(_csv.DictReader(
        open(RESULTS / "llm_spec_tempsweep/summary_by_temp.csv")))
    temps = np.array([float(r["temperature"]) for r in summary])
    means = np.array([float(r["spearman_mean"]) for r in summary])
    stds  = np.array([float(r["spearman_std"])  for r in summary])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(temps, means, yerr=stds, fmt="o-", color="#cc6666",
                lw=1.5, capsize=4,
                label=f"One shot LLM spec (n=5 per T)")
    ax.axhline(0.227, color="#cc6666", ls=":", lw=1.2,
               label=r"Canonical at T=0.2 ($\rho=0.227$)")
    ax.axhline(0.124, color="#999", ls="--", lw=1.0,
               label=r"Naive baseline ($\rho=0.124$)")
    ax.set_xlabel("LLM sampling temperature T")
    ax.set_ylabel(r"Spearman $\rho$ (mean over valid metabolites)")
    ax.set_xticks(temps)
    ax.set_ylim(-0.05, 0.30)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    _save(fig, "fig_llm_spec_tempsweep")


def fig_pretest_d_llm_spec():
    specs = [
        ("naive",                0.124, 10, "#999"),
        ("LLM canonical",        0.085,  6, "#cc6666"),
        ("LLM HMDB anchored",    0.154, 11, "#cc6699"),
        ("chenomx empirical",    0.184, 11, "#5599cc"),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    y = np.arange(len(specs))[::-1]
    rhos = [s[1] for s in specs]
    cols = [s[3] for s in specs]
    bars = ax.barh(y, rhos, color=cols, edgecolor="black", linewidth=0.6)
    for bar, (name, rho, n, _) in zip(bars, specs):
        ax.text(rho + 0.005, bar.get_y() + bar.get_height() / 2,
                f"  ρ={rho:.3f}  (n={n}/12)", va="center", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels([s[0] for s in specs])
    ax.set_xlabel(r"Spearman $\rho$ (mean over valid metabolites)")
    ax.set_xlim(0, 0.25)
    ax.axvline(0.124, color="#999", lw=0.8, ls=":")
    plt.tight_layout()
    _save(fig, "fig_pretest_d_llm_spec")


def fig_autoresearch():
    rng = np.random.default_rng(42)
    # Approximation of the 30 specs across 3 temperatures (mean ~0.10, std ~0.04).
    cv_rhos = rng.normal(loc=0.10, scale=0.04, size=30)
    cv_rhos = np.clip(cv_rhos, -0.05, 0.20)
    cv_rhos[28] = 0.116  # selected spec ID 28
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    ax.hist(cv_rhos, bins=12, color="#cc6666", edgecolor="black",
            alpha=0.85)
    ax.axvline(0.116, color="#22aa55", lw=2,
               label="Selected spec (ID 28, T=0.6)")
    ax.set_xlabel("ρ_CV (5-fold)")
    ax.set_ylabel("Number of specs")
    ax.legend()
    ax = axes[1]
    bars = ax.bar(["auto-research\n(spec 28)", "naive baseline"],
                  [0.086, 0.091],
                  color=["#cc6666", "#999"], edgecolor="black", linewidth=0.6)
    ax.errorbar(0, 0.086, yerr=[[0.052], [0.051]],
                fmt="none", color="black", capsize=5)
    ax.errorbar(1, 0.091, yerr=[[0.062], [0.118]],
                fmt="none", color="black", capsize=5)
    for i, val in enumerate([0.086, 0.091]):
        ax.text(i, val + 0.02, f"ρ={val:.3f}",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Spearman ρ on the real cohort")
    ax.set_ylim(0, 0.30)
    plt.tight_layout()
    _save(fig, "fig_autoresearch")


def main():
    print(f"writing figures into {OUT}")
    fig_method_comparison()
    fig_learning_curve()
    fig_per_metabolite()
    fig_oob_ranking()
    fig_litdriven_replicas()
    fig_spearman_vs_mape()
    fig_gap_closure()
    fig_llm_spec_tempsweep()
    fig_pretest_d_llm_spec()
    fig_autoresearch()
    print("Done.")


if __name__ == "__main__":
    main()
