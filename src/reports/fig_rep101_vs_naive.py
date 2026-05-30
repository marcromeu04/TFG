"""Per-metabolite Spearman bar chart: naive synth vs lit-driven rep-101 vs META-RF supervised.
Reads tfg_data.json::tabla_unified_per_metabolite; metabolites with any NaN are excluded."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(".")
DATA_FILE = ROOT / "tfg_data.json"
OUT_DIRS = [ROOT / "figures"]


def _to_float_or_nan(v):
    return np.nan if v == "NaN" else float(v)


def make_figure() -> None:
    with open(DATA_FILE) as f:
        d = json.load(f)
    t = d["tabla_unified_per_metabolite"]
    mets = t["metabolites"]

    naive = [_to_float_or_nan(v) for v in t["naive_default"]]
    rep101 = [_to_float_or_nan(v) for v in t["lit_driven_rep101"]]
    metarf = [_to_float_or_nan(v) for v in t["META_RF_real"]]

    keep = [i for i in range(len(mets))
            if not (np.isnan(naive[i]) or np.isnan(rep101[i])
                    or np.isnan(metarf[i]))]
    order = sorted(keep, key=lambda i: -metarf[i])
    mets_o = [mets[i] for i in order]
    naive_o = [naive[i] for i in order]
    rep101_o = [rep101[i] for i in order]
    metarf_o = [metarf[i] for i in order]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(mets_o))
    w = 0.27

    ax.bar(x - w, naive_o, width=w, color="#999999",
           edgecolor="black", linewidth=0.5, label="naive (a priori)")
    ax.bar(x, rep101_o, width=w, color="#22aa55",
           edgecolor="black", linewidth=0.5, label="lit-driven rep-101")
    ax.bar(x + w, metarf_o, width=w, color="#225577",
           edgecolor="black", linewidth=0.5, label="META-RF (real)")

    ax.axhline(0.184, color="#5599cc", lw=0.9, ls=":",
               label=r"chenomx empirical ($\rho=0.184$)")
    ax.axhline(0.124, color="#999", lw=0.9, ls=":",
               label=r"naive baseline mean ($\rho=0.124$)")
    ax.axhline(0, color="black", lw=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(mets_o, rotation=30, ha="right")
    ax.set_ylabel(
        r"Spearman $\rho$ (uniform protocol, $n_\text{synth}=5000$, ridge_bins)"
    )
    ax.set_ylim(-0.5, 0.8)
    ax.grid(axis="y", alpha=0.3, ls="--")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

    plt.tight_layout()
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        png_path = out_dir / "fig_rep101_vs_naive_per_metabolite.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_rep101_vs_naive_per_metabolite.png")


if __name__ == "__main__":
    make_figure()
