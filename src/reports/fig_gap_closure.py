"""Synth-real gap closure figure. Entries are annotated with evaluation protocol via color."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np


PROTOCOL_COLORS = {
    "uniform":  "#cc6666",  # synthetic, single shot OOF, n_synth=5000
    "agent":    "#cc8855",  # multi agent internal evaluation
    "real_aug": "#888888",  # 3 seed mean on real cohort
    "real_oob": "#225577",  # bootstrap OOB on real cohort
}


def make_gap_closure_figure(output_dir="."):
    # (name, rho, n, protocol, lo, hi)
    methods = [
        ("automated research (CV select)",   0.054,  7, "uniform", None, None),
        ("LLM canonical (one shot)",         0.085,  6, "uniform", None, None),
        ("multi agent V2 best",              0.095,  8, "agent",   None, None),
        ("naive synthetic",                  0.124, 10, "uniform", None, None),
        ("multi agent V1 best",              0.135,  8, "agent",   None, None),
        ("LLM HMDB anchored",                0.154, 11, "uniform", None, None),
        ("lit driven (rep-101, 7 T=0.4)",    0.175,  5, "uniform", 0.143, 0.186),
        ("chenomx empirical",                0.184, 11, "uniform", None, None),
        ("ridge_bins",                       0.414, 12, "real_oob",0.301, 0.509),
        ("META-RF stacking",                 0.466, 12, "real_oob",0.395, 0.535),
        ("META-RF + mixup",                  0.479, 12, "real_aug",None, None),
    ]

    fig, ax = plt.subplots(figsize=(11, 7.3))
    y = np.arange(len(methods))
    rhos = [m[1] for m in methods]
    cols = [PROTOCOL_COLORS[m[3]] for m in methods]

    bars = ax.barh(y, rhos, color=cols, edgecolor="black", linewidth=0.6)
    for i, (name, r, n, proto, lo, hi) in enumerate(methods):
        if lo is not None and hi is not None:
            ax.errorbar(r, y[i], xerr=[[r - lo], [hi - r]],
                        fmt="none", ecolor="black", capsize=4,
                        lw=1.2, zorder=4)
        ax.text(max(r, hi or r) + 0.008, y[i],
                f"ρ={r:.3f}  (n={n})",
                va="center", fontsize=9)

    labels = [m[0] for m in methods]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel(r"Spearman $\rho$ on the cohort (N=253)",
                  fontsize=11)
    ax.set_xlim(-0.02, 0.66)
    ax.set_ylim(-0.6, len(methods) - 0.4)

    ax.axvspan(0, 0.20, alpha=0.04, color="red", zorder=0)
    ax.axvspan(0.40, 0.66, alpha=0.05, color="green", zorder=0)

    ax.axvline(0.124, color="#999", lw=0.9, ls=":")
    ax.axvline(0.184, color="#5599cc", lw=0.9, ls=":")

    proto_legend = [
        Patch(facecolor=PROTOCOL_COLORS["uniform"], edgecolor="black",
              label="synthetic (uniform OOF)"),
        Patch(facecolor=PROTOCOL_COLORS["agent"], edgecolor="black",
              label="multi agent (agent eval)"),
        Patch(facecolor=PROTOCOL_COLORS["real_aug"], edgecolor="black",
              label="classical aug (3 seed mean)"),
        Patch(facecolor=PROTOCOL_COLORS["real_oob"], edgecolor="black",
              label="supervised real (bootstrap OOB)"),
    ]
    ax.legend(handles=proto_legend, loc="lower right", fontsize=8.5,
              framealpha=0.92)

    ax.set_title("")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig_gap_closure.png", dpi=150,
                bbox_inches="tight")
    plt.savefig(f"{output_dir}/fig_gap_closure.pdf",
                bbox_inches="tight")
    plt.close()
    print(f"Saved fig_gap_closure.png + .pdf in {output_dir}")


if __name__ == "__main__":
    make_gap_closure_figure(".")
