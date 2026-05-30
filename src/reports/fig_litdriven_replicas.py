"""Distribution of the 12 lit-driven replicas: 8 at T=0.4 plus 4 at other temperatures,
with baseline reference lines and 95% bootstrap CI for the T=0.4 config."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_litdriven_replicas_figure(output_dir="."):
    # T=0.4: 8 observations (3 v2 + 5 dedicated replicas).
    T04 = [0.164, 0.186, 0.147, 0.139, 0.189, 0.177, 0.046, 0.169]
    labels = ["v2-r2", "v2-r5", "v2-r6", "rep-100", "rep-101", "rep-102",
              "rep-103", "rep-104"]
    median_T04 = np.median(T04)
    ci_lo, ci_hi = 0.120, 0.176  # 95% bootstrap CI

    other_T = {
        "T=0.2 v2 r0": 0.173,
        "T=0.3 v2 r1": 0.127,
        "T=0.5 v2 r3": -0.111,
        "T=0.6 v2 r4": 0.014,
    }

    benchmarks = {
        "naive": 0.124,
        "HMDB initial": 0.160,
        "chenomx_emp": 0.184,
    }
    colors = {"naive": "#999999", "HMDB initial": "#cc6699",
              "chenomx_emp": "#5599cc"}

    fig, ax = plt.subplots(figsize=(11, 6.5))

    for name, val in benchmarks.items():
        ax.axvline(val, color=colors[name], ls="--", lw=1.5,
                   alpha=0.7, label=f"{name}={val}")

    y_T04 = np.linspace(0.85, 1.15, len(T04))
    ax.scatter(T04, y_T04, s=120, c="#cc6666", edgecolor="black",
               linewidth=1.0, label="T=0.4 (n=8)", zorder=3)
    for x, y, lab in zip(T04, y_T04, labels):
        ax.annotate(lab, (x, y), xytext=(5, 5),
                    textcoords="offset points", fontsize=8, alpha=0.6)

    ax.axvline(median_T04, color="#cc6666", lw=2, alpha=0.5,
               label=f"T=0.4 median = {median_T04:.3f}")

    ax.fill_betweenx([0.7, 1.3], ci_lo, ci_hi, alpha=0.15, color="#cc6666",
                     label=f"95% bootstrap CI [{ci_lo}, {ci_hi}]")

    y_other = 0.4
    for i, (name, val) in enumerate(other_T.items()):
        ax.scatter(val, y_other - i * 0.07, s=80, marker="s",
                   c="#999999", edgecolor="black", alpha=0.6)
        ax.annotate(name, (val, y_other - i * 0.07), xytext=(8, 0),
                    textcoords="offset points", fontsize=8, alpha=0.7,
                    va="center")

    ax.set_xlim(-0.15, 0.35)
    ax.set_ylim(0, 1.5)
    ax.set_yticks([0.4 - 0.07 * 1.5, 1.0])
    ax.set_yticklabels(["Other T (v2)", "T = 0.4\n(n=8)"], fontsize=10)
    ax.set_xlabel("Spearman ρ (mean over valid metabolites)", fontsize=11)
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig_litdriven_replicas.png",
                dpi=150, bbox_inches="tight")
    plt.savefig(f"{output_dir}/fig_litdriven_replicas.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved fig_litdriven_replicas.png + .pdf in {output_dir}")


if __name__ == "__main__":
    make_litdriven_replicas_figure(".")
