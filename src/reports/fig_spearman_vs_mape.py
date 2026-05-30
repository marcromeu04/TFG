"""Spearman vs MAPE scatter for the lit-driven runs plus reference benchmarks.
MAPE is on log scale; rep-103 is flagged as a magnitude collapse (range hallucination)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_spearman_vs_mape_figure(output_dir="."):
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
        ("naive",        0.124, 108, "#999999"),
        ("chenomx_emp",  0.184, 68,  "#5599cc"),
        ("HMDB initial", 0.160, 85,  "#cc6699"),
        ("META-RF real", 0.466, 17,  "#225577"),
    ]

    fig, ax = plt.subplots(figsize=(11, 7))

    colors = {"tier_high": "#cc6666", "tier_med": "#ee9966",
              "winner": "#22aa55", "collapse": "#000000"}
    markers = {"tier_high": "o", "tier_med": "o",
               "winner": "*", "collapse": "X"}
    sizes = {"tier_high": 150, "tier_med": 120,
             "winner": 300, "collapse": 150}

    for name, sp, mp, tier in data:
        ax.scatter(sp, mp, s=sizes[tier], c=colors[tier],
                   marker=markers[tier], edgecolor="black", linewidth=1.0,
                   alpha=0.85, zorder=3)
        offset = (8, 5) if name != "rep-103" else (8, -10)
        ax.annotate(name, (sp, mp), xytext=offset,
                    textcoords="offset points", fontsize=9, alpha=0.8)

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
        "rep-101: ONLY one with\nSpearman>0.17 and MAPE<70%\n"
        "(PMIDs 31346171, 36306677...)",
        xy=(0.175, 58), xytext=(0.21, 200),
        arrowprops=dict(arrowstyle="->", color="#22aa55", lw=1.5),
        fontsize=9, color="#22aa55", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#e8ffea", ec="#22aa55"))

    ax.annotate(
        "rep-103: MAPE max=10²⁸\n(range hallucination)",
        xy=(0.046, 2990), xytext=(0.10, 3500),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
        fontsize=9, color="black",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fff0f0", ec="black"))

    ax.set_xlabel("Spearman ρ (mean over valid metabolites)", fontsize=11)
    ax.set_ylabel("MAPE (%) - median across metabolites", fontsize=11)
    ax.set_yscale("log")
    ax.set_xlim(0, 0.55)
    ax.set_ylim(10, 5000)
    ax.grid(alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig_spearman_vs_mape.png",
                dpi=150, bbox_inches="tight")
    plt.savefig(f"{output_dir}/fig_spearman_vs_mape.pdf",
                bbox_inches="tight")
    plt.close()
    print(f"Saved fig_spearman_vs_mape.png + .pdf in {output_dir}")


if __name__ == "__main__":
    make_spearman_vs_mape_figure(".")
