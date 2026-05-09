"""Plot Materials Project coverage gaps from void_scores.csv.

Input:  data/results/void_scores.csv
Output: reports/figures/void_scores_plot.png
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "results"
FIGURES_DIR = ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_void_scores(df: pd.DataFrame, out: Path) -> None:
    df = df.sort_values("void_score", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))

    for i, (_, row) in enumerate(df.iterrows()):
        if row["zero_stable_phases"]:
            ax.barh(i, max(row["total_entries"], 0.6), color="#d62728", edgecolor="none", alpha=0.85)
            if row["total_entries"] == 0:
                ax.text(0.65, i, "no entries", va="center", fontsize=8, color="#d62728", style="italic")
        else:
            ax.barh(i, row["total_entries"], color="#d0d0d0", edgecolor="none")
            ax.barh(i, row["stable_phases"], color="#1f77b4", edgecolor="none")

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["system"])
    ax.set_xlim(0, df["total_entries"].max() * 1.5)

    for i, (_, row) in enumerate(df.iterrows()):
        label = f"  score: {row['void_score']:.2f}  |  {row['stable_phases']}/{row['total_entries']} stable"
        ax.text(max(row["total_entries"], 0.6) + 0.3, i, label, va="center", fontsize=8.5, color="#333333")

    ax.set_xlabel("Number of MP entries", fontsize=11)
    ax.set_title("Materials Project coverage gaps by binary system", fontsize=13, fontweight="bold")
    ax.legend(
        handles=[
            mpatches.Patch(color="#d62728", label="Zero stable phases (gap)"),
            mpatches.Patch(color="#1f77b4", label="Stable phases (≥1)"),
            mpatches.Patch(color="#d0d0d0", label="Unstable / above-hull entries"),
        ],
        loc="lower right",
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


def main() -> None:
    csv = DATA_DIR / "void_scores.csv"
    if not csv.exists():
        raise FileNotFoundError(f"{csv} not found — run query_systems.py first.")
    df = pd.read_csv(csv)
    plot_void_scores(df, FIGURES_DIR / "void_scores_plot.png")


if __name__ == "__main__":
    main()
