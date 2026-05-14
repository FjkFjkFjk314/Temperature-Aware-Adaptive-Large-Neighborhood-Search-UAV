#!/usr/bin/env python3
"""
Generate a single-column-width improvement ranking chart for the J-STARS manuscript.

Keeps only paired comparisons with positive improvement (Dynamic 06:00 better than
Static low-temperature baseline), and produces a vertically compact PDF sized for
IEEE single-column display (approx 3.5 inches wide).

Output: <manuscript>/Figure/improvement_ranking.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

# Reuse the existing data loader.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from generate_dynamic_6am_vs_static_low_charts import load_pairs  # noqa: E402

# Where the manuscript expects the figure.
FIGURE_TARGET = Path(
    "/Users/kunjinkao/my_project/uav-alns/applied-earth-observations-and-remote-sensing"
    "/uav-alns-jstars/Figure/improvement_ranking.pdf"
)


def configure_matplotlib() -> None:
    candidates = [
        "PingFang SC",
        "Hiragino Sans GB",
        "Heiti SC",
        "Songti SC",
        "Arial Unicode MS",
        "Noto Sans CJK SC",
        "SimHei",
    ]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300


MIN_IMPROVEMENT_PCT = 5.0  # Drop cases below 5% to suppress visually negligible bars.


def plot_positive_ranking_singlecol(pairs: list[dict[str, object]]) -> None:
    # Keep only meaningfully positive improvements (>= MIN_IMPROVEMENT_PCT);
    # sort ascending so the largest bar appears at the top.
    positive = [p for p in pairs if float(p["improvement"]) >= MIN_IMPROVEMENT_PCT]
    positive_sorted = sorted(positive, key=lambda row: float(row["improvement"]))

    n = len(positive_sorted)
    # IEEE single column ~3.5 in; height scales with bar count for readable labels.
    fig_w = 3.5
    fig_h = max(4.0, 0.13 * n + 1.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    y = list(range(n))
    values = [float(row["improvement"]) for row in positive_sorted]
    labels = [str(row["label"]) for row in positive_sorted]

    ax.barh(y, values, color="#d64a2f", height=0.78)
    ax.axvline(0, color="#1f2937", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.0)
    ax.tick_params(axis="x", labelsize=8.0)
    ax.set_xlabel("Objective Improvement (%)", fontsize=9.0)
    ax.grid(axis="x", color="#e9ecef", linewidth=0.6)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_xlim(left=0, right=max(values) * 1.18)

    # Annotate every bar with its value (chart is now sparse enough to allow).
    for i, value in enumerate(values):
        ax.text(value + max(values) * 0.012, i, f"{value:.1f}",
                va="center", ha="left", fontsize=7.0, color="#1f2937")

    ax.set_ylim(-0.7, n - 0.3)

    fig.tight_layout(pad=0.4)

    # Save preview alongside script (PDF) and overwrite manuscript figure.
    preview = SCRIPT_DIR / "improvement_ranking_singlecol_preview.pdf"
    fig.savefig(preview, format="pdf", bbox_inches="tight")
    if FIGURE_TARGET.parent.exists():
        fig.savefig(FIGURE_TARGET, format="pdf", bbox_inches="tight")
        print(f"Wrote: {FIGURE_TARGET}")
    else:
        print(f"WARNING: Target dir not found: {FIGURE_TARGET.parent}")
    print(f"Preview: {preview}")
    print(f"Bars (positive only): {n} / {len(pairs)} total pairs")
    plt.close(fig)


def main() -> None:
    configure_matplotlib()
    pairs = load_pairs()
    plot_positive_ranking_singlecol(pairs)


if __name__ == "__main__":
    main()
