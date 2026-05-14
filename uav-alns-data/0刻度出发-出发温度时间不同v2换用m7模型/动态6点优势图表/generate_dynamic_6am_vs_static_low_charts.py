#!/usr/bin/env python3
"""Generate charts comparing Dynamic 06:00 planning with Static low-temperature planning."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent
CSV_PATH = ROOT / "metrics_summary.csv"


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
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 220


def scenario_kind(row: dict[str, str]) -> str | None:
    scenario = row["Scenario"]
    if row["Mode"] == "Dynamic" and "06:00" in scenario:
        return "动态6点"
    if row["Mode"] == "Static" and "最低温" in scenario:
        return "静态低温"
    return None


def load_pairs() -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], dict[str, dict[str, str]]] = {}
    with CSV_PATH.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            kind = scenario_kind(row)
            if not kind:
                continue
            key = (row["Weather"], row["Platform"], row["Case"].replace(".txt", ""))
            grouped.setdefault(key, {})[kind] = row

    pairs = []
    for key, data in grouped.items():
        if "动态6点" not in data or "静态低温" not in data:
            continue
        static_obj = float(data["静态低温"]["Objective"])
        dynamic_obj = float(data["动态6点"]["Objective"])
        improvement = (static_obj - dynamic_obj) / static_obj * 100
        pairs.append(
            {
                "weather": key[0],
                "platform": key[1],
                "case": key[2],
                "label": f"{key[0]}-{key[1]}-{key[2]}",
                "static_obj": static_obj,
                "dynamic_obj": dynamic_obj,
                "improvement": improvement,
                "static_unserved": int(data["静态低温"]["UnservedCount"]),
                "dynamic_unserved": int(data["动态6点"]["UnservedCount"]),
            }
        )

    def sort_key(row: dict[str, object]) -> tuple[str, str, int, str]:
        case = str(row["case"])
        return str(row["weather"]), str(row["platform"]), int(case.split("_")[-1]), case

    return sorted(pairs, key=sort_key)


def plot_dumbbell(pairs: list[dict[str, object]]) -> None:
    pages = [pairs[i : i + 24] for i in range(0, len(pairs), 24)]
    for page_no, page in enumerate(pages, start=1):
        fig, ax = plt.subplots(figsize=(12.5, max(7, len(page) * 0.35)))
        for y, row in enumerate(page):
            static_obj = float(row["static_obj"])
            dynamic_obj = float(row["dynamic_obj"])
            line_color = "#d64a2f" if dynamic_obj < static_obj else "#6c757d"
            ax.plot([static_obj, dynamic_obj], [y, y], color=line_color, linewidth=1.7, alpha=0.7, zorder=1)
            ax.scatter(static_obj, y, s=34, color="#8d99ae", label="Static Low Temperature", zorder=3)
            ax.scatter(dynamic_obj, y, s=46, color="#d64a2f", label="Dynamic 06:00", zorder=4)

        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc="upper right", frameon=True, framealpha=0.92, edgecolor="#d8dee4")
        ax.set_yticks(range(len(page)))
        ax.set_yticklabels([str(row["label"]) for row in page], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Objective (lower is better)")
        ax.set_title(f"Dynamic 06:00 vs Static Low Temperature Objective Dumbbell Plot (Page {page_no})")
        ax.grid(axis="x", color="#e9ecef", linewidth=0.8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"低温对比_01_Objective配对哑铃图_{page_no}.png")
        plt.close(fig)


def plot_sorted_improvement_bars(pairs: list[dict[str, object]]) -> None:
    ordered = sorted(pairs, key=lambda row: float(row["improvement"]))
    colors = ["#3b6ea8" if float(row["improvement"]) < 0 else "#d64a2f" for row in ordered]

    fig, ax = plt.subplots(figsize=(12, max(8, len(ordered) * 0.23)))
    y = range(len(ordered))
    values = [float(row["improvement"]) for row in ordered]
    ax.barh(y, values, color=colors, height=0.68)
    ax.axvline(0, color="#1f2937", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([str(row["label"]) for row in ordered], fontsize=7.5)
    ax.set_xlabel("Objective Improvement (%)")
    ax.set_title("Improvement Ranking: Dynamic 06:00 vs Static Low Temperature")
    ax.grid(axis="x", color="#e9ecef", linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for i, value in enumerate(values):
        if abs(value) >= 8:
            ha = "left" if value >= 0 else "right"
            offset = 0.35 if value >= 0 else -0.35
            ax.text(value + offset, i, f"{value:.1f}%", va="center", ha=ha, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "低温对比_02_改善率排序条形图.png")
    plt.close(fig)


def plot_cumulative_win_curve(pairs: list[dict[str, object]]) -> None:
    improvements = sorted(float(row["improvement"]) for row in pairs)
    thresholds = list(range(-10, 51, 1))
    counts = [sum(1 for value in improvements if value >= threshold) for threshold in thresholds]
    rates = [count / len(improvements) * 100 for count in counts]

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.plot(thresholds, rates, color="#d64a2f", linewidth=2.4)
    ax.fill_between(thresholds, rates, color="#d64a2f", alpha=0.12)
    ax.axvline(0, color="#1f2937", linewidth=1)
    ax.axhline(sum(1 for v in improvements if v > 0) / len(improvements) * 100, color="#8d99ae", linewidth=1, linestyle="--")

    markers = [0, 5, 10, 20, 30]
    for threshold in markers:
        rate = sum(1 for value in improvements if value >= threshold) / len(improvements) * 100
        ax.scatter([threshold], [rate], color="#d64a2f", s=36, zorder=3)
        ax.text(threshold, rate + 2.2, f">={threshold}%: {rate:.0f}%", ha="center", fontsize=8)

    win_count = sum(1 for value in improvements if value > 0)
    ax.text(
        0.98,
        0.92,
        f"Win rate: {win_count}/{len(improvements)} = {win_count / len(improvements) * 100:.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d8dee4"},
    )
    ax.set_xlabel("Improvement Threshold (%)")
    ax.set_ylabel("Cases Meeting Threshold (%)")
    ax.set_title("Cumulative Advantage Curve / Win Rate: Dynamic 06:00 vs Static Low Temperature")
    ax.set_ylim(0, 105)
    ax.grid(color="#e9ecef", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "低温对比_03_累计优势曲线_胜率图.png")
    plt.close(fig)


def write_summary(pairs: list[dict[str, object]]) -> None:
    improvements = [float(row["improvement"]) for row in pairs]
    wins = sum(1 for value in improvements if value > 0)
    unserved_improved = sum(1 for row in pairs if int(row["dynamic_unserved"]) < int(row["static_unserved"]))
    mean = sum(improvements) / len(improvements)
    ordered = sorted(improvements)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    lines = [
        "# 动态6点 vs 静态低温图表说明",
        "",
        f"- 对比组合数: {len(pairs)}",
        f"- Objective 更优: {wins}/{len(pairs)} 组，胜率 {wins / len(pairs) * 100:.1f}%",
        f"- 平均改善率: {mean:.2f}%",
        f"- 中位数改善率: {median:.2f}%",
        f"- 未服务客户减少: {unserved_improved} 组",
        "",
        "图片文件:",
        "- `低温对比_01_Objective配对哑铃图_1.png`",
        "- `低温对比_01_Objective配对哑铃图_2.png`",
        "- `低温对比_02_改善率排序条形图.png`",
        "- `低温对比_03_累计优势曲线_胜率图.png`",
    ]
    (OUT_DIR / "低温对比_README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    configure_matplotlib()
    pairs = load_pairs()
    if not pairs:
        raise RuntimeError(f"No Dynamic 06:00 vs Static low-temperature pairs found in {CSV_PATH}")
    plot_dumbbell(pairs)
    plot_sorted_improvement_bars(pairs)
    plot_cumulative_win_curve(pairs)
    write_summary(pairs)

    images = sorted(p.name for p in OUT_DIR.glob("低温对比_*.png"))
    print(f"Generated {len(images)} low-temperature comparison images in {OUT_DIR}")
    for name in images:
        print(f"- {name}")


if __name__ == "__main__":
    main()
