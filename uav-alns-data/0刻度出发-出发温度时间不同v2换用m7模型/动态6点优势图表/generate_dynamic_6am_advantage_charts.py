#!/usr/bin/env python3
"""Generate charts highlighting Dynamic 06:00 planning advantages."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


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
    if row["Mode"] == "Dynamic":
        if "06:00" in scenario:
            return "动态6点"
        return None
    if "最低温" in scenario:
        return "静态最低温"
    if "平均温" in scenario:
        return "静态平均温"
    if "最高温" in scenario:
        return "静态最高温"
    return None


def load_rows() -> dict[tuple[str, str, str], dict[str, dict[str, str]]]:
    with CSV_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    grouped: dict[tuple[str, str, str], dict[str, dict[str, str]]] = {}
    for row in rows:
        kind = scenario_kind(row)
        if not kind:
            continue
        key = (row["Weather"], row["Platform"], row["Case"].replace(".txt", ""))
        grouped.setdefault(key, {})[kind] = row
    return grouped


def objective(row: dict[str, str]) -> float:
    return float(row["Objective"])


def unserved(row: dict[str, str]) -> int:
    return int(row["UnservedCount"])


def sorted_keys(grouped: dict[tuple[str, str, str], dict[str, dict[str, str]]]) -> list[tuple[str, str, str]]:
    def key_fn(item: tuple[str, str, str]) -> tuple[str, str, int, str]:
        weather, platform, case = item
        size = int(case.split("_")[-1])
        return weather, platform, size, case

    return sorted(grouped, key=key_fn)


def label_for(key: tuple[str, str, str]) -> str:
    weather, platform, case = key
    return f"{weather}-{platform}-{case}"


def plot_dumbbell(grouped: dict[tuple[str, str, str], dict[str, dict[str, str]]]) -> None:
    baselines = ["静态最低温", "静态平均温", "静态最高温"]
    colors = {
        "动态6点": "#d64a2f",
        "静态最低温": "#8d99ae",
        "静态平均温": "#5f6c7b",
        "静态最高温": "#2f3e46",
    }

    keys = [k for k in sorted_keys(grouped) if all(x in grouped[k] for x in ["动态6点", *baselines])]
    pages = [keys[i : i + 24] for i in range(0, len(keys), 24)]
    for page_no, page_keys in enumerate(pages, start=1):
        fig, ax = plt.subplots(figsize=(13, max(7, len(page_keys) * 0.34)))
        for y, key in enumerate(page_keys):
            values = {name: objective(grouped[key][name]) for name in ["动态6点", *baselines]}
            ax.plot(
                [min(values.values()), max(values.values())],
                [y, y],
                color="#d8dee4",
                linewidth=1.4,
                zorder=1,
            )
            for name, value in values.items():
                ax.scatter(value, y, s=45 if name == "动态6点" else 30, color=colors[name], label=name, zorder=3)

        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False)
        ax.set_yticks(range(len(page_keys)))
        ax.set_yticklabels([label_for(k) for k in page_keys], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Objective（越低越好）")
        ax.set_title(f"动态6点与静态温度基线 Objective 配对对比（第 {page_no} 页）")
        ax.grid(axis="x", color="#e9ecef", linewidth=0.8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"01_动态6点_Objective配对哑铃图_{page_no}.png")
        plt.close(fig)


def plot_improvement_heatmap(grouped: dict[tuple[str, str, str], dict[str, dict[str, str]]]) -> None:
    baselines = ["静态最低温", "静态平均温", "静态最高温"]
    keys = [k for k in sorted_keys(grouped) if all(x in grouped[k] for x in ["动态6点", *baselines])]
    matrix: list[list[float]] = []
    for baseline in baselines:
        row = []
        for key in keys:
            static_obj = objective(grouped[key][baseline])
            dynamic_obj = objective(grouped[key]["动态6点"])
            row.append((static_obj - dynamic_obj) / static_obj * 100)
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(max(14, len(keys) * 0.33), 4.8))
    cmap = LinearSegmentedColormap.from_list("advantage", ["#3b6ea8", "#f8f9fa", "#c93f2d"])
    max_abs = max(abs(v) for row in matrix for v in row)
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)

    ax.set_yticks(range(len(baselines)))
    ax.set_yticklabels([f"vs {x}" for x in baselines])
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([label_for(k) for k in keys], rotation=70, ha="right", fontsize=7)
    ax.set_title("动态6点 Objective 改善率热力图（红色为动态更优）")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("改善率 %")

    for y, row in enumerate(matrix):
        for x, value in enumerate(row):
            if abs(value) >= 8 or y == 0:
                ax.text(x, y, f"{value:.1f}", ha="center", va="center", fontsize=5.5, color="#111827")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_动态6点_改善率热力图.png")
    plt.close(fig)


def plot_unserved_bars(grouped: dict[tuple[str, str, str], dict[str, dict[str, str]]]) -> None:
    keys = []
    for key in sorted_keys(grouped):
        data = grouped[key]
        if "动态6点" in data and "静态最低温" in data:
            if unserved(data["静态最低温"]) > 0 or unserved(data["动态6点"]) > 0:
                keys.append(key)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = list(range(len(keys)))
    width = 0.36
    static_vals = [unserved(grouped[k]["静态最低温"]) for k in keys]
    dynamic_vals = [unserved(grouped[k]["动态6点"]) for k in keys]
    ax.bar([i - width / 2 for i in x], static_vals, width=width, label="静态最低温", color="#8d99ae")
    ax.bar([i + width / 2 for i in x], dynamic_vals, width=width, label="动态6点", color="#d64a2f")
    ax.set_xticks(x)
    ax.set_xticklabels([label_for(k) for k in keys], rotation=35, ha="right")
    ax.set_ylabel("未服务客户数")
    ax.set_title("低温不可行情形下动态6点的服务完整性优势")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#e9ecef", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    for i, value in enumerate(static_vals):
        ax.text(i - width / 2, value + 0.12, str(value), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_动态6点_vs_静态最低温_未服务客户数.png")
    plt.close(fig)


def plot_pareto(grouped: dict[tuple[str, str, str], dict[str, dict[str, str]]]) -> None:
    kinds = ["动态6点", "静态最低温", "静态平均温", "静态最高温"]
    colors = {
        "动态6点": "#d64a2f",
        "静态最低温": "#8d99ae",
        "静态平均温": "#5f6c7b",
        "静态最高温": "#2f3e46",
    }
    markers = {"Single": "o", "Multi": "s"}

    fig, ax = plt.subplots(figsize=(10.5, 6))
    for kind in kinds:
        for platform, marker in markers.items():
            xs, ys = [], []
            for key, data in grouped.items():
                if key[1] == platform and kind in data:
                    xs.append(objective(data[kind]))
                    ys.append(unserved(data[kind]))
            ax.scatter(
                xs,
                ys,
                s=44 if kind == "动态6点" else 30,
                color=colors[kind],
                marker=marker,
                alpha=0.82,
                label=f"{kind}-{platform}",
                edgecolor="white",
                linewidth=0.4,
            )

    ax.set_xlabel("Objective（越低越好）")
    ax.set_ylabel("未服务客户数（越低越好）")
    ax.set_title("Objective 与服务完整性 Pareto 散点图")
    ax.grid(color="#e9ecef", linewidth=0.8)
    ax.legend(ncol=2, fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_动态6点_Pareto散点图.png")
    plt.close(fig)


def write_summary(grouped: dict[tuple[str, str, str], dict[str, dict[str, str]]]) -> None:
    baselines = ["静态最低温", "静态平均温", "静态最高温"]
    lines = ["# 动态6点优势图表说明", ""]
    for baseline in baselines:
        improvements = []
        wins = 0
        unserved_improved = 0
        for data in grouped.values():
            if "动态6点" not in data or baseline not in data:
                continue
            static_obj = objective(data[baseline])
            dynamic_obj = objective(data["动态6点"])
            improvement = (static_obj - dynamic_obj) / static_obj * 100
            improvements.append(improvement)
            wins += int(dynamic_obj < static_obj)
            unserved_improved += int(unserved(data["动态6点"]) < unserved(data[baseline]))
        avg = sum(improvements) / len(improvements)
        ordered = sorted(improvements)
        mid = len(ordered) // 2
        median = (ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2)
        lines.append(
            f"- 动态6点相对{baseline}: 平均改善率 {avg:.2f}%, 中位数 {median:.2f}%, "
            f"Objective 更优 {wins}/{len(improvements)} 组, 未服务客户减少 {unserved_improved} 组。"
        )
    lines.extend(
        [
            "",
            "建议主推 `01_动态6点_Objective配对哑铃图_*.png` 和 `03_动态6点_vs_静态最低温_未服务客户数.png`。",
            "最高温属于更乐观的静态基线，更适合作为上界参照，不建议作为主要优势口径。",
        ]
    )
    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    configure_matplotlib()
    grouped = load_rows()
    if not grouped:
        raise RuntimeError(f"No data loaded from {CSV_PATH}")

    plot_dumbbell(grouped)
    plot_improvement_heatmap(grouped)
    plot_unserved_bars(grouped)
    plot_pareto(grouped)
    write_summary(grouped)

    images = sorted(p.name for p in OUT_DIR.glob("*.png"))
    print(f"Generated {len(images)} images in {OUT_DIR}")
    for name in images:
        print(f"- {name}")


if __name__ == "__main__":
    main()
