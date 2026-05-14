from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import csv
import re

import matplotlib
matplotlib.use("Agg")
from matplotlib import colormaps, cm, colors
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.interpolate import PchipInterpolator


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = Path(__file__).resolve().parent
OUT_ROOT = RESULT_DIR / "航路图_png"
BEST_ROUTE_FILES = sorted(RESULT_DIR.glob("*_best_routes.md"))
WEATHER_CSV = ROOT / "shenyang_hourly_temp_2024_01_01_to_05.csv"
TARGET_DATE = "2024-01-01"
CASE_CACHE: dict[tuple[str, bool], np.ndarray] = {}
SECTION_CACHE: dict[Path, dict[tuple[str, str, str], dict]] = {}
REAL_TEMP_SEQ: np.ndarray | None = None

EXTRA_PLATFORM_COORDS = np.array([
    [20.0, 30.0],
    [57.0, 31.0],
], dtype=float)

TEMP_NORM = colors.Normalize(vmin=-20.0, vmax=0.0, clip=True)
TEMP_CMAP = colormaps["turbo"]


@dataclass
class RouteRecord:
    source_file: Path
    algorithm: str
    case_name: str
    scenario: str
    best_value: str
    repeat_id: str
    route_tokens: list[int]
    multi_platform: bool
    scenario_group: str
    source_tag: str


def minutes_from_clock(value: str) -> int:
    value = value.strip()
    day_offset = 0
    day_match = re.match(r"D\+(\d+)\s+(\d{2}:\d{2})$", value)
    if day_match:
        day_offset = int(day_match.group(1)) * 1440
        value = day_match.group(2)
    hh, mm = map(int, value.split(":"))
    return day_offset + hh * 60 + mm


def load_real_temperature_sequence() -> np.ndarray:
    global REAL_TEMP_SEQ
    if REAL_TEMP_SEQ is not None:
        return REAL_TEMP_SEQ

    hours_min: list[int] = []
    temps: list[float] = []
    with WEATHER_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dt = datetime.strptime(row["time_beijing"], "%Y-%m-%d %H:%M:%S")
            if dt.strftime("%Y-%m-%d") != TARGET_DATE:
                continue
            hours_min.append(dt.hour * 60 + dt.minute)
            temps.append(float(row["temperature_c"]))

    unique_map: dict[int, float] = {}
    for minute, temp in zip(hours_min, temps):
        if minute not in unique_map:
            unique_map[minute] = temp

    unique_minutes = np.array(sorted(unique_map), dtype=float)
    unique_temps = np.array([unique_map[int(m)] for m in unique_minutes], dtype=float)

    if unique_minutes[-1] < 1440:
        unique_minutes = np.append(unique_minutes, 1440.0)
        unique_temps = np.append(unique_temps, unique_map.get(0, unique_temps[0]))

    minute_seq = np.arange(0, 1441, dtype=float)
    REAL_TEMP_SEQ = PchipInterpolator(unique_minutes, unique_temps)(minute_seq)
    return REAL_TEMP_SEQ


def cosine_temperature(minute: float, offset: int) -> float:
    shifted = minute + offset
    t_mean = -10.0
    t_amp = 20.0
    t_peak = 840.0
    return t_mean + (t_amp / 2.0) * np.cos(2.0 * np.pi * (shifted - t_peak) / 1440.0)


def real_temperature(minute: float, offset: int) -> float:
    seq = load_real_temperature_sequence()
    shifted = (minute + offset) % 1440.0
    lower = int(np.floor(shifted))
    upper = min(lower + 1, 1440)
    frac = shifted - lower
    if upper == lower:
        return float(seq[lower])
    return float(seq[lower] * (1 - frac) + seq[upper] * frac)


def scenario_offset_minutes(scenario: str) -> int:
    match = re.search(r"温度参考(\d{2}):(\d{2})", scenario)
    if not match:
        return 0
    hh, mm = map(int, match.groups())
    return hh * 60 + mm


def scenario_constant_temperature(scenario: str) -> float:
    match = re.search(r"\(([-+]?\d+(?:\.\d+)?)°C", scenario)
    if not match:
        raise ValueError(f"无法从场景中解析温度: {scenario}")
    return float(match.group(1))


def temperature_at(record: RouteRecord, minute: float) -> float:
    if record.scenario_group == "最低温数据":
        if "_real_" in record.source_file.name:
            return real_temperature(minute, 0)
        return cosine_temperature(minute, 0)

    if record.source_tag.startswith("static_"):
        return scenario_constant_temperature(record.scenario)

    offset = scenario_offset_minutes(record.scenario)
    if "_real_" in record.source_file.name:
        return real_temperature(minute, offset)
    return cosine_temperature(minute, offset)


def load_case(case_name: str, multi_platform: bool) -> np.ndarray:
    key = (case_name, multi_platform)
    if key in CASE_CACHE:
        return CASE_CACHE[key]

    data = np.loadtxt(ROOT / case_name)
    data[:, 0] = data[:, 0] + 1
    if multi_platform:
        base_platform = data[0:1, :].copy()
        first_customer_idx = int(np.argmax(data[:, 3] != 0))
        customers = data[first_customer_idx:, :].copy()
        extra_platforms = np.repeat(base_platform, len(EXTRA_PLATFORM_COORDS), axis=0)
        extra_platforms[:, 1:3] = EXTRA_PLATFORM_COORDS
        data = np.vstack([base_platform, extra_platforms, customers])
        data[:, 0] = np.arange(1, len(data) + 1)

    CASE_CACHE[key] = data
    return data


def source_tag_from_filename(path: Path) -> str:
    name = path.name
    prefix = "static" if "static_" in name else "dynamic"
    weather = "real" if "_real_" in name else "cos"
    platform = "multi" if "multi_platform" in name else "single"
    return f"{prefix}_{weather}_{platform}"


def parse_route_tokens(route_text: str) -> list[int]:
    return [int(x) for x in route_text.strip("[]").split()]


def classify_scenario(scenario: str) -> str | None:
    if "温度参考06:00" in scenario:
        return "6点数据"
    if "最低温" in scenario:
        return "最低温数据"
    return None


def parse_best_route_file(path: Path) -> list[RouteRecord]:
    records: list[RouteRecord] = []
    multi_platform = "multi_platform" in path.name
    source_tag = source_tag_from_filename(path)
    pattern = re.compile(
        r"^\|\s*(?P<alg>[^|]+?)\s*\|\s*(?P<case>[^|]+?)\s*\|\s*(?P<scenario>[^|]+?)\s*\|"
        r"\s*(?P<best>[^|]+?)\s*\|\s*(?P<repeat>[^|]+?)\s*\|\s*`(?P<route>\[[^\]]+\])`\s*\|$"
    )

    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        scenario = match.group("scenario").strip()
        scenario_group = classify_scenario(scenario)
        if scenario_group is None:
            continue
        records.append(
            RouteRecord(
                source_file=path,
                algorithm=match.group("alg").strip(),
                case_name=match.group("case").strip(),
                scenario=scenario,
                best_value=match.group("best").strip(),
                repeat_id=match.group("repeat").strip(),
                route_tokens=parse_route_tokens(match.group("route")),
                multi_platform=multi_platform,
                scenario_group=scenario_group,
                source_tag=source_tag,
            )
        )
    return records


def parse_markdown_sections(path: Path) -> dict[tuple[str, str, str], dict]:
    if path in SECTION_CACHE:
        return SECTION_CACHE[path]

    lines = path.read_text(encoding="utf-8").splitlines()
    sections: dict[tuple[str, str, str], dict] = {}
    current_key: tuple[str, str, str] | None = None
    current_route: dict | None = None
    in_table = False

    header_pattern = re.compile(r"^###\s+(.+?)\s+\|\s+(.+?)\s+\|\s+(.+?)$")
    route_pattern = re.compile(r"^####\s+航线\s+(\d+)$")
    bullet_pattern = re.compile(r"^- ([^:]+):\s*(.+)$")

    for raw_line in lines:
        line = raw_line.strip()
        header_match = header_pattern.match(line)
        if header_match:
            alg, case_name, scenario = [x.strip() for x in header_match.groups()]
            current_key = (alg, case_name, scenario)
            sections[current_key] = {"routes": []}
            current_route = None
            in_table = False
            continue

        if current_key is None:
            continue

        route_match = route_pattern.match(line)
        if route_match:
            current_route = {"name": f"航线 {route_match.group(1)}", "rows": []}
            sections[current_key]["routes"].append(current_route)
            in_table = False
            continue

        if current_route is None:
            continue

        bullet_match = bullet_pattern.match(line)
        if bullet_match:
            field, value = bullet_match.groups()
            current_route[field.strip()] = value.strip()
            continue

        if line.startswith("| 顺序 | 节点 | 类型 | 到达时间 | 开始服务 | 离开时间 |"):
            in_table = True
            continue
        if in_table and line.startswith("|---"):
            continue
        if in_table and line.startswith("|"):
            parts = [part.strip() for part in line.strip("|").split("|")]
            if len(parts) != 6:
                in_table = False
                continue
            current_route["rows"].append(
                {
                    "order": int(parts[0]),
                    "node": int(parts[1]),
                    "node_type": parts[2],
                    "arrival": None if parts[3] == "-" else minutes_from_clock(parts[3]),
                    "service_start": None if parts[4] == "-" else minutes_from_clock(parts[4]),
                    "leave": None if parts[5] == "-" else minutes_from_clock(parts[5]),
                }
            )
            continue

        if not line:
            in_table = False

    SECTION_CACHE[path] = sections
    return sections


def get_schedule(record: RouteRecord) -> list[dict]:
    key = (record.algorithm, record.case_name, record.scenario)
    sections = parse_markdown_sections(record.source_file)
    section = sections.get(key)
    if section is None:
        raise KeyError(f"未找到时间规划明细: {key} in {record.source_file}")
    return section["routes"]


def safe_text(value: str) -> str:
    value = value.replace(".txt", "")
    value = value.replace("固定00:00调度", "固定00-00调度")
    value = value.replace("温度参考06:00", "温度参考06-00")
    value = value.replace("06:00/08:00/14:00", "06-00_08-00_14-00")
    value = value.replace("°", "")
    value = value.replace("/", "_")
    value = value.replace("(", "_").replace(")", "")
    value = value.replace(",", "_")
    value = value.replace("=", "-")
    value = value.replace("+", "plus")
    value = value.replace(" ", "")
    value = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]", "", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def build_output_name(record: RouteRecord) -> str:
    scenario_part = safe_text(record.scenario)
    case_part = safe_text(record.case_name)
    return (
        f"{record.source_tag}__{record.algorithm}__{case_part}__"
        f"{scenario_part}__rep{int(record.repeat_id):02d}.png"
    )


def segment_collection_for_leg(
    p0: np.ndarray,
    p1: np.ndarray,
    t0: int,
    t1: int,
    record: RouteRecord,
) -> tuple[np.ndarray, np.ndarray]:
    duration = max(t1 - t0, 1)
    steps = max(int(np.ceil(duration / 5.0)), 1)
    fractions = np.linspace(0.0, 1.0, steps + 1)
    points = p0 + (p1 - p0) * fractions[:, None]
    segments = np.stack([points[:-1], points[1:]], axis=1)

    mid_fractions = (fractions[:-1] + fractions[1:]) / 2.0
    mid_times = t0 + duration * mid_fractions
    temps = np.array([temperature_at(record, t) for t in mid_times], dtype=float)
    return segments, temps


def build_route_temperature_segments(
    route_rows: list[dict],
    lookup: dict[int, np.ndarray],
    record: RouteRecord,
) -> tuple[np.ndarray, np.ndarray]:
    all_segments: list[np.ndarray] = []
    all_temps: list[np.ndarray] = []

    for current_row, next_row in zip(route_rows[:-1], route_rows[1:]):
        depart = current_row["leave"]
        arrive = next_row["arrival"]
        if depart is None or arrive is None or arrive <= depart:
            continue
        p0 = lookup[current_row["node"]][1:3].astype(float)
        p1 = lookup[next_row["node"]][1:3].astype(float)
        segments, temps = segment_collection_for_leg(p0, p1, depart, arrive, record)
        all_segments.append(segments)
        all_temps.append(temps)

    if not all_segments:
        return np.empty((0, 2, 2), dtype=float), np.empty((0,), dtype=float)
    return np.concatenate(all_segments, axis=0), np.concatenate(all_temps, axis=0)


def add_direction_arrow(ax: plt.Axes, route_rows: list[dict], lookup: dict[int, np.ndarray], record: RouteRecord, is_feasible: bool = True) -> None:
    for current_row, next_row in zip(route_rows[:-1], route_rows[1:]):
        depart = current_row["leave"]
        arrive = next_row["arrival"]
        if depart is None or arrive is None or arrive <= depart:
            continue
        p0 = lookup[current_row["node"]][1:3].astype(float)
        p1 = lookup[next_row["node"]][1:3].astype(float)
        color = TEMP_CMAP(TEMP_NORM(temperature_at(record, depart))) if is_feasible else "#ef4444"
        ax.annotate(
            "",
            xy=(p1[0], p1[1]),
            xytext=(p0[0], p0[1]),
            arrowprops=dict(arrowstyle="->", lw=1.0, color=color, mutation_scale=11, shrinkA=9, shrinkB=8, alpha=0.9),
            zorder=3,
        )
        return


def plot_record(record: RouteRecord) -> Path:
    data = load_case(record.case_name, record.multi_platform)
    lookup = {int(row[0]): row for row in data}
    schedule_routes = get_schedule(record)
    platform_count = 3 if record.multi_platform else 1
    platforms = data[:platform_count]
    customers = data[platform_count:]

    fig, ax = plt.subplots(figsize=(12.0, 8.2), facecolor="white")
    ax.set_facecolor("white")

    ax.scatter(
        customers[:, 1],
        customers[:, 2],
        s=24 + customers[:, 3] * 1.3,
        c="#e9eef5",
        edgecolors="#7a8aa0",
        linewidths=0.8,
        zorder=2,
    )
    ax.scatter(
        platforms[:, 1],
        platforms[:, 2],
        s=[320] + [220] * (platform_count - 1),
        marker="*",
        c=["#b22222"] + ["#f2b6b6"] * (platform_count - 1),
        edgecolors="#8b1e1e",
        linewidths=1.0,
        zorder=5,
    )

    for row in customers:
        ax.text(
            row[1] + 0.45,
            row[2] - 0.55,
            str(int(row[0])),
            fontsize=5.3 if len(customers) > 60 else 6.4,
            color="#4a5568",
            ha="left",
            va="top",
            zorder=3,
        )

    for row in platforms:
        ax.text(
            row[1] + 0.8,
            row[2] + 0.8,
            f"Platform {int(row[0])}",
            fontsize=8.0,
            color="#8b1e1e",
            ha="left",
            va="bottom",
            zorder=6,
        )

    legend_handles = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#b91c1c",
               markeredgecolor="#8b1e1e", markersize=11, linestyle="", label="Platform"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e2e8f0",
               markeredgecolor="#7a8aa0", markersize=6.5, linestyle="", label="Customer"),
    ]

    has_infeasible = any(route.get("可行性") == "否" for route in schedule_routes)
    if has_infeasible:
        legend_handles.append(Line2D([0], [0], color="#ef4444", lw=3.2, linestyle="--", label="Infeasible Route"))

    for idx, route in enumerate(schedule_routes, start=1):
        is_feasible = route.get("可行性") != "否"
        route_rows = route["rows"]
        segments, temps = build_route_temperature_segments(route_rows, lookup, record)
        if len(segments) > 0:
            if is_feasible:
                lc = LineCollection(
                    segments,
                    cmap=TEMP_CMAP,
                    norm=TEMP_NORM,
                    linewidths=3.2,
                    alpha=0.98,
                    zorder=4,
                )
                lc.set_array(temps)
            else:
                lc = LineCollection(
                    segments,
                    colors="#ef4444",
                    linewidths=3.2,
                    linestyles="--",
                    alpha=0.98,
                    zorder=4,
                )
            ax.add_collection(lc)
            add_direction_arrow(ax, route_rows, lookup, record, is_feasible)

            midpoint_row = route_rows[len(route_rows) // 2]
            mid_xy = lookup[midpoint_row["node"]][1:3]
            ax.text(
                mid_xy[0],
                mid_xy[1] + 0.95,
                f"R{idx}",
                fontsize=7.2,
                fontweight="semibold",
                color="#1f2937",
                ha="center",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="#5b6777", lw=0.8, alpha=0.95),
                zorder=7,
            )

    mode_label = "Static" if record.source_tag.startswith("static_") else "Dynamic"
    ax.set_title(
        f"{record.case_name.replace('.txt', '')} {mode_label}",
        fontsize=19,
        fontweight="semibold",
        pad=14,
    )

    x_min, x_max = np.min(data[:, 1]), np.max(data[:, 1])
    y_min, y_max = np.min(data[:, 2]), np.max(data[:, 2])
    ax.set_xlim(x_min - 5, x_max + 7)
    ax.set_ylim(y_min - 6, y_max + 6)
    ax.set_xlabel("X Coordinate", fontsize=12)
    ax.set_ylabel("Y Coordinate", fontsize=12)
    ax.grid(True, linestyle=(0, (3, 3)), linewidth=0.5, color="#cfd6df", alpha=0.9)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(axis="both", labelsize=11, width=0.9, length=4)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#222222")

    sm = cm.ScalarMappable(norm=TEMP_NORM, cmap=TEMP_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.042, pad=0.03)
    cbar.set_label("Flight-time Temperature (°C)", fontsize=12)
    cbar.ax.tick_params(labelsize=10.5, width=0.8, length=3.5)
    cbar.outline.set_linewidth(0.9)

    ax.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(1.12, 0.03),
        frameon=True,
        facecolor="white",
        edgecolor="#c6ccd4",
        framealpha=1.0,
        fontsize=10.5,
        borderpad=0.55,
        handletextpad=0.6,
    )

    fig.tight_layout(rect=[0, 0, 0.905, 0.965])
    out_dir = OUT_ROOT / record.scenario_group
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / build_output_name(record)
    fig.savefig(out_path, dpi=320, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def main() -> None:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = [
        "Times New Roman",
        "Times",
        "Liberation Serif",
        "Nimbus Roman",
        "DejaVu Serif",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    records: list[RouteRecord] = []
    for path in BEST_ROUTE_FILES:
        records.extend(parse_best_route_file(path))

    outputs = [plot_record(record) for record in records]
    print(f"共生成 {len(outputs)} 张 PNG 航路图")
    for group in ("6点数据", "最低温数据"):
        count = sum(1 for p in outputs if p.parent.name == group)
        print(f"{group}: {count} 张")


if __name__ == "__main__":
    main()
