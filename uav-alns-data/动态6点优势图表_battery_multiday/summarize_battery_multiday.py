#!/usr/bin/env python3
"""
Aggregate the battery-multiday sweep results into LaTeX-ready tables.

Inputs:
  ../battery_multiday_sweep/A_M7_multiday/record_fitness_*.mat
  ../battery_multiday_sweep/B_model_sensitivity_jan01/record_fitness_*.mat

For each (model, date, platform, mode) the .mat shape is
  (12 instances, REPS, 1 algo, 3 schedule_refs).

Aggregation rule (matches manuscript main experiment):
  - Column 0 of schedule_refs is the canonical 06:00/coldest comparison.
  - Both static and dynamic use column 0.

Outputs:
  - battery_multiday_summary.csv : per-(model, date, platform) mean ± SD with
    Improvement % (Dynamic vs Static).
  - latex_tables.txt : ready-to-paste LaTeX rows for §VI.F and §VI multi-day
    subsections.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import scipy.io as sio


SCRIPT_DIR = Path(__file__).resolve().parent
SWEEP_DIR  = SCRIPT_DIR.parent / "battery_multiday_sweep"
OUT_CSV    = SCRIPT_DIR / "battery_multiday_summary.csv"
OUT_TEX    = SCRIPT_DIR / "latex_tables.txt"


MODEL_NAME = {
    1: "M1 piecewise power-law",
    2: "M2 Arrhenius",
    3: "M3 generalized Peukert",
    4: "M4 polynomial",
    5: "M5 single exponential",
    6: "M6 dual-exp resistance",
    7: "M7 V-shaped (main)",
    8: "M8 piecewise linear",
}


DATE_DIR_RE = __import__("re").compile(r"^\d{4}-\d{2}-\d{2}$")


def parse_filename(p: Path) -> dict | None:
    """Parse 'record_fitness_[static_]<wx>_m<k>_<plat>_platform_fixed_dispatch_<x>.mat'.

    Date is derived from the immediate parent directory when it matches
    YYYY-MM-DD (Block A date subdirs); otherwise None (Block B Jan-01)."""
    stem = p.stem
    if not stem.startswith("record_fitness_"):
        return None
    body = stem[len("record_fitness_"):]
    is_static = False
    if body.startswith("static_"):
        is_static = True
        body = body[len("static_"):]
    parts = body.split("_")
    if len(parts) < 7:
        return None
    wx_raw   = parts[0]
    m_part   = parts[1]
    plat     = parts[2]
    model_id = int(m_part[1:])
    date     = p.parent.name if DATE_DIR_RE.match(p.parent.name) else None
    return {
        "model": model_id,
        "weather": "Real" if wx_raw == "real" else "Cos",
        "platform": "Multi" if plat == "multi" else "Single",
        "mode": "Static" if is_static else "Dynamic",
        "date": date,
        "path": p,
    }


def load_col0(p: Path) -> tuple[np.ndarray, list[str]]:
    """Load .mat -> (column-0 of record_fitness, dataset list).
    Result shape: (12 instances, REPS)."""
    m = sio.loadmat(p)
    key = next(k for k in m if k.startswith("record_fitness"))
    fit = m[key]  # (12, REPS, 1, 3)
    ds = [d[0] for d in m["datasets"][0]]
    return fit.squeeze(2)[:, :, 0], ds


def aggregate_cells(sub_dir: Path) -> dict:
    """Return nested dict: cells[(model, weather, platform, date)][mode] = array (12, REPS).

    Walks recursively so Block A's date-subdirs are picked up. date is None
    for Block B (no subdirs)."""
    files = sorted(sub_dir.rglob("record_fitness_*.mat"))
    cells = defaultdict(dict)
    datasets = None
    for p in files:
        meta = parse_filename(p)
        if meta is None:
            continue
        arr, ds = load_col0(p)
        datasets = ds
        key = (meta["model"], meta["weather"], meta["platform"], meta["date"])
        cells[key][meta["mode"]] = arr
    cells["__datasets__"] = datasets
    return cells


def scale_of(name: str) -> int:
    return int(name.rsplit(".", 1)[0].split("_")[-1])


def compute_summary(cells: dict, label: str, model_filter=None, dates=None):
    """Return list of summary rows. Each row is one (model, platform, scale-or-All) cell."""
    rows = []
    datasets = cells.get("__datasets__")
    if datasets is None:
        return rows
    keys = [k for k in cells.keys() if isinstance(k, tuple)]
    if model_filter is not None:
        keys = [k for k in keys if k[0] in model_filter]
    # Only Real weather for headline; Cos optional
    for plat in ("Single", "Multi"):
        plat_keys = [k for k in keys if k[1] == "Real" and k[2] == plat]
        for k in sorted(plat_keys, key=lambda x: (x[0], x[3] or "", x[2])):
            model = k[0]
            date  = k[3]
            if "Static" not in cells[k] or "Dynamic" not in cells[k]:
                continue
            s = cells[k]["Static"]
            d = cells[k]["Dynamic"]
            sm = s.mean(); ss = s.std(ddof=1)
            dm = d.mean(); dsd = d.std(ddof=1)
            impr = (sm - dm) / sm * 100
            wins = sum(1 for i in range(12) if s[i].mean() > d[i].mean())
            rows.append({
                "block": label,
                "model": model,
                "date": date or "2024-01-01",
                "platform": plat,
                "scale": "All",
                "n_inst": 12,
                "static_mean": sm, "static_sd": ss,
                "dyn_mean": dm,   "dyn_sd":  dsd,
                "impr_pct": impr,
                "wins": f"{wins}/12",
            })
    return rows


def write_csv(rows: list, path: Path):
    if not rows:
        print(f"WARNING: no rows to write to {path}")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {len(rows)} rows -> {path}")


def write_latex(rows_A: list, rows_B: list, path: Path):
    lines = []
    lines.append("% ============================================")
    lines.append("% Block (A): M7 multi-day robustness")
    lines.append("% Columns: date & platform & static (mean ± SD) & dynamic (mean ± SD) & impr% & wins")
    lines.append("% ============================================")
    for r in rows_A:
        lines.append(
            f"  {r['date']} & {r['platform']} & "
            f"{r['static_mean']:.2f} ({r['static_sd']:.2f}) & "
            f"{r['dyn_mean']:.2f} ({r['dyn_sd']:.2f}) & "
            f"{r['impr_pct']:.2f} & {r['wins']} \\\\"
        )
    lines.append("")
    lines.append("% ============================================")
    lines.append("% Block (B): battery model sensitivity (Jan 1)")
    lines.append("% Cross-model headline numbers")
    lines.append("% ============================================")
    for r in rows_B:
        name = MODEL_NAME.get(r['model'], f"model {r['model']}")
        lines.append(
            f"  {name} & {r['platform']} & "
            f"{r['static_mean']:.2f} ({r['static_sd']:.2f}) & "
            f"{r['dyn_mean']:.2f} ({r['dyn_sd']:.2f}) & "
            f"{r['impr_pct']:.2f} \\\\"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"LaTeX-ready table rows -> {path}")


def main() -> None:
    if not SWEEP_DIR.exists():
        print(f"ERROR: sweep directory not found: {SWEEP_DIR}", file=sys.stderr)
        print("Run run_battery_and_multiday_sweep.m first.", file=sys.stderr)
        sys.exit(1)

    # Block A
    a_dir = SWEEP_DIR / "A_M7_multiday"
    rows_A = []
    if a_dir.exists():
        cells_A = aggregate_cells(a_dir)
        rows_A = compute_summary(cells_A, "A_M7_multiday")

    # Block B
    b_dir = SWEEP_DIR / "B_model_sensitivity_jan01"
    rows_B = []
    if b_dir.exists():
        cells_B = aggregate_cells(b_dir)
        rows_B = compute_summary(cells_B, "B_model_sensitivity_jan01")

    all_rows = rows_A + rows_B
    write_csv(all_rows, OUT_CSV)
    write_latex(rows_A, rows_B, OUT_TEX)

    print(f"\nTotal aggregated rows: {len(all_rows)}  (Block A: {len(rows_A)}, Block B: {len(rows_B)})")
    print("\nSummary preview (all rows):")
    for r in all_rows:
        print(f"  [{r['block']:28s}] m={r['model']} {r['date']} {r['platform']:6s} "
              f"stat={r['static_mean']:8.2f}±{r['static_sd']:6.2f} "
              f"dyn={r['dyn_mean']:8.2f}±{r['dyn_sd']:6.2f} "
              f"impr={r['impr_pct']:+6.2f}% wins={r['wins']}")


if __name__ == "__main__":
    main()
