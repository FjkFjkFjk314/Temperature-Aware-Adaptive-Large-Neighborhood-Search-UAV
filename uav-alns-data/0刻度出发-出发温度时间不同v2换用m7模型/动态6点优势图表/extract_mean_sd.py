#!/usr/bin/env python3
"""
Extract mean and standard deviation across the 5 independent runs from the
v2-with-M7 record_fitness_*.mat files. Outputs one row per (scenario, instance)
suitable for tabularizing in the manuscript.

Data structure (per .mat file):
  record_fitness_*  : shape (12 instances, 5 reps, 1 algo, 3 schedules)
  datasets          : the 12 Solomon instance names
  schedule_labels   : Chinese labels for the three reference temperatures
                      (06:00, 08:00, 14:00)

Aggregation rules (matches metrics_summary.csv scenario_kind logic):
  - Both static and dynamic use schedule column 0 (06:00 reference / coldest
    temperature for static "lowest temperature" baseline; 06:00 dispatch for
    dynamic). This is the canonical comparison in the manuscript.
  - "best-of-5" used in current tab:main-results: min over 5 reps within
    column 0.
  - We report mean and SD over the 5 reps within column 0 — fair comparison
    paired at the same reference scenario.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

DATA_DIR = Path(__file__).resolve().parents[1]
FILES = {
    ("Static", "Real",  "Single"): "record_fitness_static_real_m7_single_platform_fixed_dispatch_0.mat",
    ("Static", "Real",  "Multi"):  "record_fitness_static_real_m7_multi_platform_fixed_dispatch_0.mat",
    ("Static", "Cos",   "Single"): "record_fitness_static_cos_m7_single_platform_fixed_dispatch_0.mat",
    ("Static", "Cos",   "Multi"):  "record_fitness_static_cos_m7_multi_platform_fixed_dispatch_0.mat",
    ("Dynamic","Real",  "Single"): "record_fitness_real_m7_single_platform_fixed_dispatch_0.mat",
    ("Dynamic","Real",  "Multi"):  "record_fitness_real_m7_multi_platform_fixed_dispatch_0.mat",
    ("Dynamic","Cos",   "Single"): "record_fitness_cos_m7_single_platform_fixed_dispatch_0.mat",
    ("Dynamic","Cos",   "Multi"):  "record_fitness_cos_m7_multi_platform_fixed_dispatch_0.mat",
}

# Map scale to the indices 25/50/100 customers (Solomon files use suffixes).
SCALES = {"25": 25, "50": 50, "100": 100}


def load_cell(path: Path) -> tuple[np.ndarray, list[str]]:
    """Return (fitness array (12, 5, 1, 3), dataset names)."""
    m = sio.loadmat(path)
    key = next(k for k in m if k.startswith("record_fitness"))
    fit = m[key]                                     # (12, 5, 1, 3)
    datasets = [d[0] for d in m["datasets"][0]]      # list of 12 strings
    return fit, datasets


def aggregate_static(fit: np.ndarray) -> np.ndarray:
    """Static side: column 0 (lowest-temperature reference, 06:00).
    This matches metrics_summary.csv scenario 'static_lowest'."""
    return fit.squeeze(2)[:, :, 0]


def aggregate_dynamic(fit: np.ndarray) -> np.ndarray:
    """Dynamic side: column 0 (06:00 canonical dispatch)."""
    return fit.squeeze(2)[:, :, 0]


def instance_scale(name: str) -> int:
    # 'C101_25.txt' -> 25
    stem = name.rsplit(".", 1)[0]
    suffix = stem.split("_")[-1]
    return int(suffix)


def main() -> None:
    cells = {}
    datasets_ref = None
    for (mode, wx, plat), fname in FILES.items():
        path = DATA_DIR / fname
        if not path.exists():
            print(f"MISSING: {path}", file=sys.stderr)
            continue
        fit, ds = load_cell(path)
        datasets_ref = ds
        if mode == "Static":
            cells[(mode, wx, plat)] = aggregate_static(fit)
        else:
            cells[(mode, wx, plat)] = aggregate_dynamic(fit)

    # Pretty per-cell summary
    print("=" * 96)
    print("Per-instance mean ± SD over 5 reps (v2 with M7)")
    print("=" * 96)
    for wx in ("Real", "Cos"):
        for plat in ("Single", "Multi"):
            print(f"\n--- {wx} weather / {plat}-platform ---")
            print(f"  {'Instance':<14}  {'Static mean (SD)':<22}  {'Dyn mean (SD)':<22}  {'Δ% mean':>8}")
            s = cells[("Static", wx, plat)]   # (12, 5)
            d = cells[("Dynamic", wx, plat)]  # (12, 5)
            for i, name in enumerate(datasets_ref):
                sm, ss = s[i].mean(), s[i].std(ddof=1)
                dm, ds_sd = d[i].mean(), d[i].std(ddof=1)
                delta = (sm - dm) / sm * 100 if sm > 0 else 0
                print(f"  {name:<14}  {sm:>8.2f} ({ss:>5.2f})  {dm:>8.2f} ({ds_sd:>5.2f})  {delta:>7.2f}%")

    # Aggregate by scale (matches tab:main-results)
    print()
    print("=" * 96)
    print("Aggregated by Platform × Scale (matches tab:main-results structure)")
    print("=" * 96)
    print(f"\n  {'Platform':<10} {'Scale':<8} {'N':>3}   {'Static mean':>14} {'Static SD':>10}   {'Dyn mean':>14} {'Dyn SD':>10}   {'Impr%':>8}")
    for plat in ("Single", "Multi"):
        rows = []
        s_all = cells[("Static", "Real", plat)]
        d_all = cells[("Dynamic", "Real", plat)]
        for scale in (25, 50, 100):
            ix = [i for i, n in enumerate(datasets_ref) if instance_scale(n) == scale]
            # Aggregate across 4 instances × 5 reps = 20 values each
            s_vals = s_all[ix].flatten()
            d_vals = d_all[ix].flatten()
            sm, ss = s_vals.mean(), s_vals.std(ddof=1)
            dm, ds_sd = d_vals.mean(), d_vals.std(ddof=1)
            impr = (sm - dm) / sm * 100
            print(f"  {plat:<10} {scale:<8} {len(ix):>3}   {sm:>14.2f} {ss:>10.2f}   {dm:>14.2f} {ds_sd:>10.2f}   {impr:>7.2f}%")
            rows.append((scale, sm, ss, dm, ds_sd, impr))
        # All scales pooled
        s_all_flat = s_all.flatten()
        d_all_flat = d_all.flatten()
        sm = s_all_flat.mean(); ss = s_all_flat.std(ddof=1)
        dm = d_all_flat.mean(); ds_sd = d_all_flat.std(ddof=1)
        impr = (sm - dm) / sm * 100
        print(f"  {plat:<10} {'All':<8} {12:>3}   {sm:>14.2f} {ss:>10.2f}   {dm:>14.2f} {ds_sd:>10.2f}   {impr:>7.2f}%")

    # Print current manuscript best-of-5 numbers for comparison
    print()
    print("=" * 96)
    print("Current manuscript best-of-5 numbers (for reference):")
    print("  Real Single 25:  Static 2848.76, Dyn 2549.28, Impr 9.49%")
    print("  Real Single 50:  Static 5396.78, Dyn 4671.24, Impr 12.88%")
    print("  Real Single 100: Static 11717.72, Dyn 7597.86, Impr 35.74%")
    print("  Real Single All: Static 6654.42, Dyn 4939.46, Impr 19.37%")
    print("  Real Multi 25:   Static 2790.69, Dyn 2514.35, Impr 9.08%")
    print("  Real Multi 50:   Static 5035.52, Dyn 4451.31, Impr 11.48%")
    print("  Real Multi 100:  Static 8161.55, Dyn 7416.86, Impr 9.90%")
    print("  Real Multi All:  Static 5329.25, Dyn 4794.17, Impr 10.15%")


if __name__ == "__main__":
    main()
