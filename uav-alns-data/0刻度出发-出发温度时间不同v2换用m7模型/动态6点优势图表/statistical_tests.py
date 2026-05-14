#!/usr/bin/env python3
"""
Statistical tests on the 48 paired comparisons (Dynamic 06:00 vs Static low-T).

Produces:
  - Binomial sign test on the win rate (43/48 pooled; 24/24 real; 19/24 cosine).
  - Wilcoxon signed-rank test on the 48 paired improvement deltas.
  - 95% bootstrap CI on the mean improvement (paired bootstrap).

Output: prints a clean table that can be pasted directly into the manuscript.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from generate_dynamic_6am_vs_static_low_charts import load_pairs  # noqa: E402


def binomial(wins: int, n: int) -> tuple[float, float]:
    """Two-sided binomial test against p=0.5. Returns (p-value, win rate)."""
    p = stats.binomtest(wins, n, p=0.5, alternative="two-sided").pvalue
    return p, wins / n


def wilcoxon_signed_rank(deltas: list[float]) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank on improvement deltas (against null=0)."""
    res = stats.wilcoxon(deltas, alternative="two-sided", zero_method="wilcox")
    return res.statistic, res.pvalue


def bootstrap_mean_ci(values: list[float], n_boot: int = 20000,
                      ci: float = 0.95, seed: int = 20240101) -> tuple[float, float, float]:
    """Paired bootstrap 95% CI on the mean. Returns (mean, lo, hi)."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    boots = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return arr.mean(), lo, hi


def summarize(label: str, pairs: list[dict]) -> None:
    deltas = [float(p["improvement"]) for p in pairs]
    wins = sum(1 for d in deltas if d > 0)
    n = len(deltas)
    bp, win_rate = binomial(wins, n)
    wstat, wp = wilcoxon_signed_rank(deltas)
    mean, lo, hi = bootstrap_mean_ci(deltas)
    print(f"--- {label} ---")
    print(f"  N pairs               : {n}")
    print(f"  Wins (improvement>0)  : {wins} / {n} ({win_rate*100:.1f}%)")
    print(f"  Mean improvement      : {mean:.2f}%   (95% bootstrap CI: [{lo:.2f}%, {hi:.2f}%])")
    print(f"  Median improvement    : {np.median(deltas):.2f}%")
    print(f"  Binomial sign test    : two-sided p = {bp:.3e}")
    print(f"  Wilcoxon signed-rank  : W = {wstat:.2f}, two-sided p = {wp:.3e}")
    print()


def main() -> None:
    pairs = load_pairs()
    real   = [p for p in pairs if p["weather"] == "Real"]
    cosine = [p for p in pairs if p["weather"] == "Cos"]
    single = [p for p in pairs if p["platform"] == "Single"]
    multi  = [p for p in pairs if p["platform"] == "Multi"]
    real_single = [p for p in real if p["platform"] == "Single"]
    real_multi  = [p for p in real if p["platform"] == "Multi"]

    print("=" * 72)
    print("Statistical tests on Dynamic 06:00 vs Static low-T paired comparisons")
    print("=" * 72)
    summarize("Pooled (48 pairs)",       pairs)
    summarize("Real weather (24 pairs)", real)
    summarize("Synthetic cosine (24)",   cosine)
    summarize("Real, single-platform (12)", real_single)
    summarize("Real, multi-platform (12)",  real_multi)


if __name__ == "__main__":
    main()
