"""Density null for the adjacent-gap share: how much of it leaderboard density alone produces.

    python code/leaderboard_null.py

**The problem.** The adjacent-gap share (column `e2_*`) is the fraction of adjacent leaderboard gaps
smaller than the annotator band: entries sorted by the published metric, gap = score_i - score_{i+1}.
With n methods spread over a score range R, the mean adjacent gap is R/(n-1), so on a dense leaderboard
almost every gap is small *whatever the annotators did*. The share then measures how crowded a
leaderboard is, not how much its reference disagrees, and it cannot be interpreted without knowing how
much of it density alone produces.

**Three questions:**

  1. **What does density alone produce?** For each leaderboard, draw n scores uniformly over its own
     observed range and compute the share against the same band. The observed share minus this null is the part
     that density does not explain. Uniform is the conservative null here: real leaderboards cluster
     near the top, which makes their adjacent gaps *smaller* than uniform, so a real board can exceed
     the null through clustering alone. The excess is therefore an upper bound on what the band
     contributes, not a measurement of it.

  2. **How many ranks does the band span?** The band divided by the median adjacent gap, and the band
     as a fraction of the score range. These are density-aware by construction and need no null.

  3. **Can a rank flip be shown?** No, and this script states why rather than implying otherwise. A
     band is how much ONE rater's score moves when the reference changes; if every method moves
     together, no rank changes. Showing a flip needs each method scored against each individual
     rater, per case. Neither the Cell Tracking Challenge nor CURVAS publishes that.

Writes results/wp_l_null_per_board.csv and results/wp_l_null_summary.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

import leaderboard_ctc as ctc

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

#: the layer's bootstrap seed; more draws than its bootstrap because each is cheap and the null is a mean
SEED = 20260903
DRAWS = 10_000


def density_null(n: int, low: float, high: float, band: float, rng: np.random.Generator) -> float:
    """Expected adjacent-gap share if n scores were spread uniformly over [low, high], against the same band."""
    if n < 2 or band <= 0 or high <= low:
        return 0.0
    draws = np.sort(rng.uniform(low, high, size=(DRAWS, n)), axis=1)
    return float((np.diff(draws, axis=1) < band).mean())


def describe(board: str, measure: str, dataset: str, scores: np.ndarray, band: float,
             rng: np.random.Generator) -> dict:
    scores = np.sort(scores[np.isfinite(scores)])[::-1]
    n = scores.size
    adjacent = np.abs(np.diff(scores))
    low, high = float(scores.min()), float(scores.max())
    score_range = high - low
    median_gap = float(np.median(adjacent)) if adjacent.size else np.nan

    # a method's "indistinguishable set": other methods within the band of it
    within = np.array([int((np.abs(scores - s) < band).sum()) - 1 for s in scores])

    observed = float((adjacent < band).mean()) if adjacent.size else np.nan
    null = density_null(n, low, high, band, rng)
    return {
        "leaderboard": board,
        "measure": measure,
        "dataset": dataset,
        "n_methods": n,
        "score_range": score_range,
        "band": band,
        "e2_observed": observed,
        "e2_density_null": null,
        "e2_excess_over_null": observed - null,
        "band_over_median_gap": band / median_gap if median_gap and median_gap > 0 else np.nan,
        "band_over_range": band / score_range if score_range > 0 else np.nan,
        "median_indistinguishable": float(np.median(within)),
        "degenerate": bool(band <= 0),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = []

    spreads = ctc.reference_substitution_spread()
    board = ctc.leaderboard()
    for measure, annotator_measure in ctc.MEASURE_PAIRS.items():
        measure_spreads = spreads[spreads.measure == annotator_measure].set_index("dataset")
        for dataset, group in board[board.measure == measure].groupby("dataset"):
            if dataset not in measure_spreads.index:
                continue
            band = float(measure_spreads.loc[dataset, "rss"])
            rows.append(describe("CTC-CSB", measure, dataset, group.value.to_numpy(float), band, rng))

    curvas_board = pd.read_csv(ROOT / "data/derived/curvas_riera2025_leaderboard.csv")
    curvas_band = float(pd.read_csv(RESULTS / "wp_l_curvas_summary.csv").iloc[0].rss_band)
    rows.append(describe("CURVAS", "DSC", "curvas",
                         curvas_board.dsc_percent.to_numpy(float) / 100.0, curvas_band, rng))

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "wp_l_null_per_board.csv", index=False)

    summaries = []
    for (board_name, measure), block in frame.groupby(["leaderboard", "measure"], sort=False):
        usable = block[~block.degenerate]
        rho = spearmanr(usable.n_methods, usable.e2_observed).correlation if len(usable) > 2 else np.nan
        rho_null = spearmanr(usable.n_methods, usable.e2_density_null).correlation if len(usable) > 2 else np.nan
        summaries.append(
            {
                "leaderboard": board_name,
                "measure": measure,
                "n_boards": len(block),
                "n_degenerate_excluded": int(block.degenerate.sum()),
                "e2_observed": usable.e2_observed.mean(),
                "e2_density_null": usable.e2_density_null.mean(),
                "e2_excess_over_null": usable.e2_excess_over_null.mean(),
                "median_band_over_median_gap": usable.band_over_median_gap.median(),
                "median_band_over_range": usable.band_over_range.median(),
                "median_indistinguishable": usable.median_indistinguishable.median(),
                "spearman_e2_vs_n": rho,
                "spearman_null_vs_n": rho_null,
            }
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(RESULTS / "wp_l_null_summary.csv", index=False)

    print("Adjacent-gap share against a density null (datasets weighted equally; degenerate bands excluded)\n")
    for _, row in summary.iterrows():
        print(f"{row.leaderboard} {row.measure}  ({row.n_boards} boards, "
              f"{row.n_degenerate_excluded} degenerate excluded)")
        print(f"  share observed           {row.e2_observed:6.1%}")
        print(f"  share from density alone {row.e2_density_null:6.1%}")
        print(f"  excess over the null     {row.e2_excess_over_null:+6.1%}   <- upper bound on the band's contribution")
        print(f"  band spans               {row.median_band_over_median_gap:6.1f} median adjacent gaps")
        print(f"  band as share of range   {row.median_band_over_range:6.1%}")
        print(f"  a typical method is indistinguishable from {row.median_indistinguishable:.0f} others")
        if pd.notna(row.spearman_e2_vs_n):
            print(f"  Spearman rank correlation, share vs method count: observed {row.spearman_e2_vs_n:+.2f}, "
                  f"null {row.spearman_null_vs_n:+.2f}")
        print()
    print("Rank flips: not assessable. Neither leaderboard publishes per-case scores against each "
          "individual rater, which is what a flip requires.")
    print("\nwrote results/wp_l_null_per_board.csv, results/wp_l_null_summary.csv")


if __name__ == "__main__":
    main()
