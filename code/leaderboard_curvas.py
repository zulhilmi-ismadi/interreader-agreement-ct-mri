"""The CURVAS leaderboard set beside the reference-substitution spread of its own raters.

Same quantity as `leaderboard_ctc.py` (the fraction of adjacent-rank gaps smaller than the
reference-substitution spread), but from the other kind of source: CURVAS releases its per-rater masks,
so the spread is **recomputed from the masks** (each rater scored against every other rater as the
reference, the range taken, then averaged) rather than transcribed from published annotator values as
CTC's must be. Having one pairing of each kind is the point — it
shows the layer works whether or not a benchmark released its annotations.

Three caveats travel with every number here, and all three are structural:

  1. **The band covers the scored cases and 30 more.** The leaderboard scores 60 cases of the 65-case
     *test* split. Those annotations were withheld while the challenge ran and released with the rest
     once it closed, so the 90 public cases used here contain all 60 scored cases, plus the other 5
     testing-split cases, 20 training and 5 validation cases (counted in the released files). This is
     therefore not a transfer to an unseen split; the band is simply not restricted to the scored cases.

  2. **The leaderboard aggregates over organs, so the spread must too.** Published DSC is one number
     per team, averaged over pancreas, liver and kidney — and those three organs have very different
     floors. Comparing a per-organ spread against an organ-averaged gap would be a category error, so
     the spread here is built the same way the leaderboard is: per case, average the pairwise rater
     Dice over the three organs first, then take the reference-substitution spread of that.

  3. **No per-case algorithm scores are published, so no rank-stability claim follows.** A
     challengeR-style bootstrap of Kendall's tau needs per-case scores; CURVAS publishes aggregate means
     only. This is the same limitation CTC has, for the same reason, and it caps what this comparison
     can assert.

Leaderboard values: `data/derived/curvas_riera2025_leaderboard.csv`, transcribed from Riera-Marin et
al. 2025, "CURVAS challenge results", Comput Biol Med 197:111024 (arXiv:2505.08685), which reports 7
teams / 10 submissions over 60 test cases. Their published rank is **multi-metric** (DSC, calibration,
CRPS), which is why BreizhSeg ranks above DLAI on a lower DSC; we re-sort by DSC, because a DSC gap
may only be compared against a DSC-derived spread.

Writes results/wp_l_curvas_summary.csv and results/wp_l_curvas_gaps.csv.

Usage: python code/leaderboard_curvas.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gtu import metrics

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
LEADERBOARD = ROOT / "data/derived/curvas_riera2025_leaderboard.csv"

DATASET = "curvas"
METRIC = "dice"  # the leaderboard metric is DSC; a gap is never compared with a spread on another metric


def organ_averaged_pairs() -> pd.DataFrame:
    """Per (image, rater pair), the Dice averaged over the three organs — caveat 2."""
    pairs = pd.read_csv(RESULTS / DATASET / "pairs.csv")
    pairs = pairs[["image_id", "task", "rater_a", "rater_b", METRIC]].dropna(subset=[METRIC])
    averaged = (
        pairs.groupby(["image_id", "rater_a", "rater_b"])[METRIC]
        .mean()
        .reset_index()
        .rename(columns={METRIC: "dice_organ_mean"})
    )
    n_organs = pairs.task.nunique()
    complete = pairs.groupby(["image_id", "rater_a", "rater_b"])[METRIC].size()
    if not (complete == n_organs).all():
        incomplete = int((complete != n_organs).sum())
        print(f"  note: {incomplete} rater pairs lack all {n_organs} organs; averaged over what exists")
    return averaged


def reference_substitution_spread(averaged: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """RSS per image on the organ-averaged score, then the mean over images."""
    rows = []
    for image_id, block in averaged.groupby("image_id"):
        scores: dict[tuple[str, str], float] = {}
        for _, row in block.iterrows():
            scores[(row.rater_a, row.rater_b)] = row.dice_organ_mean
            scores[(row.rater_b, row.rater_a)] = row.dice_organ_mean  # RSS reads directed pairs
        value = metrics.reference_substitution_spread(scores)
        rows.append({"image_id": image_id, "rss": value, "n_pairs": len(block)})
    frame = pd.DataFrame(rows)
    return float(np.nanmean(frame.rss)), frame


#: the layer's bootstrap: B = 1,000, percentile 95% CI, this seed
BOOTSTRAP_B = 1000
SEED = 20260903


def bootstrap_endpoints(
    per_image: pd.DataFrame, scores: np.ndarray
) -> dict[str, tuple[float, float]]:
    """Resample IMAGES, recompute the band, recompute the adjacent-gap and all-pairs fractions.

    The uncertainty here is the **band's**, not the leaderboard's: the published method scores are
    fixed point estimates with no per-case data behind them, so nothing about them can be resampled.
    The interval therefore answers "how much does the conclusion move if the annotator sample had
    been different", which is the question this pairing can honestly ask, and no more.
    """
    rng = np.random.default_rng(SEED)
    values = per_image.rss.to_numpy(float)
    values = values[np.isfinite(values)]
    adjacent = np.abs(np.diff(np.sort(scores)[::-1]))
    all_pairs = np.array([abs(a - b) for a, b in itertools.combinations(scores, 2)])

    e2s, e6s, bands = [], [], []
    for _ in range(BOOTSTRAP_B):
        sample = rng.choice(values, size=len(values), replace=True)
        band = float(np.mean(sample))
        bands.append(band)
        e2s.append(float(np.mean(adjacent < band)))
        e6s.append(float(np.mean(all_pairs < band)))

    def percentile(values_: list[float]) -> tuple[float, float]:
        return float(np.percentile(values_, 2.5)), float(np.percentile(values_, 97.5))

    return {"e2": percentile(e2s), "e6": percentile(e6s), "band": percentile(bands)}


def gap_table(scores: np.ndarray, teams: list[str], rss: float) -> pd.DataFrame:
    order = np.argsort(-scores)
    ranked_scores, ranked_teams = scores[order], [teams[i] for i in order]
    rows = []
    for i in range(len(ranked_scores) - 1):
        gap = ranked_scores[i] - ranked_scores[i + 1]
        rows.append(
            {
                "rank_above": i + 1,
                "team_above": ranked_teams[i],
                "team_below": ranked_teams[i + 1],
                "dsc_above": ranked_scores[i],
                "dsc_below": ranked_scores[i + 1],
                "gap": gap,
                "rss": rss,
                "inside_band": bool(gap < rss),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    board = pd.read_csv(LEADERBOARD)
    averaged = organ_averaged_pairs()
    rss, per_image = reference_substitution_spread(averaged)

    # the leaderboard is in percent; the layer is in [0, 1]
    scores = board.dsc_percent.to_numpy(float) / 100.0
    gaps = gap_table(scores, list(board.team), rss)

    all_pairs = [abs(a - b) for a, b in itertools.combinations(scores, 2)]
    e2 = float(gaps.inside_band.mean())
    e6 = float(np.mean([g < rss for g in all_pairs]))
    ci = bootstrap_endpoints(per_image, scores)

    floor = float(averaged.dice_organ_mean.mean())
    print(f"CURVAS pairing — {len(board)} teams, {len(all_pairs)} method pairs, "
          f"{averaged.image_id.nunique()} annotated cases")
    print(f"  organ-averaged inter-rater Dice (the floor): {floor:.4f}")
    print(f"  reference-substitution spread (the band): {rss:.4f}")
    print(f"  band 95% CI (image bootstrap, B={BOOTSTRAP_B}): "
          f"{ci['band'][0]:.4f}–{ci['band'][1]:.4f}")
    print(f"  adjacent rank gaps inside the band: {e2:.1%}  ({int(gaps.inside_band.sum())}/{len(gaps)})"
          f"  95% CI [{ci['e2'][0]:.1%}, {ci['e2'][1]:.1%}]")
    print(f"  all method pairs inside the band:     {e6:.1%}  "
          f"({int(np.sum([g < rss for g in all_pairs]))}/{len(all_pairs)})"
          f"  95% CI [{ci['e6'][0]:.1%}, {ci['e6'][1]:.1%}]")
    print("\n  adjacent gaps, largest first:")
    for _, row in gaps.sort_values("gap", ascending=False).iterrows():
        mark = "INSIDE " if row.inside_band else "outside"
        print(f"    {mark} {row.team_above:11s} {row.dsc_above:.4f} -> "
              f"{row.team_below:11s} {row.dsc_below:.4f}   gap {row.gap:.4f}")

    summary = pd.DataFrame(
        [
            {
                "dataset": DATASET,
                "metric": METRIC,
                "n_teams": len(board),
                "n_adjacent_gaps": len(gaps),
                "n_method_pairs": len(all_pairs),
                "n_annotated_cases": averaged.image_id.nunique(),
                "floor_organ_averaged": floor,
                "rss_band": rss,
                "e2_adjacent_inside": e2,
                "e2_ci_low": ci["e2"][0],
                "e2_ci_high": ci["e2"][1],
                "e6_all_pairs_inside": e6,
                "e6_ci_low": ci["e6"][0],
                "e6_ci_high": ci["e6"][1],
                "band_ci_low": ci["band"][0],
                "band_ci_high": ci["band"][1],
                "resampling_unit": "image (band only; published method scores are fixed)",
                "band_source": "recomputed from masks",
                "caveat_split": "band from the 90 public cases, which include the 60 test cases the "
                                "leaderboard scores",
                "caveat_aggregation": "organ-averaged to match the leaderboard's aggregation",
                "caveat_rank_stability": "no per-case algorithm scores published; no tau claim possible",
            }
        ]
    )
    summary.to_csv(RESULTS / "wp_l_curvas_summary.csv", index=False)
    gaps.to_csv(RESULTS / "wp_l_curvas_gaps.csv", index=False)
    per_image.to_csv(RESULTS / "wp_l_curvas_rss_per_image.csv", index=False)
    print(f"\nwrote results/wp_l_curvas_{{summary,gaps,rss_per_image}}.csv")


if __name__ == "__main__":
    main()
