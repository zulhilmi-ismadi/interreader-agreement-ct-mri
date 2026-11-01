"""The Cell Tracking Challenge leaderboard set beside the reference-substitution spread of its annotators.

Main quantity: the fraction of adjacent-rank gaps on a published leaderboard that are smaller than the
reference-substitution spread, i.e. gaps that the choice of reference annotator alone could have
produced. Also reported: the same over all method pairs, not just adjacent ones.

CTC is a `published` source: its per-annotator masks were never released, so the spread comes from the
published per-annotator values in Maska 2023 (per dataset and video, the range across the three
annotators) rather than from masks.

Two caveats are carried with every number this produces:
  * The published values are annotator-vs-majority-gold, and that gold was voted from the same three
    annotators, so this spread is OPTIMISTIC relative to a true pairwise floor.
  * The finest resampling unit available is the dataset; per-case algorithm scores are not public, so
    no per-case rank-stability claim can be made from this source.

Writes results/wp_l_ctc_per_dataset.csv and results/wp_l_ctc_summary.csv.

Usage: python code/leaderboard_ctc.py
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
DERIVED = ROOT / "data/derived"
RESULTS = ROOT / "results"

ANNOTATORS = ("Annotator 1", "Annotator 2", "Annotator 3")

#: The leaderboard metric must match the metric the spread is built from.
#: SEG is primary, DET secondary; no cross-metric comparison is permitted.
MEASURE_PAIRS = {"SEG": "MSEG_GT", "DET": "MDET_GT"}

LEADERBOARD_BLOCK = "per dataset: average DET and SEG values over videos 01 and 02"


def reference_substitution_spread() -> pd.DataFrame:
    """Per dataset and video, the range across the three annotators; then the mean
    over videos. Rows for 'Mean' and 'STD' in the source table are summaries, not raters."""
    table = pd.read_csv(DERIVED / "ctc_maska2023_annotator_measures.csv")
    table = table[table.rater.isin(ANNOTATORS)].dropna(subset=["value"])

    per_video = (
        table.groupby(["measure", "dataset", "video"])
        .agg(spread=("value", lambda v: v.max() - v.min()), n_raters=("value", "size"))
        .reset_index()
    )
    incomplete = per_video[per_video.n_raters != len(ANNOTATORS)]
    if not incomplete.empty:
        print(f"  note: {len(incomplete)} dataset-video cells lack all three annotators, excluded")
        per_video = per_video[per_video.n_raters == len(ANNOTATORS)]

    return (
        per_video.groupby(["measure", "dataset"])
        .agg(rss=("spread", "mean"), n_videos=("spread", "size"))
        .reset_index()
    )


def leaderboard() -> pd.DataFrame:
    table = pd.read_csv(DERIVED / "ctc_maska2023_leaderboard_csb.csv")
    table = table[table.block == LEADERBOARD_BLOCK]
    table["value"] = pd.to_numeric(table.value, errors="coerce")
    return table.dropna(subset=["value"])


def _gap_fractions(scores: np.ndarray, rss: float) -> tuple[float, float, int, int]:
    """Adjacent and all-pairs gap fractions, with scores sorted in the published direction (higher
    SEG/DET is better); a gap counts as inside when it is smaller than the spread."""
    ordered = np.sort(scores)[::-1]
    adjacent = np.abs(np.diff(ordered))
    pairs = np.array([abs(a - b) for a, b in itertools.combinations(ordered, 2)])
    return (
        float((adjacent < rss).mean()),
        float((pairs < rss).mean()),
        int(adjacent.size),
        int(pairs.size),
    )


def compute() -> tuple[pd.DataFrame, pd.DataFrame]:
    spreads = reference_substitution_spread()
    board = leaderboard()

    rows = []
    for measure, annotator_measure in MEASURE_PAIRS.items():
        measure_spreads = spreads[spreads.measure == annotator_measure].set_index("dataset")
        measure_board = board[board.measure == measure]
        for dataset, group in measure_board.groupby("dataset"):
            if dataset not in measure_spreads.index:
                continue  # no published annotator values for this dataset (the SIM sets)
            rss = float(measure_spreads.loc[dataset, "rss"])
            e2, e6, n_adjacent, n_pairs = _gap_fractions(group.value.to_numpy(float), rss)
            rows.append(
                {
                    "leaderboard": "CTC-CSB",
                    "measure": measure,
                    "dataset": dataset,
                    "rss": rss,
                    "n_methods": int(group.method.nunique()),
                    "n_adjacent_gaps": n_adjacent,
                    "n_all_pairs": n_pairs,
                    "e2_adjacent_inside": e2,
                    "e6_all_pairs_inside": e6,
                    "score_max": float(group.value.max()),
                    "score_min": float(group.value.min()),
                }
            )
    per_dataset = pd.DataFrame(rows)

    summaries = []
    for measure, group in per_dataset.groupby("measure"):
        # A dataset whose three annotators all score 1.0 has a band of exactly 0, so no gap can fall
        # inside it and its adjacent-gap fraction is 0 by construction rather than by measurement
        # (Fluo-C3DH-A549, DET).
        # Pooling it in pulls the mean down silently; exclude it and report that it was excluded.
        degenerate = int((group.rss <= 0).sum())
        group = group[group.rss > 0]
        e2 = group.e2_adjacent_inside.to_numpy(float)
        e6 = group.e6_all_pairs_inside.to_numpy(float)
        e2_low, e2_high = metrics.bootstrap_ci(e2)
        e6_low, e6_high = metrics.bootstrap_ci(e6)
        summaries.append(
            {
                "leaderboard": "CTC-CSB",
                "measure": measure,
                "n_datasets": len(group),
                "n_degenerate_excluded": degenerate,
                "e2_pooled": e2.mean(),
                "e2_ci_low": e2_low,
                "e2_ci_high": e2_high,
                "e6_pooled": e6.mean(),
                "e6_ci_low": e6_low,
                "e6_ci_high": e6_high,
                "mean_rss": group.rss.mean(),
                "resampling_unit": "dataset",
                "spread_construct": "annotator-vs-majority-gold (optimistic: each annotator against a majority reference)",
            }
        )
    return per_dataset, pd.DataFrame(summaries)


def main() -> None:
    per_dataset, summary = compute()
    RESULTS.mkdir(exist_ok=True)
    per_dataset.to_csv(RESULTS / "wp_l_ctc_per_dataset.csv", index=False)
    summary.to_csv(RESULTS / "wp_l_ctc_summary.csv", index=False)

    for _, row in summary.iterrows():
        print(
            f"\n{row.measure}: {row.e2_pooled:.1%} of adjacent rank gaps lie inside the annotator "
            f"spread\n   95% CI [{row.e2_ci_low:.1%}, {row.e2_ci_high:.1%}] over "
            f"{row.n_datasets} datasets; mean spread {row.mean_rss:.4f}\n"
            f"   all method pairs: {row.e6_pooled:.1%} "
            f"[{row.e6_ci_low:.1%}, {row.e6_ci_high:.1%}]"
        )

    primary = per_dataset[per_dataset.measure == "SEG"].sort_values("e2_adjacent_inside")
    print("\nSEG per dataset (lowest to highest fraction inside):")
    for _, row in primary.iterrows():
        print(
            f"  {row.dataset:22s} methods={row.n_methods:3d}  spread={row.rss:.4f}  "
            f"inside={row.e2_adjacent_inside:6.1%}"
        )

    print(
        "\nCaveats carried with these numbers:\n"
        "  - the spread is annotator-vs-majority-gold, voted from the same three annotators, so it\n"
        "    is optimistic relative to a true pairwise floor;\n"
        "  - the finest resampling unit is the dataset, so no per-case rank-stability claim follows."
    )


if __name__ == "__main__":
    main()
