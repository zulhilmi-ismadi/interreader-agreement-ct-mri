"""S5 — drop-any-one-rater sensitivity, for every task with at least three raters.

The question S5 exists to answer: how much of a task's floor is carried by one rater? A floor that
moves sharply when a single annotator is removed is reporting that annotator as much as the task.

This reads `results/layer/pairs.csv` and needs no recomputation: dropping a rater simply means
excluding the pairs they appear in, and the leave-one-out floor is the mean over what remains.

    python code/sensitivity_s5.py                 # every source
    python code/sensitivity_s5.py gleason2019     # one source

Writes `results/s5_drop_one_rater.csv` (every task x rater x metric) and prints the largest movers.

Note on the statistic: this compares **means over pairs**, which is the floor's point estimate. It
does not re-run the cluster bootstrap, so the deltas here carry no CI and are a screening tool, not
a replacement for the task_floor table.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LAYER = ROOT / "results" / "layer"
OUT = ROOT / "results" / "s5_drop_one_rater.csv"

#: the metric printed below; every other agreement metric is kept in the CSV
HEADLINE = "dice"
MIN_RATERS = 3


def drop_one_rater(pairs: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        c
        for c in pairs.columns
        if c
        not in {
            "dataset", "task", "image_id", "rater_a", "rater_b",
            "pair_class", "cluster_id", "source_run", "distance_unit",
        }
    ]
    rows = []
    for (dataset, task, pair_class), block in pairs.groupby(["dataset", "task", "pair_class"]):
        raters = sorted(set(block.rater_a) | set(block.rater_b))
        if len(raters) < MIN_RATERS:
            continue  # with two raters, dropping one leaves no pair at all
        # Image-then-task, exactly as the layer builds a floor. A plain mean
        # over pooled pairs weights each image by its pair count, which would make
        # these deltas incomparable with the floors they are quoted against.
        full = block.groupby("image_id")[metrics].mean().mean()
        for rater in raters:
            kept = block[(block.rater_a != rater) & (block.rater_b != rater)]
            if kept.empty:
                continue
            without = kept.groupby("image_id")[metrics].mean().mean()
            for metric in metrics:
                if pd.isna(full[metric]) or pd.isna(without[metric]):
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "task": task,
                        "pair_class": pair_class,
                        "dropped_rater": rater,
                        "metric": metric,
                        "floor_all_raters": full[metric],
                        "floor_without": without[metric],
                        "delta": without[metric] - full[metric],
                        "n_raters": len(raters),
                        "n_pairs_all": len(block),
                        "n_pairs_kept": len(kept),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    pairs = pd.read_csv(LAYER / "pairs.csv")
    wanted = [a for a in sys.argv[1:] if not a.startswith("-")]
    if wanted:
        pairs = pairs[pairs.dataset.isin(wanted)]
        if pairs.empty:
            raise SystemExit(f"no pairs for {wanted}; datasets present: {sorted(pairs.dataset.unique())}")

    frame = drop_one_rater(pairs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(ROOT)}  {len(frame)} rows")

    headline = frame[frame.metric == HEADLINE].copy()
    if headline.empty:
        return
    print(f"\nlargest movers on {HEADLINE} (a positive delta means the floor RISES without that rater):")
    headline["abs_delta"] = headline.delta.abs()
    for _, row in headline.sort_values("abs_delta", ascending=False).head(12).iterrows():
        print(
            f"  {row.dataset:12s} {row.task:16s} drop {row.dropped_rater:4s}  "
            f"{row.floor_all_raters:.3f} -> {row.floor_without:.3f}  ({row.delta:+.3f})"
        )

    print(f"\nper source, the single largest |delta| on {HEADLINE}:")
    worst = headline.loc[headline.groupby("dataset").abs_delta.idxmax()]
    for _, row in worst.sort_values("abs_delta", ascending=False).iterrows():
        print(
            f"  {row.dataset:12s} {row.n_raters} raters  worst = drop {row.dropped_rater:4s} "
            f"({row.delta:+.3f} on {row.task})"
        )


if __name__ == "__main__":
    main()
