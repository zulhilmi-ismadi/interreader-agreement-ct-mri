"""The layer's headline totals, computed rather than read off a row count.

    python code/layer_totals.py

Hand counts are easily wrong in this layer: a count of (dataset, rater_id) combinations includes
intra-observer repeats, the row count of `per_image.csv` is not a count of annotated images, and every
pair in the layer includes LIDC's 76,896 sequential-read pairs. None of those is a count of annotators,
images or independent rater pairs.

Each total is reported two ways — everything in the layer, and only what enters a reportable floor —
because the paper needs both and must say which it is quoting.

Rater identifiers are per source and **cannot be deduplicated**: nothing links a person across
sources, and several sources (C3RO across its five sites, RIGA and QUBIQ per case) do not keep one
identifier per person even internally. So this reports identifiers, and says so.

Writes results/layer_totals.csv.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LAYER = ROOT / "results" / "layer"
OUT = ROOT / "results" / "layer_totals.csv"


def main() -> None:
    floors = pd.read_csv(LAYER / "task_floor.csv")
    pairs = pd.read_csv(LAYER / "pairs.csv", low_memory=False)
    per_image = pd.read_csv(LAYER / "per_image.csv", low_memory=False)

    column = "floor_reportable" if "floor_reportable" in floors else "floor_eligible"
    key = ["dataset", "task", "pair_class"]
    reportable = floors[floors[column] & (floors.metric == "dice")][key].drop_duplicates()

    in_floor = pairs.merge(reportable, on=key, how="inner")
    images_in_floor = per_image.merge(reportable, on=key, how="inner")

    # raters: P3 rows are the SAME person re-annotating, so their second identifier is not a person
    independent = in_floor[in_floor.pair_class.isin(["P1", "P2"])]
    raters = pd.concat(
        [independent[["dataset", "rater_a"]].rename(columns={"rater_a": "rater"}),
         independent[["dataset", "rater_b"]].rename(columns={"rater_b": "rater"})]
    ).drop_duplicates()

    rows = [
        ("datasets in the layer", pairs.dataset.nunique()),
        ("datasets with a reportable floor", reportable.dataset.nunique()),
        ("rater pairs, everything in the layer", len(pairs)),
        ("rater pairs entering a reportable floor", len(in_floor)),
        ("unique images, everything in the layer",
         per_image[["dataset", "image_id"]].drop_duplicates().shape[0]),
        ("unique images entering a reportable floor",
         images_in_floor[["dataset", "image_id"]].drop_duplicates().shape[0]),
        ("rater identifiers in reportable floors (P1/P2, per source, not deduplicable)", len(raters)),
    ]
    frame = pd.DataFrame(rows, columns=["quantity", "value"])
    frame.to_csv(OUT, index=False)

    for quantity, value in rows:
        print(f"  {quantity:78s} {value:>9,}")
    print("\n  rater identifiers per source (reportable floors):")
    for dataset, count in raters.groupby("dataset").size().sort_values(ascending=False).items():
        print(f"    {dataset:14s} {count}")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
