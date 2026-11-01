"""Share of reader pairs in which both readers drew the structure, for every structure.

    python code/both_drew_share.py

A pair in which both readers drew nothing scores Dice 1, and a pair in which exactly one drew nothing
scores Dice 0. `task_floor.csv` carries the share of pairs in which both drew something as
`frac_informative`. This script tabulates it for every expert (P1) and non-expert (P2) structure and
cross-checks it against the pair table.

Definitions used here:
  - both-empty pair: Dice 1 with a 95th percentile Hausdorff distance of 0;
  - one-empty pair: Dice 0 with an undefined Hausdorff distance;
  - share both drew = 1 - (both-empty + one-empty) / pairs.

Only task-level counts are written.

Reads results/layer/task_floor.csv and results/layer/pairs.parquet.
Writes results/manuscript/both_drew_share.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LAYER = ROOT / "results" / "layer"
OUT = ROOT / "results" / "manuscript" / "both_drew_share.csv"

CT_MRI = ("c3ro", "curvas", "pdacvi", "liverhccseg", "msseg2016", "qubiq21")


def main() -> None:
    floors = pd.read_csv(LAYER / "task_floor.csv", low_memory=False)
    floors = floors[(floors.metric == "dice") & floors.pair_class.isin(["P1", "P2"])]
    pairs = pd.read_parquet(LAYER / "pairs.parquet", columns=["dataset", "task", "pair_class", "dice", "hd95"])
    pairs = pairs[pairs.pair_class.isin(["P1", "P2"])]
    counts = pairs.assign(both_empty=(pairs.dice == 1) & (pairs.hd95 == 0),
                          one_empty=(pairs.dice == 0) & pairs.hd95.isna()).groupby(
        ["dataset", "task", "pair_class"]).agg(n_pairs_table=("dice", "size"), both_empty=("both_empty", "sum"),
                                              one_empty=("one_empty", "sum")).reset_index()
    table = floors[["dataset", "task", "pair_class", "modality", "n_raters", "n_images", "n_pairs",
                    "frac_informative", "mean"]].merge(counts, on=["dataset", "task", "pair_class"], how="left")
    table["both_drew_share_from_pairs"] = 1 - (table.both_empty + table.one_empty) / table.n_pairs_table
    table["group"] = np.where(table.dataset.isin(CT_MRI), "CT and MRI", "comparison set")
    table = table.rename(columns={"frac_informative": "both_drew_share", "mean": "dice"})
    table = table.sort_values(["group", "dataset", "task", "pair_class"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    mismatch = int(((table.both_drew_share - table.both_drew_share_from_pairs).abs() > 1e-6).sum())
    print(f"{len(table)} structures; {mismatch} where the layer share and the pair-table share differ")
    print(table[table.both_drew_share < 1 - 1e-9][["dataset", "task", "pair_class", "n_pairs", "both_empty",
                                                   "one_empty", "both_drew_share", "dice"]].to_string(index=False))
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
