"""Held-out reader against a consensus of the other readers, summarised over structures.

    python code/consensus_comparator_summary.py

For every structure with three or more expert readers, each reader in turn is held out and scored
against a consensus of the others, built by STAPLE and by majority vote (`report_s1_s3_e4.py`).

Definitions used here:
  - delta: the held-out reader's Dice against the consensus minus pairwise Dice between readers on the
    same images (positive = the consensus is the easier target);
  - structures: independent expert readers (P1) with pixel outlines;
  - counts are of structures with delta > 0; medians and interquartile ranges are over structures.

Reads results/s1_e4_view_a_vs_b.csv and results/layer/task_floor.csv (outline type).
Writes results/manuscript/consensus_comparator_summary.csv.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "manuscript" / "consensus_comparator_summary.csv"

CT_MRI = ("c3ro", "curvas", "pdacvi", "liverhccseg", "msseg2016", "qubiq21")


def summarise(df: pd.DataFrame, label: str) -> dict:
    row = {"group": label, "n_structures": len(df)}
    for method in ("staple", "majority"):
        delta = df[f"delta_{method}_dice"]
        row[f"{method}_consensus_closer_n"] = int((delta > 0).sum())
        row[f"{method}_median_delta"] = float(delta.median())
        row[f"{method}_iqr_low"] = float(delta.quantile(0.25))
        row[f"{method}_iqr_high"] = float(delta.quantile(0.75))
    return row


def main() -> None:
    table = pd.read_csv(RESULTS / "s1_e4_view_a_vs_b.csv")
    floors = pd.read_csv(RESULTS / "layer" / "task_floor.csv", low_memory=False)
    types = floors[floors.metric == "dice"][["dataset", "task", "pair_class", "annotation_type"]].drop_duplicates()
    table = table.merge(types, on=["dataset", "task", "pair_class"], how="left")
    pixel = table[(table.pair_class == "P1") & (table.annotation_type == "pixel")]
    rows = [summarise(pixel, "all pixel structures with three or more expert readers"),
            summarise(pixel[pixel.dataset.isin(CT_MRI)], "CT and MRI"),
            summarise(pixel[~pixel.dataset.isin(CT_MRI)], "comparison set")]
    summary = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT, index=False)
    print(summary.round(4).to_string(index=False))
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
