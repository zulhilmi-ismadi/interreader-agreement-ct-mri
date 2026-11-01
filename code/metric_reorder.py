"""How far task floors move in rank between an overlap metric and a boundary metric.

    python code/metric_reorder.py

Every metric-reordering value comes from this script, computed on a set that excludes any task with no
informative pair.

Four constraints:

  * **Reportable floors only**, P1 only, so no tier mixing and no all-empty task (a task in which every pair is both-empty scores 1.0 by
    definition, with no rater judgement behind it).
  * **Pixel and instance floors only.** Box streams are summarised by box F1 and are never pooled
    with pixel floors, so they are not ranked.
  * **No single-pair floors as examples.** A task with one image and one pair has no interval; it can
    rank anywhere and prove nothing. They are ranked but never quoted.
  * **Two-dimensional and volumetric sources are ranked separately.** NSD@2 means 2 pixels for one and
    2 millimetres for the other. Ranking them together would let a pixel/millimetre
    difference masquerade as a difference in boundary agreement.

Writes results/r3_metric_reorder.csv.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parent.parent
LAYER = ROOT / "results" / "layer"
OUT = ROOT / "results" / "r3_metric_reorder.csv"

OVERLAP, BOUNDARY = "dice", "nsd@2"
VOLUMETRIC = {"ct", "mri"}
MIN_PAIRS_TO_QUOTE = 10


def main() -> None:
    floors = pd.read_csv(LAYER / "task_floor.csv")
    column = "floor_reportable" if "floor_reportable" in floors else "floor_eligible"
    rows = floors[floors[column] & (floors.pair_class == "P1") & floors.metric.isin([OVERLAP, BOUNDARY])]
    # box streams are summarised by box F1, never by Dice or NSD, so they do not enter the ranking
    rows = rows[rows.annotation_type != "box"]
    wide = rows.pivot_table(
        index=["dataset", "task", "modality"], columns="metric", values="mean"
    ).dropna().reset_index()
    pairs = rows[rows.metric == OVERLAP].set_index(["dataset", "task"])["n_pairs"]
    wide["n_pairs"] = [int(pairs.get((d, t), 0)) for d, t in zip(wide.dataset, wide.task)]
    wide["grid"] = wide.modality.map(lambda m: "volumetric (mm)" if m in VOLUMETRIC else "2D (px)")

    out = []
    for grid, block in wide.groupby("grid"):
        block = block.copy()
        block["rank_overlap"] = block[OVERLAP].rank(ascending=False, method="min").astype(int)
        block["rank_boundary"] = block[BOUNDARY].rank(ascending=False, method="min").astype(int)
        block["move"] = block.rank_boundary - block.rank_overlap
        block["n_in_grid"] = len(block)
        tau = kendalltau(block[OVERLAP], block[BOUNDARY]).correlation
        block["kendall_tau_grid"] = tau
        out.append(block)

        quotable = block[block.n_pairs >= MIN_PAIRS_TO_QUOTE]
        print(f"{grid}: {len(block)} reportable P1 task floors "
              f"({len(quotable)} with >= {MIN_PAIRS_TO_QUOTE} pairs, quotable)")
        print(f"  Kendall's tau, {OVERLAP} vs {BOUNDARY}: {tau:.3f}")
        print(f"  task floors changing rank: {(block.move != 0).mean():.0%}")
        for _, r in quotable.reindex(quotable.move.abs().sort_values(ascending=False).index).head(5).iterrows():
            print(f"    {r.dataset:12s} {r.task:24s} {OVERLAP} {r[OVERLAP]:.3f} (rank {r.rank_overlap:>2}) "
                  f"-> {BOUNDARY} {r[BOUNDARY]:.3f} (rank {r.rank_boundary:>2})  move {r.move:+d}  "
                  f"[{r.n_pairs} pairs]")
        print()

    frame = pd.concat(out, ignore_index=True)
    frame.to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
