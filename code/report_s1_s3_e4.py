"""Two secondary analyses: the consensus comparator (View B against View A) and the instance-matching
threshold (IoU 0.5 against 0.7).

Both are computed by `compute_layer.py`; this script only tabulates.

    python code/report_s1_s3_e4.py

Writes
  results/s1_e4_view_a_vs_b.csv  per task: View B (STAPLE, majority) against View A on the SAME images
  results/s3_instance_iou.csv    instance F1 at IoU 0.5 against 0.7, every instance and box task

View B exists only where at least three raters annotated an image, so comparing it with the
full View A floor would mix image sets. View A is therefore recomputed over exactly the images View B
covers, image-then-task as the floor is (mean over pairs per image, then mean over images). The 58
panels on which View B was not computed (57 above the panel-size cap, one on which STAPLE did not
terminate) are counted per task, not imputed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LAYER = ROOT / "results" / "layer"
RESULTS = ROOT / "results"
METRICS = ("dice", "nsd@2")


def s1_e4() -> pd.DataFrame:
    view_b = pd.read_csv(LAYER / "view_b.csv")
    pairs = pd.read_parquet(LAYER / "pairs.parquet")
    floors = pd.read_csv(LAYER / "task_floor.csv")
    skipped = pd.read_csv(LAYER / "view_b_skipped.csv")
    column = "floor_reportable" if "floor_reportable" in floors else "floor_eligible"
    reportable = floors[floors[column] & (floors.metric == "dice")]
    keep = set(zip(reportable.dataset, reportable.task, reportable.pair_class))
    full = reportable.set_index(["dataset", "task", "pair_class"])["mean"]
    skip_counts = skipped.groupby(["dataset", "task", "pair_class"]).size()

    rows = []
    for (dataset, task, pair_class), block in view_b.groupby(["dataset", "task", "pair_class"]):
        if (dataset, task, pair_class) not in keep:
            continue
        images = set(block.image_id)
        a = pairs[(pairs.dataset == dataset) & (pairs.task == task) & (pairs.pair_class == pair_class)
                  & pairs.image_id.isin(images)]
        row = {"dataset": dataset, "task": task, "pair_class": pair_class,
               "n_images_view_b": len(images),
               "n_skipped_panels": int(skip_counts.get((dataset, task, pair_class), 0)),
               "view_a_full_floor_dice": float(full[(dataset, task, pair_class)])}
        for metric in METRICS:
            view_a = a.groupby("image_id")[metric].mean().mean() if not a.empty else np.nan
            row[f"view_a_same_images_{metric}"] = view_a
            for consensus in ("staple", "majority"):
                part = block[block.consensus == consensus]
                value = part[metric].mean() if not part.empty else np.nan
                row[f"view_b_{consensus}_{metric}"] = value
                row[f"delta_{consensus}_{metric}"] = value - view_a
        rows.append(row)
    # Panels whose every image was skipped (e.g. NuCLS P2 at the cap) leave no View B rows at all,
    # so the loop above never sees them. List them too, or the skip count understates the 58.
    seen = {(r["dataset"], r["task"], r["pair_class"]) for r in rows}
    for (dataset, task, pair_class), count in skip_counts.items():
        if (dataset, task, pair_class) in seen:
            continue
        rows.append({"dataset": dataset, "task": task, "pair_class": pair_class,
                     "n_images_view_b": 0, "n_skipped_panels": int(count),
                     "view_a_full_floor_dice": float(full.get((dataset, task, pair_class), np.nan))})
    return pd.DataFrame(rows)


def s3() -> pd.DataFrame:
    floors = pd.read_csv(LAYER / "task_floor.csv")
    column = "floor_reportable" if "floor_reportable" in floors else "floor_eligible"
    part = floors[floors[column] & floors.annotation_type.isin(["instance", "box"])
                  & floors.metric.isin(["instance_f1@0.5", "instance_f1@0.7"])]
    wide = part.pivot_table(index=["dataset", "task", "pair_class", "annotation_type"],
                            columns="metric", values=["mean", "ci_low", "ci_high"])
    wide.columns = [f"{m}_{stat}" for stat, m in wide.columns]
    wide = wide.reset_index()
    wide["delta"] = wide["instance_f1@0.7_mean"] - wide["instance_f1@0.5_mean"]
    return wide


def main() -> None:
    one = s1_e4()
    one.to_csv(RESULTS / "s1_e4_view_a_vs_b.csv", index=False)
    print(f"consensus comparator: {len(one)} task floors with View B; "
          f"{int(one.n_skipped_panels.sum())} skipped panels counted")
    for consensus in ("staple", "majority"):
        d = one[f"delta_{consensus}_dice"]
        print(f"    View B ({consensus}) minus View A, same images, Dice: median {d.median():+.3f}, "
              f"range {d.min():+.3f} to {d.max():+.3f}; B above A on {(d > 0).sum()}/{d.notna().sum()}")
    three = s3()
    three.to_csv(RESULTS / "s3_instance_iou.csv", index=False)
    print(f"S3: {len(three)} instance and box task floors")
    for _, r in three.iterrows():
        print(f"    {r.dataset:10s} {r.task:11s} {r.pair_class}  F1@0.5 {r['instance_f1@0.5_mean']:.3f}  "
              f"F1@0.7 {r['instance_f1@0.7_mean']:.3f}  ({r.delta:+.3f})")
    print("\nwrote results/s1_e4_view_a_vs_b.csv, results/s3_instance_iou.csv")


if __name__ == "__main__":
    main()
