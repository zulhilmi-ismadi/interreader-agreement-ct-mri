"""Within-image dispersion of rater-pair agreement: the scale on which a method difference is read.

A difference between two methods is read against how much the reference itself varies, and the
standard deviation of rater-pair scores (quoted, for example, for the CURVAS pancreas) is that scale.
This script defines and computes it, and the release ships it as `pair_dispersion.csv`.

**Definition.** For task t, rater class c and image i with at least two rater pairs, let s_i be the
standard deviation (ddof=1) over the pairs of image i of the metric. The dispersion of the task is the
mean of s_i over the images that have at least two pairs:

    D_t = mean_i sd({ m(a, b, i) : pairs (a, b) on image i })

It is a *within-image* quantity on purpose: it measures how far two raters of the same image can sit
from each other, which is the scale a method's score against one rater is read on. It is not the
interval of the floor, which measures how precisely the mean agreement is known and narrows as images
are added, and it is not the reference-substitution spread `rss`, which is a range over
reference raters rather than a spread over pairs.

**Where it is undefined.** A two-rater source has one pair per image, so there is no within-image
spread; those tasks are reported as `n_images_used = 0` and a NaN dispersion rather than being
silently dropped. For them the across-image distribution of the single pair score is the honest
substitute, and it is reported here as `sd_across_images`.

    python code/pair_dispersion.py

Reads results/layer/pairs.csv; writes results/pair_dispersion.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ROOT / "results" / "layer" / "pairs.csv"
OUT = ROOT / "results" / "pair_dispersion.csv"
#: metrics the dispersion is reported for; the first is the one quoted in the paper
METRICS = ("dice", "nsd@2")


def dispersion(pairs: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for (dataset, task, pair_class), block in pairs.groupby(["dataset", "task", "pair_class"]):
        values = block[metric]
        if values.notna().sum() == 0:
            continue
        per_image = block.groupby("image_id")[metric]
        sizes = per_image.size()
        within = per_image.std(ddof=1)  # NaN where an image carries a single pair
        used = within.notna()
        image_means = per_image.mean()
        rows.append(
            {
                "dataset": dataset,
                "task": task,
                "pair_class": pair_class,
                "metric": metric,
                "within_image_sd": round(float(within[used].mean()), 4) if used.any() else np.nan,
                "n_images_used": int(used.sum()),
                "n_images": int(len(sizes)),
                "max_pairs_per_image": int(sizes.max()),
                "sd_across_images": round(float(image_means.std(ddof=1)), 4) if len(image_means) > 1 else np.nan,
                "definition": "mean over images of the sd across rater pairs within an image",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    pairs = pd.read_csv(PAIRS, low_memory=False)
    table = pd.concat([dispersion(pairs, metric) for metric in METRICS], ignore_index=True)
    table = table.sort_values(["metric", "dataset", "task", "pair_class"])
    table.to_csv(OUT, index=False)

    dice = table[(table.metric == "dice") & table.within_image_sd.notna()]
    print(f"{len(table)} rows; within-image dispersion defined for {len(dice)} task/class rows in Dice")
    print("\nsources with three or more raters on an image (dispersion defined):")
    print(
        dice.sort_values("within_image_sd", ascending=False)
        [["dataset", "task", "pair_class", "within_image_sd", "n_images_used", "max_pairs_per_image"]]
        .to_string(index=False)
    )
    two_rater = table[(table.metric == "dice") & table.within_image_sd.isna()]
    print(f"\ntwo-rater tasks, no within-image spread (sd_across_images reported instead): {len(two_rater)}")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
