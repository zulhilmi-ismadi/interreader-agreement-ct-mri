"""Two-way (scan and reader) bootstrap intervals for multi-image structures with three or more readers.

    python code/two_way_bootstrap.py

The layer's intervals resample images (or patients) with the readers fixed. This script adds an
interval that resamples images and readers together.

Definitions used here:
  - agreement: per image, the mean Dice over the reader pairs present; then the mean over images;
  - image interval: 1000 resamples of the per-image values with replacement (seed 20260903), 95%
    percentile interval;
  - two-way interval: in each of 1000 resamples (seed 20260903, a fresh generator per structure),
    readers are drawn with replacement and images are drawn with replacement; pairs formed by two draws
    of the same reader are dropped, a reader drawn twice contributes each of its pairs twice, and the
    per-image mean over the drawn pairs is averaged over the drawn images. 95% percentile interval;
  - two-reader structures have one pair per image, so reader resampling does not apply and only the
    image interval is given;
  - QUBIQ 2021 label-set indices are assigned per image, so each index is treated as one reader.

Only task-level values are written (MSSEG-2016 and QUBIQ 2021 pair rows are read locally and never
written out).

Reads results/{curvas,pdacvi,liverhccseg,msseg2016,qubiq21}/pairs.csv.
Writes results/manuscript/two_way_bootstrap.csv.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "manuscript" / "two_way_bootstrap.csv"

DRAWS, SEED = 1000, 20260903
STRUCTURES = {"curvas": ["liver", "kidney", "pancreas"], "pdacvi": ["lesion"], "liverhccseg": ["liver", "tumor1"],
              "msseg2016": ["ms_lesion"],
              "qubiq21": ["brain-growth_task01", "kidney_task01", "prostate_task01", "prostate_task02"]}


def image_interval(df: pd.DataFrame, rng: np.random.Generator) -> tuple[float, float]:
    per_image = df.groupby("image_id").dice.mean().to_numpy(float)
    means = [rng.choice(per_image, size=per_image.size, replace=True).mean() for _ in range(DRAWS)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def two_way_interval(df: pd.DataFrame, rng: np.random.Generator) -> tuple[float, float]:
    readers = sorted(set(df.rater_a) | set(df.rater_b))
    images = sorted(df.image_id.unique())
    lookup = {(img, frozenset((a, b))): d for img, a, b, d in
              df[["image_id", "rater_a", "rater_b", "dice"]].itertuples(index=False)}
    means = []
    for _ in range(DRAWS):
        drawn_readers = [readers[i] for i in rng.integers(0, len(readers), len(readers))]
        drawn_images = [images[i] for i in rng.integers(0, len(images), len(images))]
        pairs = [frozenset((x, y)) for x, y in combinations(drawn_readers, 2) if x != y]
        per_image = []
        for img in drawn_images:
            values = [lookup[(img, pr)] for pr in pairs if (img, pr) in lookup]
            if values:
                per_image.append(np.mean(values))
        if per_image:
            means.append(np.mean(per_image))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    rows = []
    for dataset, tasks in STRUCTURES.items():
        pairs = pd.read_csv(RESULTS / dataset / "pairs.csv", low_memory=False)
        for task in tasks:
            df = pairs[(pairs.task == task) & pairs.dice.notna()]
            readers = sorted(set(df.rater_a) | set(df.rater_b))
            point = float(df.groupby("image_id").dice.mean().mean())
            img_lo, img_hi = image_interval(df, np.random.default_rng(SEED))
            if len(readers) >= 3:
                tw_lo, tw_hi = two_way_interval(df, np.random.default_rng(SEED))
            else:
                tw_lo, tw_hi = np.nan, np.nan
            rows.append({"dataset": dataset, "structure": task, "n_images": df.image_id.nunique(),
                         "n_readers": len(readers), "n_pairs": len(df), "dice": point,
                         "image_ci_low": img_lo, "image_ci_high": img_hi,
                         "two_way_ci_low": tw_lo, "two_way_ci_high": tw_hi,
                         "note": "" if len(readers) >= 3 else "two readers: reader resampling does not apply",
                         "draws": DRAWS, "seed": SEED})
            print(f"{dataset:12s} {task:20s} n={df.image_id.nunique():3d} readers={len(readers)} dice={point:.4f} "
                  f"image ({img_lo:.3f}, {img_hi:.3f}) two-way ({tw_lo:.3f}, {tw_hi:.3f})")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
