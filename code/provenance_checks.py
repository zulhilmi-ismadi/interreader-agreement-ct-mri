"""Provenance and reproduction checks for two comparison-set sources.

Two values reported in the article come from here: Arvaniti's share of bit-identical mask pairs, and
the definitions tried against CytoCrowd's published pairwise IoU.

    python code/provenance_checks.py

Writes results/provenance_arvaniti.csv and results/e5_cytocrowd_iou.csv.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "code"))

from loaders.cytocrowd import parse_box  # noqa: E402  (after the path insert)

#: CytoCrowd's README, "Avg. Pairwise IoU"
CYTOCROWD_PUBLISHED_IOU = 0.664
THRESHOLDS = (0.0, 0.05, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5)


def arvaniti() -> pd.DataFrame:
    """Share of the 245 test cores on which the two pathologists' masks are bit identical. Adjudication
    of one pathologist's work by another would drive this towards 1; independent tracing keeps it low."""
    root = ROOT / "data/gtu-raw/arvaniti"
    cores = sorted(path.stem for path in (root / "r1").glob("*.png"))
    identical = 0
    for core in cores:
        first = np.array(Image.open(root / "r1" / f"{core}.png"))
        second = np.array(Image.open(root / "r2" / f"{core}.png"))
        identical += int(first.shape == second.shape and np.array_equal(first, second))
    frame = pd.DataFrame(
        [{"cores": len(cores), "bit_identical_mask_pairs": identical,
          "fraction_bit_identical": identical / len(cores)}]
    )
    frame.to_csv(RESULTS / "provenance_arvaniti.csv", index=False)
    print(f"Arvaniti: {identical} of {len(cores)} mask pairs bit identical "
          f"({identical / len(cores):.1%})")
    return frame


def _iou_matrix(first: list, second: list) -> np.ndarray:
    a, b = np.array(first, float), np.array(second, float)
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def cytocrowd() -> pd.DataFrame:
    """Six natural readings of 'average pairwise IoU', then a sweep of the matching threshold. The
    published matching rule is not stated, so the aim is to bracket the published figure, not to
    choose a definition after the fact that happens to land on it."""
    workers = pd.read_csv(ROOT / "data/CytoCrowd/WorkerData.csv")
    workers["box"] = [parse_box(value) for value in workers.worker_bbox]
    rater_of = {aid: f"r{i}" for i, aid in enumerate(sorted(workers.author_id.unique()), start=1)}
    workers["rater"] = workers.author_id.map(rater_of)
    truth = pd.read_csv(ROOT / "data/CytoCrowd/GroundTruth.csv")
    truth["box"] = [parse_box(value) for value in truth.bbox]

    hungarian, best_first, symmetric, per_image, against_truth = [], [], [], [], []
    for image, block in workers.groupby("img_name"):
        by_rater = {rater: list(group.box) for rater, group in block.groupby("rater")}
        image_means = []
        for first, second in itertools.combinations(sorted(by_rater), 2):
            matrix = _iou_matrix(by_rater[first], by_rater[second])
            rows, cols = linear_sum_assignment(-matrix)
            matched = matrix[rows, cols]
            hungarian.append(matched)
            best_a, best_b = matrix.max(axis=1), matrix.max(axis=0)
            best_first.extend(best_a.tolist())
            symmetric.extend(np.concatenate([best_a, best_b]).tolist())
            if (matched > 0).any():
                image_means.append(float(matched[matched > 0].mean()))
        if image_means:
            per_image.append(float(np.mean(image_means)))
        reference = list(truth[truth.img_name == image].box)
        if reference:
            for boxes in by_rater.values():
                best = _iou_matrix(boxes, reference).max(axis=1)
                against_truth.extend(best[best > 0].tolist())

    matched_all = np.concatenate(hungarian)
    best_first, symmetric = np.array(best_first), np.array(symmetric)
    rows = [
        ("definition", "Hungarian matched pairs, IoU > 0", matched_all[matched_all > 0]),
        ("definition", "Hungarian matched pairs, IoU > 0.5", matched_all[matched_all > 0.5]),
        ("definition", "best match per box, unmatched excluded", best_first[best_first > 0]),
        ("definition", "best match both directions, unmatched excluded", symmetric[symmetric > 0]),
        ("definition", "per image mean of Hungarian matches, then mean", np.array(per_image)),
        ("definition", "each worker box against the ground truth, best match", np.array(against_truth)),
    ]
    rows += [
        ("threshold", f"Hungarian matched pairs, IoU > {threshold}", matched_all[matched_all > threshold])
        for threshold in THRESHOLDS
    ]
    frame = pd.DataFrame(
        [{"kind": kind, "reading": name, "value": float(values.mean()), "n": int(values.size),
          "published": CYTOCROWD_PUBLISHED_IOU,
          "difference": float(values.mean()) - CYTOCROWD_PUBLISHED_IOU}
         for kind, name, values in rows]
    )
    frame.to_csv(RESULTS / "e5_cytocrowd_iou.csv", index=False)

    print(f"\nCytoCrowd published average pairwise IoU {CYTOCROWD_PUBLISHED_IOU}")
    for kind in ("definition", "threshold"):
        block = frame[frame.kind == kind]
        print(f"  {kind}s tried: {len(block)}")
        for _, row in block.iterrows():
            print(f"    {row.reading:56s} {row.value:.4f}  ({row.difference:+.4f})")
    sweep = frame[frame.kind == "threshold"].reset_index(drop=True)
    below = sweep[sweep.value < CYTOCROWD_PUBLISHED_IOU]
    above = sweep[sweep.value > CYTOCROWD_PUBLISHED_IOU]
    if not below.empty and not above.empty:
        print(f"  the published value lies between {below.iloc[-1].reading.split('> ')[1]} "
              f"({below.iloc[-1].value:.4f}) and {above.iloc[0].reading.split('> ')[1]} "
              f"({above.iloc[0].value:.4f})")
    return frame


if __name__ == "__main__":
    arvaniti()
    cytocrowd()
    print("\nwrote results/provenance_arvaniti.csv, results/e5_cytocrowd_iou.csv")
