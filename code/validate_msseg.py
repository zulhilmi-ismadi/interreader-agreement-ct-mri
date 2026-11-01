"""Reproduction of the published MSSEG-2016 agreement: per-expert Dice against a leave-one-out consensus.

Commowick et al. (2021, section 4.2) score each expert, on each patient, against a consensus of the
other six built by LOP-STAPLE, and report that the per-expert median Dice "varies between 0.66 and 0.76".
View B keeps only the mean over held-out raters per image, so it cannot give a per-expert median; this
script recomputes the per-(patient, expert) scores it averages.

  consensus = staple | majority   leave-one-out consensus of the other six (View B's constructions; the
                                  source's LOP-STAPLE is not implemented, a stated definitional difference)
  consensus = pairwise            the expert's mean View A Dice against each of the other six, the
                                  common pairwise construct carried to the same per-expert median

Writes results/e5_msseg_experts.csv (one row per patient, expert, consensus) and
results/e5_msseg_summary.csv (per-expert medians). validate_published.py reads the summary.

Usage: python code/validate_msseg.py [--workers N]
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gtu import consensus, metrics
from gtu.io import load_mask
from loaders import msseg2016

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _score_patient(image_id: str, paths: dict[str, Path]) -> list[dict]:
    masks = {rater: metrics.binarise(load_mask(path)) for rater, path in paths.items()}
    raters = sorted(masks)
    rows = []
    for held_out in raters:
        others = [masks[r] for r in raters if r != held_out]
        for method, built in (
            ("staple", consensus.staple_consensus(others)),
            ("majority", consensus.majority_vote(others)),
        ):
            rows.append({"image_id": image_id, "expert": held_out, "consensus": method,
                         "dice": metrics.dice(masks[held_out], built)})
        pairwise = [metrics.dice(masks[held_out], masks[r]) for r in raters if r != held_out]
        rows.append({"image_id": image_id, "expert": held_out, "consensus": "pairwise",
                     "dice": float(np.mean(pairwise))})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    by_image: dict[str, dict[str, Path]] = {}
    for row in msseg2016.load(ROOT / "data/gtu-raw/msseg2016"):
        by_image.setdefault(row.image_id, {})[row.rater_id] = row.mask_path

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_score_patient, image_id, paths) for image_id, paths in sorted(by_image.items())]
        for done, future in enumerate(futures, 1):
            rows += future.result()
            print(f"  {done}/{len(futures)}", flush=True)

    scores = pd.DataFrame(rows)
    scores.to_csv(RESULTS / "e5_msseg_experts.csv", index=False)
    summary = (scores.groupby(["consensus", "expert"]).dice.median()
               .rename("median_dice").reset_index())
    summary.to_csv(RESULTS / "e5_msseg_summary.csv", index=False)
    print(summary.pivot(index="expert", columns="consensus", values="median_dice").round(3))
    span = summary.groupby("consensus").median_dice.agg(["min", "max"]).round(2)
    print("\nrange of per-expert medians (published: 0.66 to 0.76)\n", span)


if __name__ == "__main__":
    main()
