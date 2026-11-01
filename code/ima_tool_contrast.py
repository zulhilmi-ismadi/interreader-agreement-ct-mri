"""IMA++ agreement by drawing tool, and the same-annotator pairs the loader sets aside.

The IMA++ release records a tool per segmentation: T1 manual polygon tracing, T2 semi-automated
flood-fill from a human seed point, T3 a fully-automated mask reviewed and accepted by a human
expert (Zenodo 10.5281/zenodo.14201692). T3 is class P5 (accepted algorithm output) and T2 is P1
flagged seeded (a human working from an algorithmic seed), so the
dermoscopy floor rests on T1 and T2 only (`loaders/ima.py`, notes 4 and 5). Two questions follow,
and neither is answered by the layer tables:

1. **Does agreement depend on the tool?** Every IMA++ rater pair is labelled by the tools of its two
   masks, and the floor is reported per combination. A pair of accepted algorithm outputs agrees far
   more closely than two hand-drawn masks, which is the same lesson as the NuCLS shared contour: the
   agreement is between tools, not between people.

2. **What do the set-aside masks say?** 218 images carry two masks from one annotator, usually the
   same person's T1 and T3. The loader keeps the lowest tool and drops the other, so these never
   enter a pair. Scored here, they measure how far one person's accepted algorithm output sits from
   their own tracing — an intra-annotator, cross-tool contrast, not a second opinion.

    python code/ima_tool_contrast.py

Writes results/ima_tool_contrast.csv (floor per tool combination) and
results/ima_same_annotator_pairs.csv (one row per set-aside pair). Needs scipy, as the layer does.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from gtu import metrics  # noqa: E402
from gtu.io import align, WindowedMask  # noqa: E402
from loaders import ima  # noqa: E402

RAW = ROOT / "data/gtu-raw/ima"
PAIRS = ROOT / "results/ima/pairs.csv"
OUT_CONTRAST = ROOT / "results/ima_tool_contrast.csv"
OUT_SAME = ROOT / "results/ima_same_annotator_pairs.csv"
#: the primary NSD tolerance for 2D sources, in pixels (the grid is 1, 2 and 4 px)
TAU = 2


def tool_index() -> pd.DataFrame:
    """(ISIC_id, annotator) -> tool, for the mask each pair actually used."""
    frame = pd.read_csv(ima._subset_csv(RAW))
    human = frame[~frame.annotator.isin(ima.FUSED_ANNOTATORS)].copy()
    human["_tool_rank"] = human.tool.map(ima.TOOL_ORDER)
    kept = (
        human.sort_values(["ISIC_id", "annotator", "_tool_rank", "mskObjectID"])
        .drop_duplicates(["ISIC_id", "annotator"], keep="first")
    )
    return human, kept


def contrast(kept: pd.DataFrame) -> pd.DataFrame:
    """Floor per tool combination, over the pairs the layer actually scored."""
    pairs = pd.read_csv(PAIRS)
    tools = dict(zip(zip(kept.ISIC_id, kept.annotator), kept.tool))
    pairs["tool_a"] = [tools.get((i, a)) for i, a in zip(pairs.image_id, pairs.rater_a)]
    pairs["tool_b"] = [tools.get((i, b)) for i, b in zip(pairs.image_id, pairs.rater_b)]
    missing = pairs[pairs.tool_a.isna() | pairs.tool_b.isna()]
    if len(missing):
        raise ValueError(f"{len(missing)} IMA++ pairs have no tool in the release index")
    pairs["combination"] = ["-".join(sorted((a, b))) for a, b in zip(pairs.tool_a, pairs.tool_b)]

    rows = []
    for combination, block in pairs.groupby("combination"):
        per_image = block.groupby("image_id").dice.mean()
        rows.append(
            {
                "combination": combination,
                "pair_class": "/".join(sorted(block.pair_class.unique())),
                "n_pairs": len(block),
                "n_images": block.image_id.nunique(),
                "floor_dice": round(float(per_image.mean()), 4),
                "floor_iou": round(float(block.groupby("image_id").iou.mean().mean()), 4),
                f"floor_nsd@{TAU}": round(float(block.groupby("image_id")[f"nsd@{TAU}"].mean().mean()), 4),
            }
        )
    return pd.DataFrame(rows).sort_values("combination")


def same_annotator(human: pd.DataFrame, kept: pd.DataFrame) -> pd.DataFrame:
    """Score the masks the loader set aside against the mask it kept, for the same annotator."""
    kept_ids = set(zip(kept.ISIC_id, kept.annotator, kept.mskObjectID))
    duplicated = human.groupby(["ISIC_id", "annotator"]).filter(lambda g: len(g) > 1)

    rows = []
    for (image_id, annotator), group in duplicated.groupby(["ISIC_id", "annotator"]):
        group = group.sort_values(["_tool_rank", "mskObjectID"])
        first = group.iloc[0]
        for _, other in group.iloc[1:].iterrows():
            a = WindowedMask.from_array(_mask(first.seg_filename))
            b = WindowedMask.from_array(_mask(other.seg_filename))
            scores = metrics.pairwise_scores(*align(a, b), instance=False, spacing=None)
            rows.append(
                {
                    "image_id": image_id,
                    "annotator": annotator,
                    "tool_kept": first.tool,
                    "tool_set_aside": other.tool,
                    "combination": "-".join(sorted((first.tool, other.tool))),
                    "dice": round(float(scores["dice"]), 4),
                    "iou": round(float(scores["iou"]), 4),
                    f"nsd@{TAU}": round(float(scores[f"nsd@{TAU}"]), 4),
                    "kept_in_layer": (image_id, annotator, first.mskObjectID) in kept_ids,
                }
            )
    return pd.DataFrame(rows)


def _mask(seg_filename: str) -> np.ndarray:
    from gtu.io import load_mask

    return load_mask(RAW / "segs" / seg_filename)


def main() -> None:
    human, kept = tool_index()
    table = contrast(kept)
    table.to_csv(OUT_CONTRAST, index=False)
    print("floor by tool combination (T1 manual, T2 seeded flood-fill, T3 accepted algorithm):")
    print(table.to_string(index=False))

    same = same_annotator(human, kept)
    same.to_csv(OUT_SAME, index=False)
    print(f"\nset-aside same-annotator masks: {len(same)} pairs on {same.image_id.nunique()} images")
    if len(same):
        by_combination = same.groupby("combination").dice.agg(["size", "mean", "median"]).round(3)
        print(by_combination.to_string())
    print(f"\nwrote {OUT_CONTRAST.relative_to(ROOT)} and {OUT_SAME.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
