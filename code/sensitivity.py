"""Sensitivity analyses: S2 (tolerance), S4 (empty-mask policy), S6 (seeded against unseeded).
S5 (drop-any-one-rater) lives in `sensitivity_s5.py` and is not repeated here.

    python code/sensitivity.py

Writes `results/s2_tolerance.csv`, `results/s4_empty_mask_policy.csv`, `results/s6_seeded.csv`.

**S4 matters to cross-dataset comparisons, not only as a sensitivity table.** The Arvaniti and Gleason 2019
floors look like a clean cross-dataset replication (gleason5 0.855 vs 0.881) only because
~80 % of Gleason-5 pairs are both-empty and score 1.0 under the empty-mask rule. Any replication or
cross-source comparison must therefore be stated on informative pairs.

**The identification problem S4 has to solve, stated openly.** In `pairs.csv` a both-empty pair
(DSC = IoU = 1, HD95 = 0 by the empty-mask rule) is indistinguishable from a pair of *identical non-empty*
masks, which produces the same numbers legitimately. `validate_empty_classification()` measures how
often that confusion actually occurs, by reading Arvaniti's masks directly, and prints the error rate
so the S4 numbers are quoted with it rather than in spite of it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LAYER = ROOT / "results" / "layer"
RESULTS = ROOT / "results"


def classify_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    """Label each pair both-empty / one-empty / informative (both empty: DSC = IoU = NSD = 1, HD95 = 0;
    exactly one empty: DSC = IoU = NSD = 0, HD95 = NaN)."""
    pairs = pairs.copy()
    both = (pairs.dice == 1.0) & (pairs.iou == 1.0) & (pairs.hd95 == 0.0)
    one = (pairs.dice == 0.0) & (pairs.hd95.isna())
    pairs["empty_class"] = np.where(both, "both_empty", np.where(one, "one_empty", "informative"))
    return pairs


def validate_empty_classification() -> dict[str, float]:
    """Read Arvaniti's masks and count how many `both_empty`-looking pairs are really identical
    non-empty masks. Arvaniti is the right test bed: small, single-channel, and it is known to contain
    bit-identical mask pairs, so it is the case most likely to expose the confusion."""
    from PIL import Image

    from loaders.arvaniti import TASKS

    root = ROOT / "data/gtu-raw/arvaniti"
    if not root.is_dir():
        return {}

    cores = sorted(p.stem for p in (root / "r1").glob("*.png"))
    truly_empty = perfect_match = 0
    for core in cores:
        a = np.array(Image.open(root / "r1" / f"{core}.png"))
        b = np.array(Image.open(root / "r2" / f"{core}.png"))
        for label in TASKS.values():
            ma, mb = a == label, b == label
            if not ma.any() and not mb.any():
                truly_empty += 1
            elif np.array_equal(ma, mb):
                perfect_match += 1  # scores 1.0 too, but is NOT an empty pair
    total = truly_empty + perfect_match
    rate = perfect_match / total if total else 0.0
    print(f"S4 validation on Arvaniti — pairs scoring a perfect 1.0: {total}")
    print(f"  genuinely both-empty : {truly_empty}")
    print(f"  identical non-empty  : {perfect_match}   <- misread as both-empty by the heuristic")
    print(f"  heuristic error rate : {rate:.2%}\n")
    return {"perfect_total": total, "truly_empty": truly_empty,
            "perfect_match": perfect_match, "error_rate": rate}


def _floor(block: pd.DataFrame) -> float:
    """The floor exactly as the layer builds it: mean over pairs within each
    image, then mean over images. A plain mean over pooled pairs is NOT the same quantity — it weights
    each image by its pair count — so a pooled-pair mean set against layer floors would not be like
    for like."""
    if block.empty:
        return np.nan
    return float(block.groupby("image_id").dice.mean().mean())


def s4_empty_mask_policy(pairs: pd.DataFrame, floors: pd.DataFrame) -> pd.DataFrame:
    """Primary empty-mask policy versus excluding both-empty pairs, both computed as layer floors."""
    layer = (
        floors[floors.metric == "dice"]
        .set_index(["dataset", "task", "pair_class"])["mean"]
        .to_dict()
    )
    rows = []
    for (dataset, task, pair_class), block in pairs.groupby(["dataset", "task", "pair_class"]):
        informative = block[block.empty_class == "informative"]
        counts = block.empty_class.value_counts()
        primary = _floor(block)
        corrected = _floor(informative)
        rows.append(
            {
                "dataset": dataset,
                "task": task,
                "pair_class": pair_class,
                "n_pairs": len(block),
                "n_images": block.image_id.nunique(),
                "frac_both_empty": counts.get("both_empty", 0) / len(block),
                "frac_one_empty": counts.get("one_empty", 0) / len(block),
                "frac_informative": len(informative) / len(block),
                "dice_primary": primary,
                "dice_layer_floor": layer.get((dataset, task, pair_class), np.nan),
                "dice_informative_only": corrected,
                "delta": corrected - primary if np.isfinite(corrected) else np.nan,
            }
        )
    frame = pd.DataFrame(rows).sort_values("frac_both_empty", ascending=False)
    # Guard: the recomputed primary must equal the layer's own floor, or this table is not comparable
    # with anything else in the paper. Cluster-weighted sources can differ; report, do not hide.
    gap = (frame.dice_primary - frame.dice_layer_floor).abs()
    mismatched = frame[gap > 1e-6]
    print(f"S4 like-for-like check — primary recomputed as image-then-task matches the layer floor "
          f"on {len(frame) - len(mismatched)}/{len(frame)} task floors")
    for _, row in mismatched.head(5).iterrows():
        print(f"    mismatch {row.dataset}/{row.task}/{row.pair_class}: "
              f"{row.dice_primary:.4f} vs layer {row.dice_layer_floor:.4f}")
    return frame


def s2_tolerance(floors: pd.DataFrame) -> pd.DataFrame:
    """NSD at every tolerance on the grid — does the choice of tau reorder the sources?"""
    from scipy.stats import kendalltau

    # 2D and volumetric sources have DIFFERENT tolerance grids ({1, 2, 4} px; {1, 2, 3} mm), so no task carries
    # all four values and an intersection across every tau is empty. Compare within each grid.
    # floor_reportable: an all-empty task must not enter a ranking
    reportable = floors["floor_reportable"] if "floor_reportable" in floors else floors.floor_eligible
    nsd = floors[floors.metric.str.startswith("nsd@") & reportable]
    wide = nsd.pivot_table(index=["dataset", "task", "pair_class"], columns="metric", values="mean")
    grids: dict[tuple[str, ...], list[str]] = {}
    for index, row in wide.iterrows():
        present = tuple(sorted(row.dropna().index, key=lambda c: float(c.split("@")[1])))
        grids.setdefault(present, []).append(index)

    for present, members in sorted(grids.items(), key=lambda kv: -len(kv[1])):
        block = wide.loc[members, list(present)]
        print(f"S2 tolerance — {len(block)} task floors on grid {list(present)}")
        for a, b in zip(present, present[1:]):
            tau, _ = kendalltau(block[a], block[b])
            moved = (block[a].rank() != block[b].rank()).mean()
            print(f"    {a:7s} -> {b:7s}  Kendall tau {tau:.3f}   floors changing rank {moved:.1%}")
        if len(present) > 2:
            tau, _ = kendalltau(block[present[0]], block[present[-1]])
            print(f"    {present[0]:7s} -> {present[-1]:7s}  Kendall tau {tau:.3f}  (widest span)")
        print(f"    mean floor: " + "  ".join(f"{t}={block[t].mean():.3f}" for t in present))
    print()
    return wide.reset_index()


def s6_seeded(floors: pd.DataFrame) -> pd.DataFrame:
    """Algorithm-seeded sources versus the rest, on the same modality where possible."""
    import sys

    sys.path.insert(0, str(ROOT / "code"))
    import compute_layer as cl

    seeded = set()
    for name, (loader, root) in cl.SOURCES.items():
        try:
            if any(r.seeded for r in loader(root)):
                seeded.add(name)
        except Exception:
            continue

    reportable = floors["floor_reportable"] if "floor_reportable" in floors else floors.floor_eligible
    # P1 only: comparing a seeded expert panel against a pool that includes C3RO's non-expert tier
    # confounds seeding with expertise.
    dice = floors[(floors.metric == "dice") & reportable & (floors.pair_class == "P1")].copy()
    dice["seeded"] = dice.dataset.isin(seeded)
    print(f"S6 seeded — algorithm-seeded sources: {sorted(seeded) or 'none'}")
    # The flag is per rater, so a source enters this comparison only when its release says WHICH
    # raters worked from a seed. PDACVI's documentation reports an algorithmic seed for one rater, but
    # the release does not identify that rater, so PDACVI cannot enter here.
    print("  sources documented as partly seeded but not comparable here (seeded rater not "
          "identified in the release): pdacvi")
    print("  DESCRIPTIVE ONLY. Seeded and unseeded sources differ in structure, cohort and panel, so a")
    print("  difference here cannot be attributed to seeding. Matched structures are shown where they exist.")
    for modality, block in dice.groupby("modality"):
        if block.seeded.nunique() < 2:
            continue
        for flag, part in block.groupby("seeded"):
            label = "seeded    " if flag else "not seeded"
            print(f"  {modality:11s} {label} n={len(part):3d} P1 floors  mean Dice {part['mean'].mean():.3f}")
    # The only like-for-like check available: the same anatomical structure, seeded vs not. Match on
    # MODALITY AND task, never on the task name alone -- "lesion" names a skin lesion in dermoscopy
    # and a pancreatic lesion in CT, and matching by name alone would pair those two, since IMA++ is
    # in the seeded set.
    structure = dice.assign(organ=dice.task.str.replace(r"_task\d+$", "", regex=True))
    shared = structure.groupby(["modality", "organ"]).seeded.nunique()
    for modality, organ in shared[shared > 1].index:
        part = structure[(structure.modality == modality) & (structure.organ == organ)]
        print(f"  matched structure '{organ}' ({modality}): " + ", ".join(
            f"{r.dataset} {'seeded' if r.seeded else 'unseeded'} {r['mean']:.3f}" for _, r in part.iterrows()))
    print()
    return dice[["dataset", "task", "modality", "seeded", "mean"]]


def main() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "code"))

    floors = pd.read_csv(LAYER / "task_floor.csv")
    pairs = classify_pairs(pd.read_csv(LAYER / "pairs.csv"))

    validate_empty_classification()

    four = s4_empty_mask_policy(pairs, floors)
    four.to_csv(RESULTS / "s4_empty_mask_policy.csv", index=False)
    worst = four[four.frac_both_empty > 0.25].sort_values("delta")
    print(f"S4 empty-mask policy — {len(four)} task floors; "
          f"{len(worst)} have >25% both-empty pairs")
    print("  the floors most inflated by the primary policy:")
    for _, row in worst.head(10).iterrows():
        print(f"    {row.dataset:12s} {row.task:16s} both-empty {row.frac_both_empty:5.1%}  "
              f"{row.dice_primary:.3f} -> {row.dice_informative_only:.3f}  ({row.delta:+.3f})")
    print()

    two = s2_tolerance(floors)
    two.to_csv(RESULTS / "s2_tolerance.csv", index=False)

    six = s6_seeded(floors)
    six.to_csv(RESULTS / "s6_seeded.csv", index=False)

    print("wrote results/s2_tolerance.csv, results/s4_empty_mask_policy.csv, results/s6_seeded.csv")


if __name__ == "__main__":
    main()
