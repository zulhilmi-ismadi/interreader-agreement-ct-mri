"""The census: the survey of candidate benchmarks against the eligibility criteria, showing how many
can support an agreement floor, and how few.

It answers, over every benchmark this project examined:

  * how many release **two or more genuinely independent** human annotations per image,
  * how many of those are **individually downloadable** rather than pre-fused,
  * how many survive the **provenance** test (P1 independent raters and P2 independent non-experts
    are eligible; P3 intra-observer repeats, P4 sequential unblinded revision, P5 algorithm-fused or
    uncorrected algorithm output and P6 single-consensus references are not),
  * how many yield a **pixel** floor rather than a box one, and
  * how many may actually be **released** under their licence.

The recomputed half is read from `results/layer/` so it cannot drift from the layer; everything that
never became a floor is curated in `data/derived/census_non_recomputed.csv` with its reason.

    python code/census.py

Writes `results/census.csv` (the survey table) and prints the funnel.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LAYER = ROOT / "results" / "layer"
CURATED = ROOT / "data/derived/census_non_recomputed.csv"
OUT = ROOT / "results" / "census.csv"

#: the provenance classes that may enter a pooled floor: independent experts and independent non-experts
ELIGIBLE_PROVENANCE = {"P1", "P2"}
#: an unstated licence withholds a source's rows from release; NoDerivatives licences and data use
#: agreements are tested separately in `recomputed_rows`
UNRELEASABLE = {"unstated"}


def seeded_datasets() -> set[str]:
    """Sources with any rater flagged `seeded=True`: raters who corrected an algorithmic
    pre-segmentation, kept as independent raters with a flag. Read from the
    loaders, which build annotation lists without opening a mask, so this is cheap."""
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
    return seeded


def recomputed_rows() -> pd.DataFrame:
    """One row per recomputed dataset, built from the layer itself."""
    floors = pd.read_csv(LAYER / "task_floor.csv")
    seeded = seeded_datasets()
    rows = []
    for dataset, block in floors.groupby("dataset"):
        eligible = block[block.floor_eligible]
        licence = sorted(block.licence.dropna().unique())[0]
        # a NoDerivatives (-ND-) licence and a data use agreement (-DUA) are both confirmed licences
        # that still forbid per-image release
        releasable = (licence not in UNRELEASABLE and "-ND-" not in licence.upper()
                      and "-DUA" not in licence.upper())
        rows.append(
            {
                "source": dataset,
                "display_name": dataset,
                "status": "recomputed",
                "modality": "/".join(sorted(block.modality.dropna().unique())),
                "n_raters": int(block.n_raters.max()),
                "n_images": int(block.n_images.max()),
                "annotation_type": "/".join(sorted(block.annotation_type.dropna().unique())),
                "provenance": "/".join(sorted(block.pair_class.dropna().unique())),
                "n_tasks": int(block.task.nunique()),
                "n_tasks_floor_eligible": int(eligible.task.nunique()),
                "has_floor": bool(len(eligible)),
                "pixel_floor": bool(len(eligible[eligible.annotation_type != "box"])),
                "licence": licence,
                "releasable": releasable,
                "seeded": dataset in seeded,
                "reason": "",
                "reference": "",
            }
        )
    return pd.DataFrame(rows)


def curated_rows() -> pd.DataFrame:
    frame = pd.read_csv(CURATED)
    frame["n_tasks"] = pd.NA
    frame["n_tasks_floor_eligible"] = 0
    frame["has_floor"] = False
    frame["pixel_floor"] = False
    frame["licence"] = ""
    frame["releasable"] = False
    frame["seeded"] = False
    return frame


#: the five census criteria, in funnel order
CRITERIA = [
    ("c1_multirater", ">=2 human annotations"),
    ("c2_individually_released", "released individually"),
    ("c3_provenance_ok", "provenance eligible"),
    ("c4_pixel_floor", "pixel floor"),
    ("c5_releasable", "licence permits release"),
]


def add_criteria(census: pd.DataFrame) -> pd.DataFrame:
    """The funnel is STRICTLY NESTED: each criterion is a subset of the one before it.

    Computed on its own, "licence permits release" would also count sources that fail provenance
    (LIDC) or have no pixel floor (the box sources), and a sentence such as "of those, n yield a pixel
    floor and m may be released" would then be false. Nesting makes every count a subset of the
    previous one by construction."""
    census["c1_multirater"] = census.n_raters.fillna(0) >= 2
    # CTC is the reason this is not simply "not census_only": it has three annotators and published
    # their agreement, but never released the individual masks — which is exactly the failure this
    # column exists to record.
    census["c2_individually_released"] = census.c1_multirater & census.status.isin(
        {"recomputed", "pending_access"}
    )
    census["c3_provenance_ok"] = census.c2_individually_released & census.has_floor
    census["c4_pixel_floor"] = census.c3_provenance_ok & census.pixel_floor
    census["c5_releasable"] = census.c4_pixel_floor & census.releasable
    # outside the funnel, reported beside it: releasable floors of either stream, and the two sources
    # that pass c1-c2 but have not been ingested — which is not a provenance failure and must not be
    # counted as one
    census["releasable_floor_any_stream"] = census.c3_provenance_ok & census.releasable
    census["pending_ingestion"] = census.status == "pending_access"
    return census


def main() -> None:
    census = pd.concat([recomputed_rows(), curated_rows()], ignore_index=True)
    order = {"recomputed": 0, "published_band": 1, "pending_access": 2, "census_only": 3}
    census["_order"] = census.status.map(order)
    census = census.sort_values(["_order", "n_raters"], ascending=[True, False]).drop(columns="_order")
    census = add_criteria(census)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    census.to_csv(OUT, index=False)

    # Printed FROM THE SAME COLUMNS the CSV and the census figure use, so a quoted count cannot disagree with
    # the file. Recomputing each count with its own filter would let the two drift apart.
    examined = len(census)
    print(f"wrote {OUT.relative_to(ROOT)}  {examined} benchmarks examined "
          f"(nnU-Net's 23 datasets count as one)\n")
    print("the census funnel (strictly nested):")
    print(f"  benchmarks examined                                    {examined}")
    for column, label in CRITERIA:
        print(f"  ...{label:52s} {int(census[column].sum())}")
    print(f"\nbeside the funnel:")
    print(f"  releasable floors of either stream (pixel or box)      "
          f"{int(census.releasable_floor_any_stream.sum())}")
    pending = census[census.pending_ingestion]
    print(f"  eligible but not yet ingested (NOT a provenance fail)  {len(pending)}  "
          f"{', '.join(pending.display_name)}")

    print("\nby status:")
    for status, block in census.groupby("status", sort=False):
        print(f"  {status:16s} {len(block):2d}  {', '.join(sorted(block.display_name))}")

    print("\nwhy the rest cannot support a floor:")
    for _, row in census[census.status == "census_only"].iterrows():
        print(f"  {row.display_name:22s} {row.provenance}  {row.reason[:96]}")

    print("\nsources whose floor exists but cannot be released:")
    for _, row in census[(census.has_floor) & (~census.releasable)].iterrows():
        why = "licence unstated" if row.licence in UNRELEASABLE else f"licence {row.licence}"
        print(f"  {row.display_name:22s} {why}")


if __name__ == "__main__":
    main()
