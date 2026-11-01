"""NuCLS multi-rater evaluation-set loader (amgad-2022-nucls). Breast H&E, TCGA slides, 39 raters in
three source-defined tiers: SP.1-3 senior pathologists, JP.1-6 junior pathologists, NP.1-30
non-pathologists.

READ THIS BEFORE USING NuCLS AS A SEGMENTATION SOURCE. Its per-rater BOUNDARIES are not independent.
`code/nucls_boundary_check.py` establishes it on the released files: of 1,429 nuclei annotated by
three or more raters, **99.1 % carry exactly one boundary shape shared by every one of those raters**,
bit-identical after translation. One contour per nucleus is served to all raters, who label it. The
release's own anchors table agrees - `has_manual_boundary` / `has_algorithmic_boundary` are one column
pair per nucleus, and the per-rater columns hold class labels, not geometry.

So under the provenance classes this source splits in two, inside one dataset:

  polyline objects   P5, boundary correlated by construction -> provenance row only, NEVER a pixel floor.
                     This loader does not emit them, and `--report` prints how many were dropped.
  rectangle objects  independent per-rater geometry (0 % identical; raters who annotate the same
                     nucleus draw about as many distinct boxes as there are raters) -> admitted, as
                     the BOX STREAM, which is kept separate from every pixel floor.

The consequence is a subtraction and should be reported as one: NuCLS does not raise the count of
recomputed *microscopy* floors, because the floor it supports is a box floor. What it does supply is
the second source found to be "correlated by construction" after LIDC - and unlike LIDC, whose
sequential unblinded read is stated in Armato 2011, nothing in NuCLS's documentation discloses it.

TWO-STEP USE. Rater crops of the same field differ by a few pixels each (every rater has their own
`left/top/bottom/right`), so there is no shared array to compare until they are put in one frame:

    python code/loaders/nucls.py --prepare     # writes _prepared/<fov>/<rater>.tif once
    python code/loaders/nucls.py --report

`prepare` groups crops into fields, takes the INTERSECTION of the member crops as the canonical frame
so every rater is scored only where every rater actually looked, and rasterises each rater's boxes
into a uint16 instance-label image over that frame. A box is kept when its CENTRE lies in the frame,
and is then clipped to it. Requiring boxes to lie wholly inside instead was tried first and discarded:
on these fields the raters' crops differ by tens of pixels, so the intersection is inset far enough to
put roughly half of all boxes on the boundary, and dropping them threw away 10,873 of 21,969. Clipping
does shorten an edge box, but the frame is common to every rater in the field, so the shortening falls
on all of them identically and a pairwise comparison stays fair - which is what the earlier rule was
protecting, and it protects it at half the cost in data.

IMAGE IDS ARE MINTED, although the release key is otherwise the source's own identifier. NuCLS's
evaluation set ships no field identifier shared across raters - each file name carries that rater's own
crop box - so the id here is `<slide>_left-<L>_top-<T>` of the canonical frame, which is deterministic
and re-derivable from the released file names by rerunning `prepare`.

BOX METRICS. The box stream is scored by box IoU and box F1@0.5. `compute_layer` would otherwise select
instance-matching metrics on `annotation_type is INSTANCE` alone, so a BOX row would get semantic
metrics only. Box F1@0.5 is exactly instance F1@0.5 computed on box masks, so `compute_layer` runs
instance matching for BOX as well. No metric definition changes.

Licence: CC0-1.0 for the NuCLS annotations (amgad-2022-nucls, released to the public domain);
the underlying TCGA images are separately governed. Only derived statistics are released either way.

Published statistic: "NuCLS DSC 85.0 +/- 5.9". Given the finding above, that figure cannot be an
independent inter-rater boundary agreement, and the anchors table's own `algorithmic_vs_manual_DICE`
averages 0.802 over the 496 nuclei carrying both a manual and an algorithmic contour. Which construct the
published number measures is not stated, so it is not used as a reproduction target.
"""

from __future__ import annotations

import collections
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gtu.schema import (
    Annotation,
    AnnotationType,
    Expertise,
    Modality,
    Provenance,
    SourceType,
)

DATASET = "nucls"
LICENCE = "CC0-1.0"
TASK = "nuclei_box"
PREPARED = "_prepared"

#: The one object type with independent per-rater geometry; see the module docstring.
ADMITTED_TYPE = "rectangle"
EXCLUDED_TYPE = "polyline"

#: Crops within this many slide pixels of each other are the same field of view.
TOLERANCE_PX = 40

#: Source-defined tiers (amgad-2022-nucls groups SPs and JPs as "pathologists", NPs as
#: "non-pathologists"). Expertise is never re-assigned by us, so this mapping is
#: the source's, not a judgement about individual raters.
TIERS = {
    "SP": (Expertise.EXPERT, Provenance.P1),
    "JP": (Expertise.EXPERT, Provenance.P1),
    "NP": (Expertise.NON_EXPERT, Provenance.P2),
}

FILENAME = re.compile(
    r"^(?P<rater>[A-Za-z]+\.\d+)_#_E_#_(?P<slide>.+?)_id-[0-9a-f]+"
    r"_left-(?P<left>\d+)_top-(?P<top>\d+)_bottom-(?P<bottom>\d+)_right-(?P<right>\d+)$"
)


def _crops(evalset: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(evalset.glob("*.csv")):
        match = FILENAME.match(path.stem)
        if match is None:
            raise ValueError(f"unexpected NuCLS filename: {path.name}")
        rows.append(
            {
                "path": path,
                "rater": match["rater"],
                "tier": match["rater"].split(".")[0],
                "slide": match["slide"],
                "left": int(match["left"]),
                "top": int(match["top"]),
                "bottom": int(match["bottom"]),
                "right": int(match["right"]),
            }
        )
    crops = pd.DataFrame(rows).sort_values(["slide", "left", "top"]).reset_index(drop=True)

    group, current, previous = [], -1, None
    for row in crops.itertuples():
        if (
            previous is None
            or row.slide != previous.slide
            or abs(row.left - previous.left) > TOLERANCE_PX
            or abs(row.top - previous.top) > TOLERANCE_PX
        ):
            current += 1
        group.append(current)
        previous = row
    crops["fov"] = group
    return crops


def prepare(root: Path | str) -> dict[str, int]:
    """Rasterise every rater's boxes into one common frame per field. Run once; safe to rerun."""
    import tifffile

    root = Path(root)
    crops = _crops(root / "EvaluationSet/csv")
    out_root = root / PREPARED
    counts = collections.Counter()

    for fov, members in crops.groupby("fov"):
        if members.rater.nunique() < 2:
            counts["fov_single_rater"] += 1
            continue
        # every rater is scored only where every rater looked
        left, top = int(members.left.max()), int(members.top.max())
        right, bottom = int(members.right.min()), int(members.bottom.min())
        height, width = bottom - top, right - left
        if height <= 0 or width <= 0:
            counts["fov_empty_intersection"] += 1
            continue

        image_id = f"{members.slide.iloc[0]}_left-{left}_top-{top}"
        folder = out_root / image_id
        folder.mkdir(parents=True, exist_ok=True)

        wrote = 0
        for row in members.itertuples():
            table = pd.read_csv(row.path)
            boxes = table[table.type == ADMITTED_TYPE]
            counts["objects_excluded_polyline"] += int((table.type == EXCLUDED_TYPE).sum())

            absolute = []
            for obj in boxes.itertuples():
                x0, y0 = int(obj.xmin) + row.left, int(obj.ymin) + row.top
                x1, y1 = int(obj.xmax) + row.left, int(obj.ymax) + row.top
                if not (left <= (x0 + x1) // 2 < right and top <= (y0 + y1) // 2 < bottom):
                    counts["boxes_centre_outside_frame"] += 1
                    continue
                clipped = (
                    max(x0, left) - left,
                    max(y0, top) - top,
                    min(x1, right) - left,
                    min(y1, bottom) - top,
                )
                if (x0, y0, x1, y1) != (max(x0, left), max(y0, top), min(x1, right), min(y1, bottom)):
                    counts["boxes_clipped_to_frame"] += 1
                absolute.append(clipped)

            mask = np.zeros((height, width), dtype=np.uint16)
            # large first so a small nucleus inside a large one survives the overwrite
            for index, (x0, y0, x1, y1) in enumerate(
                sorted(absolute, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True),
                start=1,
            ):
                mask[y0:y1, x0:x1] = index
            counts["boxes_written"] += len(absolute)
            tifffile.imwrite(folder / f"{row.rater}.tif", mask)
            wrote += 1
        counts["fov_prepared"] += 1
        counts["rater_masks"] += wrote
    return dict(counts)


def load(root: Path | str) -> list[Annotation]:
    """Emit every NuCLS box-stream row. `root` is normally data/gtu-raw/nucls; run `prepare` first."""
    root = Path(root)
    prepared = root / PREPARED
    if not prepared.is_dir():
        raise FileNotFoundError(
            f"{prepared} missing - run `python code/loaders/nucls.py --prepare` first"
        )

    rows = []
    for folder in sorted(p for p in prepared.iterdir() if p.is_dir()):
        masks = sorted(folder.glob("*.tif"))
        if len(masks) < 2:
            continue
        slide = folder.name.split("_left-")[0]
        for path in masks:
            rater = path.stem
            expertise, provenance = TIERS[rater.split(".")[0]]
            rows.append(
                Annotation(
                    dataset=DATASET,
                    task=TASK,
                    image_id=folder.name,
                    rater_id=rater,
                    mask_path=path,
                    spacing=None,  # 0.2 um/px nominal for TCGA 40x, not shipped per file
                    modality=Modality.HISTOLOGY,
                    annotation_type=AnnotationType.BOX,
                    rater_provenance=provenance,
                    expertise=expertise,
                    licence=LICENCE,
                    source_type=SourceType.RECOMPUTED,
                    cluster_id=slide,  # fields nest in slides, so the bootstrap resamples slides
                )
            )
    return rows


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    root = Path(args[0]) if args else Path(__file__).resolve().parents[2] / "data/gtu-raw/nucls"

    if "--prepare" in sys.argv:
        stats = prepare(root)
        print(f"prepared {root / PREPARED}")
        for key, value in sorted(stats.items()):
            print(f"  {key:26s} {value}")
        print(f"\n  {stats.get('objects_excluded_polyline', 0)} polyline objects excluded: "
              f"boundary is shared across raters (P5), see nucls_boundary_check.py")

    loaded = load(root)
    tiers = collections.Counter(r.rater_id.split(".")[0] for r in loaded)
    classes = collections.Counter(r.rater_provenance.value for r in loaded)
    print(f"\n{len(loaded)} rows from {root}")
    print(f"  task={TASK}  type={loaded[0].annotation_type.value}  "
          f"images={len({r.image_id for r in loaded})}  "
          f"raters={len({r.rater_id for r in loaded})}  "
          f"slides={len({r.cluster_id for r in loaded})}")
    print(f"  rows per tier: {dict(sorted(tiers.items()))}")
    print(f"  rows per provenance class: {dict(sorted(classes.items()))}")
    print(f"  licence={loaded[0].licence}  floor_eligible={loaded[0].floor_eligible()} "
          f"(box stream only - never pooled with a pixel floor)")

    panels = collections.Counter(
        len({r.rater_id for r in loaded if r.image_id == i}) for i in {r.image_id for r in loaded}
    )
    print(f"  panel size -> n fields: {dict(sorted(panels.items()))}")
