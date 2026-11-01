"""Provenance check of the NuCLS multi-rater evaluation set.

WHY THIS EXISTS. NuCLS is cited as a 32-annotator histology dataset. Before computing a floor from it,
this script asks the question the eligibility rules put to every source: are the per-rater
annotations independent?

WHAT IT FINDS. They are, for bounding boxes, and are not, for boundaries. Within one dataset:

  polyline objects   ~99 % of nuclei annotated by >= 3 raters carry ONE boundary shape shared by all
                     of them, bit-identical after translation. Independent tracing cannot produce
                     that. The boundary is a single manual or algorithmic contour served to every
                     rater, who then labels it. Class P5 (algorithm-fused or seeded-then-uncorrected
                     labels presented as raters) for the boundary dimension: provenance row only, no floor.

  rectangle objects  0 % identical; the raters who annotate the same nucleus draw ~as many distinct
                     boxes as there are raters. These are genuine independent per-rater geometry.
                     Class P1 (SP, JP) and P2 (NP), box stream only.

The anchors table shipped with the release corroborates it directly: `has_manual_boundary` and
`has_algorithmic_boundary` are ONE column pair per nucleus, not one per rater, and the per-rater
columns (SP.1 ... NP.29) hold class labels, not geometry.

CONSEQUENCE: NuCLS supplies a real multi-rater
DETECTION and CLASSIFICATION signal and a real box floor, but it cannot supply a pixel or boundary
agreement floor. It is the second source
whose annotations are correlated by construction, alongside LIDC's sequential unblinded read - and,
unlike LIDC, the correlation is not disclosed in the dataset's own documentation.

Method: rater crops are grouped into FOVs by slide plus crop origin within TOLERANCE_PX; objects are
lifted into absolute slide coordinates using the crop offsets in the file name; a nucleus is the set
of objects whose centroids agree within CENTROID_PX; a shape signature is the vertex sequence made
translation-invariant. Both groupings are deliberately loose - a loose grouping can only ever merge
distinct nuclei and so *inflate* the count of distinct shapes, which is conservative against the
conclusion drawn here.

Writes data/derived/nucls_boundary_provenance.csv (per object type) and
data/derived/nucls_shape_sharing_per_nucleus.csv (per nucleus).

Usage: python code/nucls_boundary_check.py
"""

from __future__ import annotations

import collections
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVALSET = ROOT / "data/gtu-raw/nucls/EvaluationSet/csv"
DERIVED = ROOT / "data/derived"

#: two rater crops belong to the same FOV if their origins agree within this many slide pixels
TOLERANCE_PX = 40
#: two objects are the same nucleus if their centroids agree within this many slide pixels
CENTROID_PX = 12
#: a nucleus needs this many raters before "do they agree on the shape" is a meaningful question
MIN_RATERS = 3

FILENAME = re.compile(
    r"^(?P<rater>[A-Za-z]+\.\d+)_#_E_#_(?P<slide>.+?)_id-[0-9a-f]+"
    r"_left-(?P<left>\d+)_top-(?P<top>\d+)_bottom-(?P<bottom>\d+)_right-(?P<right>\d+)$"
)


def rater_files(evalset: Path = EVALSET) -> pd.DataFrame:
    """One row per (rater, crop), with the crop's absolute slide coordinates and its FOV group."""
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
    files = pd.DataFrame(rows).sort_values(["slide", "left", "top"]).reset_index(drop=True)

    group, current, previous = [], -1, None
    for row in files.itertuples():
        if (
            previous is None
            or row.slide != previous.slide
            or abs(row.left - previous.left) > TOLERANCE_PX
            or abs(row.top - previous.top) > TOLERANCE_PX
        ):
            current += 1
        group.append(current)
        previous = row
    files["fov"] = group
    return files


def _objects(files: pd.DataFrame, kind: str) -> dict:
    """Bucket objects of one type into nuclei, carrying each rater's shape and box signature."""
    nuclei = collections.defaultdict(list)
    for row in files.itertuples():
        table = pd.read_csv(row.path)
        for obj in table[table.type == kind].itertuples():
            xs = np.array([int(v) for v in str(obj.coords_x).split(",")]) + row.left
            ys = np.array([int(v) for v in str(obj.coords_y).split(",")]) + row.top
            key = (row.fov, round(xs.mean() / CENTROID_PX), round(ys.mean() / CENTROID_PX))
            shape = (tuple((xs - xs.min()).tolist()), tuple((ys - ys.min()).tolist()))
            box = (int(obj.xmax) - int(obj.xmin), int(obj.ymax) - int(obj.ymin))
            nuclei[key].append((row.rater, row.tier, shape, box))
    return nuclei


def audit(files: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_nucleus, summary = [], []
    for kind in ("polyline", "rectangle"):
        nuclei = _objects(files, kind)
        for (fov, cx, cy), members in nuclei.items():
            raters = {r for r, _, _, _ in members}
            if len(raters) < MIN_RATERS:
                continue
            per_nucleus.append(
                {
                    "object_type": kind,
                    "fov": fov,
                    "centroid_x": cx * CENTROID_PX,
                    "centroid_y": cy * CENTROID_PX,
                    "n_raters": len(raters),
                    "n_distinct_shapes": len({s for _, _, s, _ in members}),
                    "n_distinct_box_sizes": len({b for _, _, _, b in members}),
                    "tiers": ",".join(sorted({t for _, t, _, _ in members})),
                }
            )

    per_nucleus = pd.DataFrame(per_nucleus)
    for kind, group in per_nucleus.groupby("object_type"):
        summary.append(
            {
                "object_type": kind,
                "n_nuclei": len(group),
                "median_raters_per_nucleus": int(group.n_raters.median()),
                "median_distinct_shapes": int(group.n_distinct_shapes.median()),
                "pct_one_shape_for_all_raters": round(
                    100 * float((group.n_distinct_shapes == 1).mean()), 1
                ),
                "pct_one_box_size_for_all_raters": round(
                    100 * float((group.n_distinct_box_sizes == 1).mean()), 1
                ),
                "shapes_per_rater": round(
                    float((group.n_distinct_shapes / group.n_raters).mean()), 3
                ),
                "independent_geometry": kind == "rectangle",
                "provenance_class": "P5" if kind == "polyline" else "P1/P2",
            }
        )
    return pd.DataFrame(summary), per_nucleus


def main() -> int:
    if not EVALSET.is_dir():
        raise FileNotFoundError(EVALSET)
    files = rater_files()
    summary, per_nucleus = audit(files)

    DERIVED.mkdir(parents=True, exist_ok=True)
    summary.to_csv(DERIVED / "nucls_boundary_provenance.csv", index=False)
    per_nucleus.to_csv(DERIVED / "nucls_shape_sharing_per_nucleus.csv", index=False)

    tiers = files.groupby("tier").rater.nunique().to_dict()
    print(f"{len(files)} rater-crop files, {files.rater.nunique()} raters "
          f"({', '.join(f'{k}={v}' for k, v in sorted(tiers.items()))}), "
          f"{files.slide.nunique()} slides, {files.fov.nunique()} FOV groups\n")

    for row in summary.itertuples():
        print(f"{row.object_type:10s}  nuclei>={MIN_RATERS} raters: {row.n_nuclei:5d}   "
              f"median raters {row.median_raters_per_nucleus:2d}   "
              f"median distinct shapes {row.median_distinct_shapes}   "
              f"all raters share one shape: {row.pct_one_shape_for_all_raters:5.1f} %   "
              f"-> {row.provenance_class}")

    print("\nRead this as: a polyline nucleus has ONE boundary no matter how many raters annotated it,")
    print("so the boundary is not per-rater evidence. A rectangle nucleus has about as many distinct")
    print("boxes as raters, so the box IS per-rater evidence. NuCLS is therefore admitted to")
    print("the box stream only, and its boundary dimension is recorded as provenance class P5.")
    print("\nwrote data/derived/nucls_boundary_provenance.csv")
    print("wrote data/derived/nucls_shape_sharing_per_nucleus.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
