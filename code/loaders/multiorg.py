"""MultiOrg loader (bukas-2024-multiorg). Bright-field lung-organoid microscopy, two expert
annotators, **bounding boxes**.

This is a **box-stream source**: box IoU and box F1@0.5 only, never pooled with a
pixel floor. It is the box stream's microscopy source and the second intra-observer source among the
datasets analysed, after CryoNuSeg.

Layout (the `MultiOrg_v2` tree, which is richer than the top level):
`<split>/<condition>/<plate>/<image_N>/{image_N.tiff, image_N_<label set>.json}`.

Notes that matter:

1. **Only the 55 test images can form a pair.** The train split has 356 image folders carrying exactly
   one label set each -- 196 `Annotator_A` and 160 `Annotator_B`, never both on the same image -- so it
   contributes no rater pair at all and is not loaded. Every pairing MultiOrg can support lives in the
   test split.

2. **`t0`/`t1` are two annotation passes on one image, not two timepoints.** This was checked rather
   than assumed: each folder holds exactly one TIFF, and the median nearest-neighbour distance between
   `t0_A` and `t1_A` box centres is 0-4.3 px on a 6240x6940 image. So the test panel is
   `t1_A` + `t1_B` (both annotators, all 55 images) plus one `t0` per image (22 from A, 33 from B),
   which is the release's "3 test label sets". Two tasks follow, exactly as CryoNuSeg's do:

     organoid        t1_A vs t1_B                 both P1 -> inter-observer box row
     organoid_intra  t0_X vs t1_X, same annotator P3      -> intra-observer row only

3. **Annotator B marks far fewer organoids than A** -- 63 vs 130, 101 vs 141, 51 vs 147 on the first
   images checked. Most of the disagreement here is about *existence*, not extent, which is what a box
   F1 measures and a Dice would hide. It is a different failure mode from the boundary disagreement
   every pixel source shows.

4. **Boxes are rasterised to instance label images** by `--prepare`, because the canonical schema
   stores a mask path. Each box becomes one integer label in a uint16 PNG at the
   TIFF's own shape; PNG keeps rectangles small on disk. Overlapping boxes resolve last-wins, recorded
   here because it is a real choice: a later box overwrites an earlier one's pixels, so a heavily
   overlapped panel loses a little instance area. Box **coordinates are (row, col)**, verified against
   the TIFF shape -- coord0 tops out at 5478 against 6240 rows, coord1 at 6128 against 6940 columns.

5. `compute_layer` treats `box` like `instance` (see its own note), so this source
   gets instance F1 at IoU 0.5 and 0.7 -- which is box F1@0.5, the box stream's metric -- plus
   the semantic overlap metrics on the union of boxes. Report only the box metrics.

Licence: **CC BY-NC-SA 4.0** for the data (Apache 2.0 for their code), per the release. Non-commercial
and share-alike, which constrain derived statistics; `licences.tsv` records the licence, and its rows
are not to be treated like those of the CC BY sources.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gtu.schema import (
    Annotation,
    AnnotationType,
    Expertise,
    Modality,
    Provenance,
    SourceType,
)

DATASET = "multiorg"
LICENCE = "CC-BY-NC-SA-4.0"

INTER_TASK = "organoid"
INTRA_TASK = "organoid_intra"

#: only the test split has images with two annotators; see note 1
SPLIT = "test"

#: load-integrity checks: counts in the released MultiOrg_v2 tree
EXPECTED_IMAGES = 55
EXPECTED_MASKS = 165  # t1_A and t1_B on all 55, plus one t0 per image

_LABEL_SET = re.compile(r"^image_\d+_(?P<pass>t\d+)_(?P<annotator>[AB])$")


def load(root: Path | str) -> list[Annotation]:
    """Emit every MultiOrg row. `root` holds the prepared masks, normally data/gtu-raw/multiorg."""
    root = Path(root)
    rows = []
    for image_dir in sorted(d for d in root.rglob("image_*") if d.is_dir()):
        image_id = f"{image_dir.parent.parent.name}_{image_dir.parent.name}_{image_dir.name}"
        sets: dict[tuple[str, str], Path] = {}
        for mask_path in sorted(image_dir.glob("*.png")):
            match = _LABEL_SET.match(mask_path.stem)
            if match is None:
                raise ValueError(f"unexpected MultiOrg mask name: {mask_path}")
            sets[(match["annotator"], match["pass"])] = mask_path

        # the inter-observer panel: both annotators' second pass, present on every test image
        for annotator in ("A", "B"):
            path = sets.get((annotator, "t1"))
            if path is None:
                raise ValueError(f"{image_dir}: missing {annotator}_t1, the inter-observer panel")
            rows.append(_row(path, image_id, f"{annotator}_t1", INTER_TASK, Provenance.P1))

        # the intra-observer repeat: ONLY the annotator who has a first pass on this image. Both t1
        # masks are present, so pairing the whole folder would silently score A_t0 against B_t1 —
        # a cross-annotator pair inside an intra-observer task, which is by definition a
        # repeat by the same rater.
        repeated = [a for (a, p) in sets if p == "t0"]
        if len(repeated) != 1:
            raise ValueError(f"{image_dir}: expected exactly one t0 pass, found {repeated}")
        annotator = repeated[0]
        rows.append(_row(sets[(annotator, "t0")], image_id, f"{annotator}_t0", INTRA_TASK, Provenance.P3))
        rows.append(_row(sets[(annotator, "t1")], image_id, f"{annotator}_t1", INTRA_TASK, Provenance.P3))
    return rows


def _row(path: Path, image_id: str, rater_id: str, task: str, provenance: Provenance) -> Annotation:
    return Annotation(
        dataset=DATASET,
        task=task,
        image_id=image_id,
        rater_id=rater_id,
        mask_path=path,
        spacing=None,  # no physical pixel size ships; distances in pixels
        modality=Modality.MICROSCOPY,
        annotation_type=AnnotationType.BOX,  # box stream
        rater_provenance=provenance,
        expertise=Expertise.EXPERT,
        licence=LICENCE,
        source_type=SourceType.RECOMPUTED,
        seeded=True,  # raters corrected tool pseudo-labels
        cluster_id=image_id.rsplit("_image", 1)[0],  # images nest in plates
        label_value=None,
    )


def prepare(source: Path | str, out: Path | str) -> None:
    """Rasterise every test-split box set into a uint16 instance PNG (note 4)."""
    import numpy as np
    import tifffile
    from skimage import io as skimage_io

    source, out = Path(source), Path(out)
    image_dirs = sorted(d for d in (source / SPLIT).rglob("image_*") if d.is_dir())
    if len(image_dirs) != EXPECTED_IMAGES:
        raise ValueError(f"expected {EXPECTED_IMAGES} test images, found {len(image_dirs)}")

    written = 0
    for index, image_dir in enumerate(image_dirs, 1):
        tiffs = list(image_dir.glob("*.tiff"))
        if len(tiffs) != 1:
            raise ValueError(f"{image_dir}: expected one TIFF, found {len(tiffs)}")
        with tifffile.TiffFile(tiffs[0]) as handle:
            shape = handle.series[0].shape[:2]

        target_dir = out / image_dir.parent.parent.name / image_dir.parent.name / image_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)
        for json_path in sorted(image_dir.glob("*.json")):
            target = target_dir / f"{json_path.stem}.png"
            if target.exists():
                continue
            boxes = json.loads(json_path.read_text())
            mask = np.zeros(shape, dtype=np.uint16)
            for label, corners in enumerate(boxes.values(), start=1):
                points = np.asarray(corners)
                r0, c0 = points.min(axis=0)
                r1, c1 = points.max(axis=0)
                # coordinates are (row, col), verified against the TIFF shape (note 4)
                mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = label
            skimage_io.imsave(target, mask, check_contrast=False)
            written += 1
        if index % 10 == 0:
            print(f"  prepared {index}/{len(image_dirs)} images", flush=True)
    print(f"wrote {written} instance masks to {out}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    repo = Path(__file__).resolve().parents[2]
    out = Path(args[0]) if args else repo / "data/gtu-raw/multiorg"

    if "--prepare" in sys.argv:
        prepare(repo / "data/MultiOrg/MultiOrg_v2", out)

    loaded = load(out)
    print(f"{len(loaded)} rows from {out}")
    print(f"  images={len({r.image_id for r in loaded})}  plates={len({r.cluster_id for r in loaded})}")
    for task in (INTER_TASK, INTRA_TASK):
        task_rows = [r for r in loaded if r.task == task]
        if not task_rows:
            continue
        print(
            f"  {task}: {len(task_rows)} rows, images={len({r.image_id for r in task_rows})}, "
            f"raters={sorted({r.rater_id for r in task_rows})}, "
            f"provenance={sorted({r.rater_provenance.value for r in task_rows})}, "
            f"floor_eligible={any(r.floor_eligible() for r in task_rows)}"
        )
    print(f"  annotation_type={loaded[0].annotation_type.value} (box stream)  licence={loaded[0].licence}")
