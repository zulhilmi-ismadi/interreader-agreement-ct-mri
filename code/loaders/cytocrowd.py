"""CytoCrowd loader (WWW '26, Si et al.). Cervical cytology, four board-certified pathologists,
**bounding boxes**.

A **box-stream source**: box IoU and box F1@0.5 only, never pooled with a pixel
floor. It is the box stream's third domain, after MultiOrg (microscopy) and
NuCLS (histology).

Source: https://github.com/YHSI5358/CytoCrowd-Datasets — `Datasets/WorkerData.csv` (the four
pathologists) and `Datasets/GroundTruth.csv` (a separate gold standard).

Notes that matter:

1. **Provenance is P1, stated in the methods**: "four board-certified pathologists ... To ensure the
   independence of each expert's opinion, they are asked to annotate the images separately", each
   drawing bounding boxes on a custom platform over 34 diagnostic classes. Hand-drawn, blinded.

2. **The `sam_bbox` / `sam_conf` columns are NOT part of the annotation loop.** Every worker row
   carries a SAM box alongside the human one, and the two are never equal (checked: 0 of 14,579
   match). SAM appears in the paper only as related work. We read `worker_bbox`; the SAM columns are
   ignored, and no row here is algorithm-seeded (`seeded=False`).

3. **The design is complete and crossed** — 4 raters x 446 images, every rater on every image, 14,579
   boxes. No other source in the layer has no missing pairs.

4. **Rater identity is pseudonymised on ingest.** `WorkerData.csv` carries the pathologists' real
   names in its `author` column. We key on `author_id` and emit `r1..r4`; the names are never read
   into the layer, because the release publishes per-rater agreement statistics and those can single
   out a named individual as an outlier (cf. Gleason 2019's r2). The source does not pseudonymise;
   we do.

5. **Image geometry is 2160 x 3840 (h x w)**, recovered from the annotations rather than assumed: box
   coordinates are `[x0, y0, x1, y1]`, and their maxima are exactly 3840 and 2160 — boxes clipped at
   the image edge. `prepare` asserts this, so a wrong guess fails loudly rather than silently
   rasterising into the wrong canvas. The 517 MB of JPEGs in the repo's `imgs/` are therefore not
   needed: box IoU depends only on coordinates.

6. **Coordinates parse in two formats.** Most rows are `[2944, 1229, 3263, 1812]`; five are numpy
   scientific notation, `[3.220e+03 3.000e+00 ...]`. Both are read; a regex that misses the exponent
   silently truncates those five, so the pattern below matches it explicitly.

7. **One class-agnostic task.** The 34 diagnostic categories are recorded in the CSV but not split
   into tasks: the box stream is scored by box IoU and box F1@0.5, which are spatial, and
   most of the 34 classes are too sparse to floor. Category disagreement is a different measurement
   and is not part of the layer.

8. **`GroundTruth.csv` is deliberately not loaded as a rater.** It is a consolidation by a *separate*
   senior pathologist who reviewed all 14,579 annotations — class P6 (a single consensus reference), ineligible. It is
   scientifically interesting precisely because it is independent of the four raters (unlike CTC's
   majority vote or Gleason's STAPLE, both voted from their own annotators), but that comparison is not
   part of the layer.

Licence: **unstated**. The paper is CC BY 4.0 (ACM, doi:10.1145/3774904.3792891) but the repository
carries no LICENSE file and the README states no terms, so the dataset's own licence is unconfirmed
and its per-image and per-rater rows are withheld at release until the authors state one.
"""
from __future__ import annotations

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

DATASET = "cytocrowd"
LICENCE = "unstated"
TASK = "cells"

#: note 5 — asserted in prepare(), not assumed
IMAGE_SHAPE = (2160, 3840)

#: load-integrity checks, from the released CSVs
EXPECTED_IMAGES = 446
EXPECTED_RATERS = 4
EXPECTED_BOXES = 14579

#: note 6 — plain integers, floats, and numpy scientific notation
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


def parse_box(value: str) -> tuple[int, int, int, int]:
    """`[x0, y0, x1, y1]` in either of the two released formats (note 6)."""
    numbers = [float(n) for n in _NUMBER.findall(str(value))]
    if len(numbers) != 4:
        raise ValueError(f"expected 4 box coordinates, got {len(numbers)}: {value!r}")
    x0, y0, x1, y1 = numbers
    return int(x0), int(y0), int(x1), int(y1)


def safe_stem(image_name: str) -> str:
    """A filesystem-safe directory name. `image_id` itself stays the verbatim released identifier
    — this only names the mask on disk."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(image_name))[:120]


def load(root: Path | str) -> list[Annotation]:
    """Emit every CytoCrowd row. `root` holds the prepared masks, normally data/gtu-raw/cytocrowd."""
    root = Path(root)
    index_path = root / "index.tsv"
    if not index_path.exists():
        raise FileNotFoundError(f"{index_path} missing — run `python code/loaders/cytocrowd.py --prepare`")

    rows = []
    for line in index_path.read_text(encoding="utf-8").splitlines()[1:]:
        image_id, rater_id, relative = line.split("\t")
        rows.append(
            Annotation(
                dataset=DATASET,
                task=TASK,
                image_id=image_id,  # verbatim released name; the release key
                rater_id=rater_id,  # pseudonymised (note 4)
                mask_path=root / relative,
                spacing=None,  # no physical pixel size ships; distances in pixels
                modality=Modality.CYTOLOGY,
                annotation_type=AnnotationType.BOX,  # box stream
                rater_provenance=Provenance.P1,  # note 1
                expertise=Expertise.EXPERT,
                licence=LICENCE,
                source_type=SourceType.RECOMPUTED,
                seeded=False,  # note 2
                cluster_id=None,
                label_value=None,
            )
        )
    return rows


def prepare(csv_dir: Path | str, out: Path | str) -> None:
    """Rasterise each rater's boxes into a uint16 instance PNG, and write the index the loader reads."""
    import numpy as np
    import pandas as pd
    from skimage import io as skimage_io

    csv_dir, out = Path(csv_dir), Path(out)
    frame = pd.read_csv(csv_dir / "WorkerData.csv")
    if len(frame) != EXPECTED_BOXES:
        raise ValueError(f"expected {EXPECTED_BOXES} worker boxes, found {len(frame)}")

    boxes = [parse_box(v) for v in frame["worker_bbox"]]
    xs = [max(b[0], b[2]) for b in boxes]
    ys = [max(b[1], b[3]) for b in boxes]
    height, width = IMAGE_SHAPE
    if max(xs) != width or max(ys) != height:
        raise ValueError(  # note 5: fail loudly rather than rasterise into a wrong canvas
            f"box maxima {max(xs)}x{max(ys)} do not match the assumed {width}x{height} image"
        )
    frame = frame.assign(_box=boxes)

    # note 4: pseudonymise by sorted author_id, and never carry the name column further
    rater_of = {aid: f"r{i}" for i, aid in enumerate(sorted(frame["author_id"].unique()), start=1)}
    if len(rater_of) != EXPECTED_RATERS:
        raise ValueError(f"expected {EXPECTED_RATERS} raters, found {sorted(rater_of)}")

    images = sorted(frame["img_name"].unique())
    if len(images) != EXPECTED_IMAGES:
        raise ValueError(f"expected {EXPECTED_IMAGES} images, found {len(images)}")

    out.mkdir(parents=True, exist_ok=True)
    index = ["image_id\trater_id\tmask_path"]
    for position, image_name in enumerate(images, 1):
        stem = safe_stem(image_name)
        (out / stem).mkdir(parents=True, exist_ok=True)
        block = frame[frame["img_name"] == image_name]
        for author_id, rater_id in rater_of.items():
            target = out / stem / f"{rater_id}.png"
            index.append(f"{image_name}\t{rater_id}\t{stem}/{rater_id}.png")
            if target.exists():
                continue
            mask = np.zeros(IMAGE_SHAPE, dtype=np.uint16)
            for label, (x0, y0, x1, y1) in enumerate(block[block["author_id"] == author_id]["_box"], 1):
                # [x0, y0, x1, y1] -> rows are y, columns are x; clip the edge-clipped boxes
                r0, r1 = max(0, y0), min(height - 1, y1)
                c0, c1 = max(0, x0), min(width - 1, x1)
                mask[r0 : r1 + 1, c0 : c1 + 1] = label  # overlaps resolve last-wins, as MultiOrg's do
            skimage_io.imsave(target, mask, check_contrast=False)
        if position % 50 == 0:
            print(f"  prepared {position}/{len(images)} images", flush=True)

    (out / "index.tsv").write_text("\n".join(index), encoding="utf-8")
    print(f"wrote {len(index) - 1} instance masks to {out}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    repo = Path(__file__).resolve().parents[2]
    out = Path(args[0]) if args else repo / "data/gtu-raw/cytocrowd"

    if "--prepare" in sys.argv:
        prepare(repo / "data/CytoCrowd", out)

    loaded = load(out)
    print(f"{len(loaded)} rows from {out}")
    print(f"  images={len({r.image_id for r in loaded})}  raters={sorted({r.rater_id for r in loaded})}")
    print(
        f"  task={TASK}  modality={loaded[0].modality.value}  "
        f"annotation_type={loaded[0].annotation_type.value} (box stream)  "
        f"provenance={sorted({r.rater_provenance.value for r in loaded})}  "
        f"floor_eligible={any(r.floor_eligible() for r in loaded)}  licence={loaded[0].licence}"
    )
