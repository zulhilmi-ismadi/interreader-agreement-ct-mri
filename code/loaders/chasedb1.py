"""CHASE_DB1 loader (Fraz et al. 2012). Retinal fundus, vessel networks hand-labelled by two observers.

The second two-observer retinal vessel source beside STARE, and the first one whose licence permits
releasing its derived rows.

Source: Kingston University's dataset record, "CHASE_DB1 retinal vessel reference dataset"
(doi:10.1109/TBME.2012.2205687 for the paper). The archive holds, for each of 28 images,
`Image_NNX.jpg` (the fundus photograph), `Image_NNX_1stHO.png` and `Image_NNX_2ndHO.png` (the first
and second human observer's vessel masks), where NN is the child (01-14) and X the eye (L or R).

Notes that matter:

1. **Two observers, complete overlap.** Both label all 28 images, so every image forms exactly one
   pair. As for STARE: no between-rater spread, so the interval comes from the image bootstrap only; no View B
   (it needs three raters); no reference substitution spread (it needs three raters).

2. **Both eyes of one child are not independent.** The bootstrap resamples the child (a cluster
   bootstrap), so both eyes of child NN share `cluster_id = "child_NN"`.

3. **Masks are 8-bit PNGs.** They are binarised at > 0 by `gtu.io.load_mask`'s usual path; this loader
   checks the values on first use so that an antialiased or 0/255 mask cannot pass silently.

4. **Independence is stated in Fraz et al. 2012** (§IV.A, p. 2542: "two manual segmentations
   available made by two independent human observers", said of DRIVE, STARE and CHASE_DB1
   together). The archive's readme.txt names the two observers but does not say they worked
   independently. Blinding between them is not documented, and the paper says so.

Licence: **CC BY 4.0** (Kingston University dataset record), so its derived rows are releasable.
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

DATASET = "chasedb1"
LICENCE = "CC-BY-4.0"
TASK = "vessel"

#: the dataset's own observer identifiers
RATERS = ("1stHO", "2ndHO")
EXPECTED_IMAGES = 28
NAME = re.compile(r"^Image_(\d{2})([LR])_(1stHO|2ndHO)\.png$")


def _index(root: Path) -> dict[str, dict[str, Path]]:
    """image_id -> rater -> mask path, searching `root` recursively (archives differ in nesting)."""
    found: dict[str, dict[str, Path]] = {}
    for path in sorted(root.rglob("Image_*_*HO.png")):
        match = NAME.match(path.name)
        if not match:
            continue
        child, eye, rater = match.groups()
        found.setdefault(f"Image_{child}{eye}", {})[rater] = path
    return found


def check_masks(root: Path | str) -> None:
    """Confirm every mask is binary before the layer reads it (note 3)."""
    import numpy as np
    from PIL import Image

    for image_id, raters in sorted(_index(Path(root)).items()):
        for rater, path in raters.items():
            array = np.array(Image.open(path))
            if array.ndim == 3:
                array = array[..., 0]
            values = set(np.unique(array).tolist())
            if not values <= {0, 1, 255}:
                raise ValueError(f"{path.name}: expected a binary mask, found {len(values)} values")
    print("all CHASE_DB1 masks are binary")


def load(root: Path | str) -> list[Annotation]:
    """Emit every CHASE_DB1 row. `root` holds the extracted archive, normally data/gtu-raw/chasedb1."""
    root = Path(root)
    index = _index(root)
    shared = sorted(image_id for image_id, raters in index.items() if set(RATERS) <= set(raters))
    if len(shared) != EXPECTED_IMAGES:
        raise ValueError(
            f"expected {EXPECTED_IMAGES} images labelled by both observers under {root}, found {len(shared)}"
        )
    return [
        Annotation(
            dataset=DATASET,
            task=TASK,
            image_id=image_id,
            rater_id=rater,
            mask_path=index[image_id][rater],
            spacing=None,  # no physical pixel size ships; distances in pixels
            modality=Modality.FUNDUS,
            annotation_type=AnnotationType.PIXEL,
            rater_provenance=Provenance.P1,
            expertise=Expertise.EXPERT,
            licence=LICENCE,
            source_type=SourceType.RECOMPUTED,
            seeded=False,
            cluster_id=f"child_{image_id[6:8]}",  # both eyes of one child (note 2)
            label_value=None,
        )
        for image_id in shared
        for rater in RATERS
    ]


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    repo = Path(__file__).resolve().parents[2]
    root = Path(args[0]) if args else repo / "data/gtu-raw/chasedb1"
    check_masks(root)
    loaded = load(root)
    print(f"{len(loaded)} rows from {root}")
    print(f"  images={len({r.image_id for r in loaded})}  children={len({r.cluster_id for r in loaded})}  "
          f"raters={sorted({r.rater_id for r in loaded})}")
    print(f"  modality={loaded[0].modality.value}  provenance={loaded[0].rater_provenance.value}  "
          f"licence={loaded[0].licence}")
