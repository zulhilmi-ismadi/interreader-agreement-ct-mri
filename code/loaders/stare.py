"""STARE loader (hoover-2000). Retinal fundus, vessel networks hand-labelled by two observers.

Included for the **thin-structure** question. Retinal vessels are the extreme case of the
C3RO finding that thin structures floor near 0.2-0.3, and STARE is the canonical two-observer
instance of it.

Source: https://cecas.clemson.edu/~ahoover/stare/probing/ — `labels-ah.tar` (Adam Hoover) and
`labels-vk.tar` (Valentina Kouznetsova), 20 images each, downloaded without registration.

Notes that matter:

1. **Two observers, complete overlap.** Both label all 20 images, so every image forms exactly one
   pair. With 2 raters there is no between-rater spread and no View B (which needs
   three raters), so this source contributes a floor and an image bootstrap and nothing else.

2. **The observers are named, and the dataset's own identifiers are their initials** (`ah`, `vk`).
   Unlike CytoCrowd, where we pseudonymised because the names sat in a data column, these initials
   *are* the published convention the literature refers to, so they are kept verbatim as the release
   key. Per-rater statistics here are therefore attributable to named observers.

3. **`vk` marks systematically more vessel than `ah`** — 39,171 vs 33,799 positive pixels on im0001,
   and the pattern holds across the set. That is a calibration difference, not noise, and it is the
   kind of disagreement a Dice floor summarises but does not explain.

4. **Masks ship as gzipped PPM** (`im<NNNN>.ah.ppm.gz`), 605x700, values {0, 255}. `--prepare` writes
   them as 0/1 single-channel PNGs under `data/gtu-raw/stare/`, because `gtu.io.load_mask` reads no
   PPM and a 0/255 mask invites a label-vs-binary ambiguity downstream.

Licence: **unstated**. The STARE pages state no terms of use, so its rows are computed and cited, and its
per-image and per-rater rows are withheld from release — the same treatment as Gleason 2019 and CytoCrowd.
"""
from __future__ import annotations

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

DATASET = "stare"
LICENCE = "unstated"
TASK = "vessel"

#: the dataset's own observer identifiers (note 2)
RATERS = ("ah", "vk")
EXPECTED_IMAGES = 20


def prepare(source: Path | str, out: Path | str) -> None:
    """Gunzip each PPM into a 0/1 single-channel PNG (note 4)."""
    import gzip
    import io

    import numpy as np
    from PIL import Image

    source, out = Path(source), Path(out)
    written = 0
    for rater in RATERS:
        directory = source / f"labels-{rater}"
        if not directory.is_dir():
            raise FileNotFoundError(f"{directory} missing — extract labels-{rater}.tar")
        target_dir = out / rater
        target_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(directory.glob(f"*.{rater}.ppm.gz")):
            image_id = path.name.split(".")[0]
            target = target_dir / f"{image_id}.png"
            if target.exists():
                continue
            with gzip.open(path, "rb") as handle:
                array = np.array(Image.open(io.BytesIO(handle.read())))
            if array.ndim == 3:
                array = array[..., 0]
            values = set(np.unique(array).tolist())
            if not values <= {0, 255}:
                raise ValueError(f"{path.name}: expected a binary mask, found values {sorted(values)}")
            Image.fromarray((array > 0).astype("uint8"), mode="L").save(target)
            written += 1
    print(f"wrote {written} binary masks to {out}")


def load(root: Path | str) -> list[Annotation]:
    """Emit every STARE row. `root` holds the prepared masks, normally data/gtu-raw/stare."""
    root = Path(root)
    per_rater: dict[str, dict[str, Path]] = {}
    for rater in RATERS:
        directory = root / rater
        if not directory.is_dir():
            raise FileNotFoundError(
                f"{directory} missing — run `python code/loaders/stare.py --prepare`"
            )
        per_rater[rater] = {path.stem: path for path in sorted(directory.glob("*.png"))}

    shared = sorted(set.intersection(*(set(v) for v in per_rater.values())))
    if len(shared) != EXPECTED_IMAGES:
        raise ValueError(f"expected {EXPECTED_IMAGES} images labelled by both, found {len(shared)}")

    return [
        Annotation(
            dataset=DATASET,
            task=TASK,
            image_id=image_id,
            rater_id=rater,
            mask_path=per_rater[rater][image_id],
            spacing=None,  # no physical pixel size ships; distances in pixels
            modality=Modality.FUNDUS,
            annotation_type=AnnotationType.PIXEL,
            rater_provenance=Provenance.P1,
            expertise=Expertise.EXPERT,
            licence=LICENCE,
            source_type=SourceType.RECOMPUTED,
            seeded=False,
            cluster_id=None,
            label_value=None,
        )
        for image_id in shared
        for rater in RATERS
    ]


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    repo = Path(__file__).resolve().parents[2]
    root = Path(args[0]) if args else repo / "data/gtu-raw/stare"

    if "--prepare" in sys.argv:
        prepare(repo / "data/STARE", root)

    loaded = load(root)
    print(f"{len(loaded)} rows from {root}")
    print(f"  images={len({r.image_id for r in loaded})}  raters={sorted({r.rater_id for r in loaded})}")
    print(
        f"  modality={loaded[0].modality.value}  annotation_type={loaded[0].annotation_type.value}  "
        f"provenance={loaded[0].rater_provenance.value}  licence={loaded[0].licence}"
    )
