"""Mesmer 5-annotator loader (greenwald-2022-mesmer, TissueNet side study).

Four multiplex-microscopy tissues, five annotators each, from the paper's public
`publication_data_folder.zip` -> `Human_agreement/`. Small but the only dataset analysed here with five
independent raters on every image, so it is what exercises View B (which needs >= 3 raters).

Three load-time traps, all confirmed in the released files:
  1. In cHL_Point8908 christine's file is `christine_labels_new.tiff`, not `christine_labels.tiff`.
  2. cHL_Point8908 also carries `rgb_bri_labels.tif` — a 3-channel RGB visualisation, NOT a mask.
  3. `geneva_labels.tiff` is float32 in DCIS_2328 and float64 in cHL_Point8908 (int elsewhere), and
     `bri_labels.tiff` is int32 in P101_T3_T4_Point2. gtu.io.load_mask normalises integral floats.

Each tissue folder also holds `segmentation_label.tiff`, which is Mesmer's own prediction, not a
rater — excluded here, and it must never enter a floor.

Licence: UNSTATED. The zip carries no LICENSE or README (full 28k-entry listing checked). Rows are
computable, but per-image and per-rater rows are withheld from release unless the Van Valen lab states terms that permit it.

Only 4 images, too few for an image bootstrap alone: report the rater bootstrap alongside the image bootstrap
and quote the wider interval.
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
    UNSTATED_LICENCE,
)

DATASET = "mesmer"
TASK = "cells"
ANNOTATORS = ("bri", "christine", "geneva", "ke", "noah")
TISSUES = ("DCIS_2328", "Eliot_Point17", "P101_T3_T4_Point2", "cHL_Point8908")

#: Mesmer's own prediction, not a rater.
PREDICTION_FILENAME = "segmentation_label.tiff"


def load(root: Path | str) -> list[Annotation]:
    """Emit every Mesmer annotation row. `root` is normally data/gtu-raw/mesmer."""
    root = Path(root)
    rows = []
    for tissue in TISSUES:
        directory = root / tissue / "annotations"
        if not directory.is_dir():
            raise FileNotFoundError(f"missing Mesmer tissue: {directory}")
        for annotator in ANNOTATORS:
            rows.append(
                Annotation(
                    dataset=DATASET,
                    task=TASK,
                    image_id=tissue,
                    rater_id=annotator,
                    mask_path=_mask_path(directory, annotator),
                    spacing=None,
                    modality=Modality.MICROSCOPY,
                    annotation_type=AnnotationType.INSTANCE,
                    rater_provenance=Provenance.P1,
                    expertise=Expertise.EXPERT,
                    licence=UNSTATED_LICENCE,
                    source_type=SourceType.RECOMPUTED,
                    cluster_id=None,  # four unrelated tissues; no higher unit to resample
                )
            )
    return rows


def _mask_path(directory: Path, annotator: str) -> Path:
    """Resolve one annotator's label image, tolerating the `_new` suffix and never matching the
    `rgb_` visualisation."""
    for name in (f"{annotator}_labels.tiff", f"{annotator}_labels_new.tiff"):
        candidate = directory / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no label mask for annotator {annotator!r} in {directory}")


if __name__ == "__main__":
    from gtu.io import load_mask

    args = sys.argv[1:]
    root = Path(args[0]) if args else Path(__file__).resolve().parents[2] / "data/gtu-raw/mesmer"
    rows = load(root)
    print(f"{len(rows)} rows from {root}")
    for tissue in TISSUES:
        tissue_rows = [r for r in rows if r.image_id == tissue]
        summary = []
        for row in tissue_rows:
            mask = load_mask(row.mask_path)
            summary.append(f"{row.rater_id}={len(set(mask.ravel().tolist())) - 1}")
        shapes = {load_mask(r.mask_path).shape for r in tissue_rows}
        print(f"  {tissue}: {len(tissue_rows)} raters, shape={shapes.pop()}, cells " + " ".join(summary))
