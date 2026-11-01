"""CryoNuSeg loader. Template for every other source: resolve files, assert the expected structure
loudly, emit canonical rows in the `gtu.schema.Annotation` data model.

Source (mahbod-2021-cryonuseg): 30 H&E cryosection patches, 30 whole-slide images, 3 per organ across
10 human organs, from TCGA. Annotator 1 is a biologist, Annotator 2 a bioinformatician; both were given
the same background information and written instructions, so both are recorded as `expert` — the source
uses no expert/non-expert tier language, and expertise is never re-assigned by us.
Annotator 1 repeated the whole set after a three-month gap (intra-observer repeat, class P3).

Two tasks, because P3 may not enter a pooled floor and is reported as its own intra-observer row:

  nuclei        annotator1_round1 vs annotator2          both P1 -> inter-observer floor
  nuclei_intra  annotator1_round1 vs annotator1_round2   P3     -> intra-observer row only

Published agreement values to reproduce, from the paper's Tables 7 and 8:
  inter (A2 vs A1r1):  Dice 78.9 +/- 6.5 %, AJI 54.1 +/- 7.3 %, PQ 50.9 +/- 7.3 %
  intra (A1r1 vs A1r2): Dice 83.8 +/- 3.1 %, AJI 62.4 +/- 4.8 %, PQ 57.0 +/- 4.2 %
AJI is not one of the layer's metrics and is not recomputed; Dice and PQ are directly comparable.

Published nuclei counts, used by --verify as a load-integrity check:
  annotator1_round1 7596, annotator1_round2 8044, annotator2 8251.

GOTCHA: each annotator zip ships BOTH `label masks/` and `label masks modify/`. Use **modify** — it is
the corrected set the paper's numbers come from. Checked against the counts above:

  folder                totals (r1 / r2 / a2)      difference from published
  label masks           8488 / 8950 / 9270         +892 / +906 / +1019  (~12 % too many)
  label masks modify    7592 / 8036 / 8242         -4 / -8 / -9         (~0.1 %)

Reading `label masks` would have inflated every CryoNuSeg instance count by about a tenth, silently.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import tifffile

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

DATASET = "cryonuseg"
N_IMAGES = 30

#: the corrected mask set; see the GOTCHA in the module docstring before changing this
MASK_FOLDER = "label_masks_modify"

# Licence: the Kaggle deposit that hosts the data states a licence in its dataset
# metadata -- `"license": {"name": "MIT"}` on ipateam/segmentation-of-nuclei-in-cryosectioned-he-images,
# the host the project's own GitHub README points to. The article's CC BY does not cover the data and
# is not the source of this string. The underlying H&E images derive from TCGA and are governed
# separately; we redistribute only derived statistics keyed by image_id, as for NuCLS.
LICENCE = "MIT"

INTER_TASK = "nuclei"
INTRA_TASK = "nuclei_intra"

#: folder -> (rater_id, task, provenance)
RATER_SETS = {
    "annotator1_r1": ("annotator1_round1", INTER_TASK, Provenance.P1),
    "annotator2": ("annotator2", INTER_TASK, Provenance.P1),
    "annotator1_r2": ("annotator1_round2", INTRA_TASK, Provenance.P3),
}
INTRA_REFERENCE_FOLDER = "annotator1_r1"

PUBLISHED_NUCLEI_COUNTS = {
    "annotator1_round1": 7596,
    "annotator1_round2": 8044,
    "annotator2": 8251,
}

_FILENAME = re.compile(r"^Human_(?P<organ>[A-Za-z]+)_(?P<replicate>\d+)\.tif$")


def load(root: Path | str) -> list[Annotation]:
    """Emit every CryoNuSeg row. `root` holds the three rater-set folders, normally
    data/gtu-raw/cryonuseg."""
    root = Path(root)
    rows = []
    for folder, (rater_id, task, provenance) in RATER_SETS.items():
        for path in _mask_paths(root, folder):
            image_id, organ = _parse(path)
            rows.append(_row(path, image_id, organ, rater_id, task, provenance))
            if folder == INTRA_REFERENCE_FOLDER:
                # round 1 is also the reference the intra-observer repeat is scored against
                rows.append(_row(path, image_id, organ, rater_id, INTRA_TASK, Provenance.P3))
    return rows


def _row(
    path: Path, image_id: str, organ: str, rater_id: str, task: str, provenance: Provenance
) -> Annotation:
    return Annotation(
        dataset=DATASET,
        task=task,
        image_id=image_id,
        rater_id=rater_id,
        mask_path=path,
        spacing=None,  # no physical pixel size ships with the masks; distances stay in pixels
        modality=Modality.HISTOLOGY,
        annotation_type=AnnotationType.INSTANCE,
        rater_provenance=provenance,
        expertise=Expertise.EXPERT,
        licence=LICENCE,
        source_type=SourceType.RECOMPUTED,
        cluster_id=organ,  # 3 patches per organ; the organ is the higher unit the bootstrap resamples
    )


def _mask_paths(root: Path, folder: str) -> list[Path]:
    directory = root / folder / MASK_FOLDER
    if not directory.is_dir():
        raise FileNotFoundError(f"missing CryoNuSeg rater set: {directory}")
    paths = sorted(directory.glob("*.tif"))
    if len(paths) != N_IMAGES:
        raise ValueError(f"{directory}: expected {N_IMAGES} masks, found {len(paths)}")
    return paths


def _parse(path: Path) -> tuple[str, str]:
    match = _FILENAME.match(path.name)
    if match is None:
        raise ValueError(f"unexpected CryoNuSeg filename: {path.name}")
    return path.stem, match["organ"]


def _verify_nuclei_counts(rows: list[Annotation]) -> None:
    """Check the loader reads the intended masks by reproducing the paper's Table 2 nuclei counts.
    This validates file resolution only; it computes no agreement statistic."""
    import numpy as np

    seen: dict[str, set[Path]] = {}
    for row in rows:
        seen.setdefault(row.rater_id, set()).add(row.mask_path)

    print("\nnuclei counts vs mahbod-2021 Table 2:")
    for rater_id, paths in sorted(seen.items()):
        total = sum(len(np.unique(tifffile.imread(p))) - 1 for p in sorted(paths))
        published = PUBLISHED_NUCLEI_COUNTS[rater_id]
        deviation = (total - published) / published
        # 0.5% tolerance: a few instances differ from the paper's count (merged or split labels),
        # which is not the failure this check exists to catch — reading the wrong mask folder is.
        flag = "OK" if abs(deviation) < 0.005 else "MISMATCH - wrong mask folder?"
        print(f"  {rater_id}: {total} vs published {published} ({deviation:+.2%}) {flag}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--verify"]
    root = Path(args[0]) if args else Path(__file__).resolve().parents[2] / "data/gtu-raw/cryonuseg"
    loaded = load(root)
    print(f"{len(loaded)} rows from {root}")
    for task in sorted({r.task for r in loaded}):
        task_rows = [r for r in loaded if r.task == task]
        raters = sorted({r.rater_id for r in task_rows})
        classes = sorted({r.rater_provenance.value for r in task_rows})
        eligible = any(r.floor_eligible() for r in task_rows)
        print(
            f"  {task}: {len(task_rows)} rows, raters={raters}, provenance={classes}, "
            f"images={len({r.image_id for r in task_rows})}, "
            f"clusters={len({r.cluster_id for r in task_rows})}, "
            f"floor_eligible={eligible}, licence={task_rows[0].licence}"
        )
    if "--verify" in sys.argv:
        _verify_nuclei_counts(loaded)
