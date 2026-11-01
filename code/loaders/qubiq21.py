"""QUBIQ 2021 loader (training and validation splits). Six structures, nine (structure, task) units,
2-7 raters per task, 240 case folders (239 with masks), 1,474 per-rater masks.

Layout: <structure>/{Training,Validation}/<case>/{image.nii.gz, task<NN>_seg<NN>.nii.gz}. The training
archive ships pancreas without the `Training/` level; extraction puts it there.

**Use the training and validation splits, not the test split.** `QUBIQ21_test.zip` ships `image.nii.gz`
and nothing else - challenge test labels are withheld - so it yields no floor at all. Notes 4 and 6
describe the validation split.

Notes that matter:

1. **brain-tumor is P5 and never enters a floor.** Its three label sets are **1 human + 2
   algorithm-fused**, as the challenge records them, which fails the independence test. Algorithm-fused
   labels presented as raters are class P5, reported as provenance information only and never in a
   floor. All three brain-tumor tasks are loaded and marked, not dropped - an exclusion with a stated
   reason is evidence about the source, and dropping it silently would hide that.

2. **Task means QUBIQ's own task index, not a structure.** `prostate` ships task01 and task02 (two
   different structures on one image), `pancreas` likewise, `brain-tumor` three. Task names are
   `<structure>_task<NN>` so the pairing stays inside the unit QUBIQ actually annotated. Nothing is
   pooled across task indices.

3. **Rater identity is per-case, not global** - the same caveat RIGA carries. `seg03` on case49 and
   `seg03` on case55 are both "the third label set released for this case"; nothing in the release says
   they are the same person. The pairwise floor does not depend on it (it needs only that the N sets on
   one case come from N different raters), but the per-rater table, the rater bootstrap and the
   drop-one-rater sensitivity analysis all assume stable ids, so nothing is reported per rater for
   QUBIQ.

4. **22 of the 235 masks are entirely empty** - a rater who marked nothing on that case. That is data,
   not corruption: the empty-mask policy scores both-empty pairs as DSC = IoU = NSD = 1 and HD95 = 0,
   and one-empty pairs as 0 with HD95 = NaN, and both counts are reported per task. Read the task
   floors for the small structures with those counts beside them.

5. Panel sizes are uneven across structures (2 to 7), so the sparse-coverage rule applies: pairs form
   only among raters present on that case.

6. **The headers carry no calibration and the masks are not all the same rank.** Checked across all 235
   masks: every zoom is exactly 1.0, and 83 masks are 2D (brain-growth, brain-tumor, kidney) against
   152 3D (pancreas, pancreatic-lesion, prostate). So `spacing` is `None` - a 3-tuple of ones would be
   a fiction, and would also fail on the 2D masks ("sequence argument must have length equal to
   input rank", because they have rank 2). **Consequence:** QUBIQ's HD95 and NSD are in *voxels*, not
   millimetres, even though the modality column says MRI and CT. They are not comparable with the
   calibrated volumetric sources (PDACVI, CURVAS, LiverHccSeg, C3RO), whose distances are physical,
   and every QUBIQ boundary number is reported with that unit.

7. **Pancreas and pancreatic-lesion cases come in pairs** (`case1-1`, `case1-2`), in both splits: two
   different volumes that may be one patient. Where images may nest in a higher unit the bootstrap
   resamples that unit, so `cluster_id` is the shared stem (`case1`). `brain-growth/Training/case13` ships an image
   and no mask and so emits no row.

Licence: **unstated**. The challenge is account-gated and no licence ships in the archive. Computed,
withheld from release while the licence is unstated.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gtu.schema import (
    UNSTATED_LICENCE,
    Annotation,
    AnnotationType,
    Expertise,
    Modality,
    Provenance,
    SourceType,
)

DATASET = "qubiq21"
LICENCE = UNSTATED_LICENCE

#: structure -> modality. QUBIQ 2021's imaging per structure.
MODALITIES = {
    "brain-growth": Modality.MRI,
    "brain-tumor": Modality.MRI,
    "prostate": Modality.MRI,
    "kidney": Modality.CT,
    "pancreas": Modality.CT,
    "pancreatic-lesion": Modality.CT,
}

#: Structures whose label sets are not independent humans. See note 1 - do not add to this without a
#: source that says so, and do not remove `brain-tumor` without one either.
FUSED_STRUCTURES = frozenset({"brain-tumor"})

#: load-integrity check: 235 masks in validation_data_qubiq2021_QC.zip, plus 1,239 in
#: training_data_v3_QC.zip
EXPECTED_MASKS = 1474
#: 240 case folders across the six structures, but only 113 distinct case names - QUBIQ numbers cases
#: within a structure, so `case04` exists under more than one. Grouping is by (dataset, task,
#: image_id) and `task` carries the structure, so they never collide; do not "fix" this by minting
#: composite ids, which would break the release key (the source's own `image_id`, verbatim).
EXPECTED_CASE_FOLDERS = 240
EXPECTED_DISTINCT_IMAGE_IDS = 113

#: `case1-1` -> `case1`: the pair stem used as the cluster (note 7)
_PAIRED = re.compile(r"^(?P<stem>case\d+)-\d+$")

_MASK = re.compile(r"^task(?P<task>\d+)_seg(?P<seg>\d+)\.nii\.gz$")


def load(root: Path | str) -> list[Annotation]:
    """Emit every QUBIQ 2021 validation row. `root` holds the structure folders, normally
    data/gtu-raw/qubiq21."""
    root = Path(root)
    rows = []
    for structure_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        structure = structure_dir.name
        if structure not in MODALITIES:
            raise ValueError(f"unknown QUBIQ structure {structure!r}; add it to MODALITIES")
        provenance = (
            Provenance.P5 if structure in FUSED_STRUCTURES else Provenance.P1
        )
        for case_dir in sorted(_case_dirs(structure_dir)):
            for path in sorted(case_dir.glob("task*_seg*.nii.gz")):
                match = _MASK.match(path.name)
                if match is None:
                    raise ValueError(f"unexpected QUBIQ filename: {path}")
                rows.append(
                    Annotation(
                        dataset=DATASET,
                        task=f"{structure}_task{match['task']}",
                        image_id=case_dir.name,  # unique within a structure, and task carries it
                        rater_id=f"seg{match['seg']}",  # per-case identity only, note 3
                        mask_path=path,
                        spacing=None,  # headers are uncalibrated, note 6
                        modality=MODALITIES[structure],
                        annotation_type=AnnotationType.PIXEL,
                        rater_provenance=provenance,
                        expertise=Expertise.UNKNOWN,  # not stated per rater by the challenge
                        licence=LICENCE,
                        source_type=SourceType.RECOMPUTED,
                        seeded=False,
                        cluster_id=_cluster(case_dir.name),  # note 7
                        label_value=None,  # already binary per task
                    )
                )
    return rows


def _cluster(case: str) -> str | None:
    """The pair stem for paired cases (note 7); None, i.e. the image is its own unit, otherwise."""
    match = _PAIRED.match(case)
    return match["stem"] if match else None


def _case_dirs(structure_dir: Path) -> list[Path]:
    """Case folders, under the split folders (`Training`, `Validation`)."""
    cases = []
    for split in sorted(p for p in structure_dir.iterdir() if p.is_dir()):
        cases += [p for p in split.iterdir() if p.is_dir()]
    return cases


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    root = (
        Path(args[0])
        if args
        else Path(__file__).resolve().parents[2] / "data/gtu-raw/qubiq21"
    )
    loaded = load(root)
    print(f"{len(loaded)} rows from {root}")
    if len(loaded) != EXPECTED_MASKS:
        raise SystemExit(f"expected {EXPECTED_MASKS} masks, found {len(loaded)}")
    print(f"  cases={len({r.image_id for r in loaded})}  tasks={len({r.task for r in loaded})}")
    for task in sorted({r.task for r in loaded}):
        task_rows = [r for r in loaded if r.task == task]
        raters = sorted({r.rater_id for r in task_rows})
        eligible = any(r.floor_eligible() for r in task_rows)
        note = "" if eligible else "   <- P5, provenance only (note 1)"
        print(
            f"  {task:<26} images={len({r.image_id for r in task_rows}):>3} "
            f"raters={len(raters)}  {task_rows[0].modality.value:<4} "
            f"floor_eligible={eligible}{note}"
        )
    if "--verify" in sys.argv:
        import numpy as np

        from gtu.io import load_mask

        empty = [r for r in loaded if not load_mask(r.mask_path).any()]
        print(f"\nentirely empty masks: {len(empty)} of {len(loaded)} (note 4)")
        for row in empty[:10]:
            print(f"  {row.task}/{row.image_id}/{row.rater_id}")
        if len(empty) > 10:
            print(f"  ... and {len(empty) - 10} more")
