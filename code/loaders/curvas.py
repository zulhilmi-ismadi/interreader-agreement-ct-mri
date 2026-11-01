"""CURVAS 2024 loader. One of the two leaderboard pairings, scored on DSC: three radiologists
annotated pancreas, kidney and liver on abdominal CT, and the challenge publishes per-algorithm DSC
on the test split, so leaderboard gaps can be set against a floor recomputed from the same cases.

Layout: <split>/<case>/annotation_{1,2,3}.nii.gz, one multi-class label map per rater.

Notes that matter:

1. **Provenance is P1 with `seeded=True`.** The raters corrected a TotalSegmentator pre-segmentation
   independently. Raters who corrected an algorithmic pre-segmentation stay in the floor, and the
   seeding is a sensitivity stratum (seeded against unseeded floors), not an exclusion - but every CURVAS number carries the flag, because three raters
   editing one algorithmic starting point are not three raters drawing from scratch.

2. **One file holds all three organs**, and each task is one binary structure, so it is split into three tasks via
   `label_value`. Label order confirmed by volume on every case inspected: 1 pancreas, 2 kidney,
   3 liver (pancreas is by far the smallest, liver by far the largest).

3. **The label maps need rounding, not casting.** They are int16 spanning the whole dtype range with
   `scl_slope = nan`, so nibabel returns 0.9999999997 rather than 1; `gtu.io.load_mask` rounds within
   tolerance. A plain `== 1` on the raw scaled array matches nothing at all.

4. **All 90 cases load once every archive is extracted.** `training_set.zip` (20 cases) is Deflate64,
   which Python's zipfile, Info-ZIP unzip, .NET and Windows' bundled libarchive `tar` all refuse. 7-Zip
   reads it, and so does the Windows shell's own zip handler (`Shell.Application.CopyHere`); check the
   result file by file (60 annotations, labels a subset of {0,1,2,3}). If the split is not extracted,
   the loader silently returns the 70-case subset.

Licence: CC BY 4.0 (Zenodo).
"""

from __future__ import annotations

import sys
from pathlib import Path

import nibabel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gtu.schema import (
    Annotation,
    AnnotationType,
    Expertise,
    Modality,
    Provenance,
    SourceType,
)

DATASET = "curvas"
LICENCE = "CC-BY-4.0"
N_RATERS = 3

#: label value in the multi-class map -> task name (one binary task per organ)
ORGANS = {1: "pancreas", 2: "kidney", 3: "liver"}

SPLITS = ("training_set", "validation_set", "testing_set")


def load(root: Path | str) -> list[Annotation]:
    """Emit every CURVAS row. `root` holds the split folders, normally data/gtu-raw/curvas."""
    root = Path(root)
    rows = []
    for split in SPLITS:
        split_root = root / split
        if not split_root.is_dir():
            continue  # a split that has not been extracted (training_set is Deflate64, note 4)
        for case_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
            paths = sorted(case_dir.glob("annotation_*.nii.gz"))
            if len(paths) != N_RATERS:
                raise ValueError(f"{case_dir}: expected {N_RATERS} annotations, found {len(paths)}")
            spacing = tuple(
                float(z) for z in nibabel.load(paths[0]).header.get_zooms()[:3]
            )
            for path in paths:
                rater = path.stem.replace(".nii", "").replace("annotation_", "r")
                for label, organ in ORGANS.items():
                    rows.append(
                        Annotation(
                            dataset=DATASET,
                            task=organ,
                            image_id=case_dir.name,
                            rater_id=rater,
                            mask_path=path,
                            spacing=spacing,
                            modality=Modality.CT,
                            annotation_type=AnnotationType.PIXEL,
                            rater_provenance=Provenance.P1,
                            expertise=Expertise.EXPERT,
                            licence=LICENCE,
                            source_type=SourceType.RECOMPUTED,
                            seeded=True,  # raters corrected a TotalSegmentator pre-segmentation
                            cluster_id=None,  # one scan per patient
                            label_value=label,
                        )
                    )
    return rows


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    root = Path(args[0]) if args else Path(__file__).resolve().parents[2] / "data/gtu-raw/curvas"
    loaded = load(root)
    cases = {r.image_id for r in loaded}
    print(f"{len(loaded)} rows from {root}")
    print(f"  cases={len(cases)}  tasks={sorted({r.task for r in loaded})}  raters={sorted({r.rater_id for r in loaded})}")
    for split in SPLITS:
        present = sorted(p.name for p in (root / split).iterdir() if p.is_dir()) if (root / split).is_dir() else []
        print(f"  {split}: {len(present)} cases")
    print(f"  seeded={loaded[0].seeded}  licence={loaded[0].licence}  provenance={loaded[0].rater_provenance.value}")
