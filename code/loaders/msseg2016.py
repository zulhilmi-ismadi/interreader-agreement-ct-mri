"""MSSEG-2016 loader. Seven raters delineated multiple sclerosis lesions on
the 3D FLAIR of 53 patients from four centres (Commowick et al. 2021, NeuroImage 244:118589). One task,
`ms_lesion`; 53 images; 371 per-rater masks.

Layout: {Training,Testing}/Center_NN/Patient_NN/Masks/{ManualSegmentation_1..7, Consensus,
Brain_Mask}.nii.gz. Only the `Masks/` folders are extracted; the MRI volumes stay in the archives because
no floor reads them.

Notes that matter:

1. **Both splits carry all seven raters**, so both enter. One scan per patient: an image bootstrap, no
   cluster.

2. **Rater identity is global.** The segmentation numbers are "consistent throughout the patients"
   (Commowick 2021 section 3): Lyon 1, 2, 4, 5; Rennes 3, 6; Bordeaux 7. So the per-rater table, the drop-any-one-rater
   sensitivity analysis and the rater bootstrap are all meaningful here, unlike QUBIQ.

3. **P1, unseeded, `expert`.** Manual delineation in ITK-SNAP on the raw FLAIR, "each case was segmented
   in isolation of the other cases and raters" (section 2.2). The source calls the seven "experts" and
   also "trained junior radiologists"; expertise is always the source's own term, and the paper states both.

4. **`Consensus.nii.gz` (LOP-STAPLE) and `Brain_Mask.nii.gz` are not raters** and are never loaded.

5. **Spacing is physical**, read per patient from the header (mm), so the volumetric tolerance grid
   applies. All seven masks share one grid and affine per patient (checked on every patient). Two masks are
   empty (raters 4 and 6 on Center_07/Patient_08, the patient with no consensus lesion); the empty-mask
   rule scores them like any other.

Licence: the OFSEP data access agreement (March 2021), a copy beside the masks. No redistribution;
derivatives only at group level. Per-image and per-rater rows are withheld from release; task-level rows
ship, as group-level derivatives.
"""

from __future__ import annotations

import re
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

DATASET = "msseg2016"
LICENCE = "LicenseRef-OFSEP-MSSEG2016-DUA"
TASK = "ms_lesion"
N_RATERS = 7
SPLITS = ("Training", "Testing")

#: load-integrity check: counts in the released files
EXPECTED_IMAGES = 53
EXPECTED_MASKS = 371

_MASK = re.compile(r"^ManualSegmentation_(?P<rater>[1-7])\.nii\.gz$")


def load(root: Path | str) -> list[Annotation]:
    """Emit every MSSEG-2016 row. `root` holds the split folders, normally data/gtu-raw/msseg2016."""
    root = Path(root)
    rows = []
    for split in SPLITS:
        for patient_dir in sorted((root / split).glob("Center_*/Patient_*")):
            masks = patient_dir / "Masks"
            paths = sorted(masks.glob("ManualSegmentation_*.nii.gz"))
            if len(paths) != N_RATERS:
                raise ValueError(f"{masks}: expected {N_RATERS} rater masks, found {len(paths)}")
            spacing = tuple(float(z) for z in nibabel.load(paths[0]).header.get_zooms()[:3])
            for path in paths:
                match = _MASK.match(path.name)
                if match is None:
                    raise ValueError(f"unexpected MSSEG filename: {path}")
                rows.append(
                    Annotation(
                        dataset=DATASET,
                        task=TASK,
                        # Patient numbers repeat across centres and across the two splits (53 distinct
                        # patients); the source's path is the identifier, so the id keeps all of it.
                        image_id=f"{split}_{patient_dir.parent.name}_{patient_dir.name}",
                        rater_id=f"expert{match['rater']}",  # global, note 2
                        mask_path=path,
                        spacing=spacing,
                        modality=Modality.MRI,
                        annotation_type=AnnotationType.PIXEL,
                        rater_provenance=Provenance.P1,
                        expertise=Expertise.EXPERT,  # the source's term, note 3
                        licence=LICENCE,
                        source_type=SourceType.RECOMPUTED,
                        seeded=False,
                        cluster_id=None,  # one scan per patient
                        label_value=None,  # binary already
                    )
                )
    return rows


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    root = Path(args[0]) if args else Path(__file__).resolve().parents[2] / "data/gtu-raw/msseg2016"
    loaded = load(root)
    images = {r.image_id for r in loaded}
    print(f"{len(loaded)} rows from {root}")
    print(f"  images={len(images)}  raters={sorted({r.rater_id for r in loaded})}")
    if len(loaded) != EXPECTED_MASKS or len(images) != EXPECTED_IMAGES:
        raise SystemExit(f"expected {EXPECTED_MASKS} masks on {EXPECTED_IMAGES} images")
