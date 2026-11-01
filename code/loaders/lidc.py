"""LIDC-IDRI loader, via the Kohl et al. probabilistic-U-Net 2D crops. Four radiologists, 12,816
180x180 crops over the train/val/test splits, one mask per reader per crop.

Layout: <split>/images/<patient>/z<Z>_c<C>.png and <split>/gt/<patient>/z<Z>_c<C>_l<L>.png,
with L in 0..3. Every crop carries exactly four masks (35,372/8,843, 7,972/1,993 and 7,920/1,980 --
4.0 in every split).

**This source is P4 and never enters a pooled floor.** Armato 2011 states it directly: the four
radiologists read in two phases, a blinded independent read and then an **unblinded** read in which
each sees the others' anonymised marks and may revise, delete, recategorise or add. Only the
post-unblinded marks are released, and the paper says *"The marks provided in the LIDC/IDRI Database,
therefore, are correlated and do not represent the independent interpretations of the radiologists."*
Class P4 (sequential unblinded revision): provenance row only.

That is the entire reason to compute it. LIDC is the cleanest example in the corpus of agreement
numbers that are routinely quoted as inter-observer variability but are correlated by construction,
and quantifying its apparent agreement **next to** the genuinely independent sources is the point of
including it. The number is not a floor and must never be printed in the same column as one.

Notes that matter:

1. **`cluster_id` is the patient and it matters more here than anywhere else.** 12,816 crops come from
   roughly a thousand patients, many crops per patient, often consecutive slices of the same nodule.
   Treating them as independent images would understate every interval by a large factor.
   The cluster bootstrap, which resamples patients, is what makes this source reportable at all.

2. **Image ids are minted**, like NuCLS's. The release keys a crop by its path, not by an identifier
   inside the file, so the id is `<patient>__<crop stem>` (e.g. `LIDC-IDRI-0007__z-102.0_c0`). The
   patient and the z-position are both recoverable from it by construction.

3. **`l0..l3` is a slot, not a person.** Armato 2011 anonymises readers per scan and the Kohl
   preprocessing preserves only the order. Reader `l2` on one patient is not reader `l2` on another.
   The pairwise comparison does not depend on it, but the per-rater table, the rater bootstrap
   and the drop-any-one-rater sensitivity analysis all do -- do not report LIDC per rater.

4. **Empty masks are expected and are data.** A reader who did not mark a nodule on a crop contributes
   an empty mask, which is exactly how the source encodes disagreement about existence.
   The empty-mask rule applies: both empty -> 1, one empty -> 0 with HD95 NaN, both counted and reported.
   For LIDC that count is itself a result, since existence disagreement is most of what P4 preserves.

5. **No spacing.** The crops are resampled 180x180 patches and the release carries no pixel size, so
   distances stay in pixels and the 2D tolerance grid applies (tau = 2 px, sensitivity 1 and 4).
   These are CT data reported in pixels -- state it, or the numbers look comparable with the mm-based
   volumetric sources when they are not.

Licence: **CC BY 3.0**, stated in the release's own LICENSE file for the preprocessed data, which permits
publishing derived statistics.
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

DATASET = "lidc"
LICENCE = "CC-BY-3.0"
TASK = "nodule"
N_RATERS = 4

SPLITS = ("train", "val", "test")

#: load-integrity check, from the released crops
EXPECTED_CROPS = 12816
EXPECTED_MASKS = EXPECTED_CROPS * N_RATERS

_MASK = re.compile(r"^(?P<stem>.+)_l(?P<rater>\d+)\.png$")


def load(root: Path | str) -> list[Annotation]:
    """Emit every LIDC row. `root` holds the split folders, normally data/gtu-raw/lidc."""
    root = Path(root)
    rows = []
    for split in SPLITS:
        gt = root / split / "gt"
        if not gt.is_dir():
            raise FileNotFoundError(f"missing LIDC split: {gt}")
        for patient_dir in sorted(p for p in gt.iterdir() if p.is_dir()):
            for path in sorted(patient_dir.glob("*.png")):
                match = _MASK.match(path.name)
                if match is None:
                    raise ValueError(f"unexpected LIDC filename: {path}")
                rows.append(
                    Annotation(
                        dataset=DATASET,
                        task=TASK,
                        image_id=f"{patient_dir.name}__{match['stem']}",  # minted, note 2
                        rater_id=f"l{match['rater']}",  # a slot, not a person, note 3
                        mask_path=path,
                        spacing=None,  # resampled crops, no pixel size ships, note 5
                        modality=Modality.CT,
                        annotation_type=AnnotationType.PIXEL,
                        rater_provenance=Provenance.P4,  # sequential unblinded revision
                        expertise=Expertise.EXPERT,
                        licence=LICENCE,
                        source_type=SourceType.RECOMPUTED,
                        seeded=False,
                        cluster_id=patient_dir.name,  # note 1
                        label_value=None,
                    )
                )
    return rows


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    root = (
        Path(args[0])
        if args
        else Path(__file__).resolve().parents[2] / "data/gtu-raw/lidc"
    )
    loaded = load(root)
    crops = {r.image_id for r in loaded}
    patients = {r.cluster_id for r in loaded}
    print(f"{len(loaded)} rows from {root}")
    print(
        f"  crops={len(crops)}  patients={len(patients)}  "
        f"raters={sorted({r.rater_id for r in loaded})}"
    )
    print(
        f"  provenance={loaded[0].rater_provenance.value} "
        f"(floor_eligible={loaded[0].floor_eligible()}, provenance row only)  "
        f"licence={loaded[0].licence}"
    )
    if len(loaded) != EXPECTED_MASKS:
        raise SystemExit(f"expected {EXPECTED_MASKS} masks, found {len(loaded)}")
    print(f"  every crop carries {N_RATERS} masks: {len(crops) * N_RATERS == len(loaded)}")
    per_patient: dict[str, set[str]] = {}
    for row in loaded:
        per_patient.setdefault(row.cluster_id, set()).add(row.image_id)
    sizes = sorted(len(v) for v in per_patient.values())
    print(
        f"  crops per patient (the cluster unit, note 1): "
        f"min={sizes[0]} median={sizes[len(sizes) // 2]} max={sizes[-1]}"
    )
