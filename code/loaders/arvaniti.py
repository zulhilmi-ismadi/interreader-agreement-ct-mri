"""Arvaniti prostate TMA loader (arvaniti-2018, Sci Rep 8:12054). H&E prostate tissue microarrays,
two pathologists, pixel-level Gleason pattern maps.

A recomputed histology **pixel** source whose licence permits release (Gleason 2019 and Mesmer are
both withheld on licence).

Source: Harvard Dataverse, doi:10.7910/DVN/OCYCMP, **CC0 1.0**. Only the two test-cohort mask sets are
needed (12.3 MB); the 2.5 GB of TMA images are not, because a floor is computed from masks alone.

Notes that matter:

1. **Provenance is P1, and it was verified twice.** The paper states the test cohort "were
   independently annotated by a second pathologist (J.H.R.)" after a first pathologist (K.S.F.).
   Independently, but the paper does **not** say explicitly that the second was blinded to the first,
   so the class is recorded as P1 with that caveat rather than assumed. The data corroborates
   independence: only 10 of 245 mask pairs are bit-identical (4.1 %), and per-class agreement runs
   0.25-0.43 Dice. Adjudication of one pathologist's work by another cannot produce that.
   Contrast NuCLS, where 99.1 % of multiply-annotated nuclei shared one bit-identical contour.

2. **The training cohort has a single annotator** (K.S.F. alone) and forms no pair, so
   `Gleason_masks_train.tar.gz` is not loaded. It contributes no pair and no floor.

3. **Label encoding, from the deposit's own description**: `0=Benign, 1=Gleason_3, 2=Gleason_4,
   3=Gleason_5, 4=unlabelled`. Tasks are the three Gleason patterns, named to match the
   `gleason2019` loader so the two prostate sources are directly comparable.

4. **Label 4 (unlabelled) is never a task.** It marks absence of annotation, not a structure. Scoring
   it would report a floor of ~0.84 for "the parts nobody annotated", which is meaningless.

5. **Benign (label 0) is also excluded**, on evidence rather than by preference: it appears on only
   12 of 245 cores. In this release benign functions as the unmarked remainder rather than a
   delineated structure, so a 12-core "task" would report the annotation convention, not agreement.
   Recorded here because it is a judgement, not a rule.

6. **Cores nest in TMA blocks**, so `cluster_id` is the block (`ZT80_38_A` from `ZT80_38_A_7_10`) and
   the cluster bootstrap resamples blocks rather than cores.

7. **Published agreement statistic** (reproduction of published values): the paper reports Cohen's **quadratic
   kappa 0.71** between the two pathologists at the spot level for the assigned Gleason score. That is
   a core-level score agreement, not a pixel overlap, so it is NOT a reproduction target for our Dice
   — report it beside ours as a definitional difference, exactly as Gleason 2019's kappa is handled.

8. **`--prepare` exists for one reason: these are PALETTE PNGs.** The class code is the palette
   *index*, and `gtu.io.load_mask` reads PNGs with skimage, which expands a palette to RGB — so the
   released files arrive as (3100, 3100, 3) and load_mask correctly refuses them. Gleason 2019's PNGs
   are single-channel, which is why that path never hit this. `prepare` re-writes each mask as a
   single-channel index PNG under `data/gtu-raw/arvaniti/`, changing no value. The alternative,
   teaching `load_mask` about palettes, would alter a shared reader that every other source depends
   on, to fix one source's packaging.

Licence: **CC0-1.0**, stated on the Dataverse record. No restriction on releasing derived statistics.
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

DATASET = "arvaniti"
LICENCE = "CC0-1.0"

#: note 3 — the deposit's own encoding; note 4 and 5 explain the two excluded values
TASKS = {"gleason3": 1, "gleason4": 2, "gleason5": 3}
BENIGN, UNLABELLED = 0, 4

RATERS = {"r1": ("Gleason_masks_test_pathologist1", "mask1_"),
          "r2": ("Gleason_masks_test_pathologist2", "mask2_")}

EXPECTED_CORES = 245

#: `ZT80_38_A_7_10` -> block `ZT80_38_A` (note 6)
_CORE = re.compile(r"^(?P<block>ZT\d+_\d+_[A-Z])_(?P<row>\d+)_(?P<col>\d+)$")


def prepare(source: Path | str, out: Path | str) -> None:
    """Re-write each palette PNG as a single-channel index PNG (note 8). Values are unchanged."""
    import numpy as np
    from PIL import Image

    source, out = Path(source), Path(out)
    written = 0
    for rater, (folder, prefix) in RATERS.items():
        directory = source / folder
        if not directory.is_dir():
            raise FileNotFoundError(
                f"{directory} missing — extract the two mask tarballs from doi:10.7910/DVN/OCYCMP"
            )
        target_dir = out / rater
        target_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(directory.glob(f"{prefix}*.png")):
            target = target_dir / f"{path.stem[len(prefix):]}.png"
            if target.exists():
                continue
            image = Image.open(path)
            if image.mode not in ("P", "L"):
                raise ValueError(f"{path.name}: expected a palette or grey mask, got mode {image.mode}")
            array = np.array(image)  # PIL gives the palette INDEX, which is the class code
            unexpected = set(np.unique(array)) - set(range(5))
            if unexpected:
                raise ValueError(f"{path.name}: unexpected class codes {sorted(unexpected)}")
            Image.fromarray(array.astype("uint8"), mode="L").save(target)
            written += 1
    print(f"wrote {written} single-channel masks to {out}")


def load(root: Path | str) -> list[Annotation]:
    """Emit every Arvaniti row. `root` holds the prepared masks, normally data/gtu-raw/arvaniti."""
    root = Path(root)
    per_rater: dict[str, dict[str, Path]] = {}
    for rater in RATERS:
        directory = root / rater
        if not directory.is_dir():
            raise FileNotFoundError(
                f"{directory} missing — run `python code/loaders/arvaniti.py --prepare`"
            )
        per_rater[rater] = {path.stem: path for path in sorted(directory.glob("*.png"))}

    shared = sorted(set.intersection(*(set(v) for v in per_rater.values())))
    if len(shared) != EXPECTED_CORES:
        raise ValueError(f"expected {EXPECTED_CORES} cores annotated by both, found {len(shared)}")

    rows = []
    for core in shared:
        match = _CORE.match(core)
        if match is None:
            raise ValueError(f"unexpected Arvaniti core id: {core}")
        for task, label_value in TASKS.items():
            for rater, masks in per_rater.items():
                rows.append(
                    Annotation(
                        dataset=DATASET,
                        task=task,
                        image_id=core,  # the deposit's own identifier; the release key
                        rater_id=rater,
                        mask_path=masks[core],
                        spacing=None,  # no physical pixel size ships; distances in pixels
                        modality=Modality.HISTOLOGY,
                        annotation_type=AnnotationType.PIXEL,
                        rater_provenance=Provenance.P1,  # note 1
                        expertise=Expertise.EXPERT,
                        licence=LICENCE,
                        source_type=SourceType.RECOMPUTED,
                        seeded=False,
                        cluster_id=match["block"],  # note 6
                        label_value=label_value,
                    )
                )
    return rows


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    repo = Path(__file__).resolve().parents[2]
    root = Path(args[0]) if args else repo / "data/gtu-raw/arvaniti"

    if "--prepare" in sys.argv:
        prepare(repo / "data/Arvaniti", root)

    loaded = load(root)
    print(f"{len(loaded)} rows from {root}")
    print(f"  cores={len({r.image_id for r in loaded})}  blocks={sorted({r.cluster_id for r in loaded})}")
    print(f"  raters={sorted({r.rater_id for r in loaded})}  tasks={sorted({r.task for r in loaded})}")
    print(
        f"  modality={loaded[0].modality.value}  annotation_type={loaded[0].annotation_type.value}  "
        f"provenance={loaded[0].rater_provenance.value}  licence={loaded[0].licence}  "
        f"floor_eligible={loaded[0].floor_eligible()}"
    )
