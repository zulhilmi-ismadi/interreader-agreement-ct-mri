"""IMA++ loader. ISIC Archive multi-annotator dermoscopic skin-lesion segmentation
(abhishek-2026-ima-plus-plus-isic): 15 human annotators over the 2,394 multi-annotated images.

Layout: `segs/<ISIC_id>_<annotator>_<tool>_<skill>_<mskObjectID>.png`, with the panel for each image
listed in `seg_metadata_multiannotator_subset.csv`. The loader is driven by that CSV, not by globbing:
the filename encodes the metadata but the CSV is the release's own index of which masks belong to the
multi-annotated subset.

Notes that matter:

1. **`MV` and `ST` sit in the `annotator` column and are not people.** They are the release's majority
   vote and STAPLE consensus masks -- exactly 2 per image, 4,788 of the 9,899 rows in the subset file --
   carrying `tool` and `skill_level` values of `MV`/`ST` too. Loading them as raters would put an
   algorithm-fused label into every pairwise comparison and inflate the floor of the largest
   multi-annotator source in the layer. They are **excluded from the rows entirely** and extracted to
   `fused/` instead, where `fused_paths()` exposes them. This is the same trap as algorithm-fused labels
   presented as raters in QUBIQ's brain tumour (class P5), and the same free external check that
   PDACVI's `annotation_staple` offers: a consensus computed by someone else, against which our own
   leave-one-rater-out STAPLE consensus (View B) can be compared.

2. **144 of the 2,394 images have only one human annotator.** They form no pair, so they are
   excluded from the floor and counted -- `compute_layer` records them in the `unpaired` table rather than failing. 2,250 images carry a real panel:
   2,043 pairs of raters, 170 triples, 35 quads and 2 quintets.

3. **Skill level cannot be mapped to `expertise`, and the reason is interesting.** The release tags
   each *segmentation* S1 or S2, and 7 of the 15 annotators appear under both -- so skill is a property
   of the annotation event, not of the person, and the schema's per-row `expertise` cannot carry it
   faithfully without knowing what S1 and S2 mean. The paper's abstract does not define them and
   expertise is never assigned by us, so every row is `unknown`, and IMA++ gives no expertise-tier
   result. With a definition of S1 and S2 it would add a third tiering axis, alongside C3RO's
   expert/non-expert and CryoNuSeg's intra/inter, and the only one that varies within a rater.

4. **Tool decides provenance, and the release's own definitions force the mapping.** The Zenodo
   record defines T1 as "Manual polygon tracing", T2 as "Semi-automated flood-fill, a human expert
   provides a seed point and tunes flood-fill parameters", and T3 as "Fully-automated algorithm, an
   automated segmentation algorithm generates the mask, reviewed and accepted by a human expert".
   The provenance classes therefore place T3 in **P5** (a boundary accepted from a suggestion without
   being drawn) and treat T2 as **P1 seeded** (a human working from an algorithmic seed). Only T1 is
   unseeded P1. Loading every mask as unseeded P1 would put 1,876 accepted algorithm outputs into the
   dermoscopy floor; the floor over pairs without a T3 mask is 0.718 against 0.784 with them. `ima_tool_contrast.py` reports the floor per tool combination.

5. **218 images carry two masks from the same annotator, and only one may enter a pair.** The
   duplicates are a tool contrast (usually T1 beside T3 by the same person), not a second opinion,
   so a pair built from both would compare an annotator with themselves across tools. The loader
   keeps the lowest tool (T1 before T2 before T3) and then the first `mskObjectID`, and emits one
   row per (image, annotator). `compute_layer` refuses to run if any source still has two masks for
   one rater on one image, which guards against a silent last-row-wins collapse.

6. **Masks are uint8 valued {0, 255}**, not {0, 1}; every metric binarises on `!= 0`, so this is safe,
   but do not write `== 1` against them.

Licence: Zenodo record 10.5281/zenodo.14201692; the article is separate from the data. The dataset's own
terms are CC BY-NC-ND 4.0 (see `LICENCE` below), so per-image rows are computed and withheld from release.
"""

from __future__ import annotations

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

DATASET = "ima"
# Licence, from DataCite (10.5281/zenodo.14201692): Creative Commons Attribution
# NonCommercial NoDerivatives 4.0 International, plus the ISIC Archive Terms of Use Agreement.
#
# **ND is the problem, and it is worse than an unstated licence.** Derived per-image agreement rows are
# plausibly adapted material, which CC BY-NC-ND forbids redistributing. So the licence is confirmed
# but the rows are still NOT releasable: naming the licence accurately must not make the constraint
# weaker than an unstated licence would. merge_layer and make_release carry this as a second withholding
# category, for licences that forbid derivatives.
LICENCE = "CC-BY-NC-ND-4.0"
TASK = "lesion"

#: values of the `annotator` column that are consensus artefacts, not people. See note 1.
FUSED_ANNOTATORS = frozenset({"MV", "ST"})

#: the release's own index of the multi-annotated subset
SUBSET_CSV = "seg_metadata_multiannotator_subset.csv"

#: tool -> (provenance, seeded), from the release's own definitions. See note 4.
TOOL_PROVENANCE = {
    "T1": (Provenance.P1, False),  # manual polygon tracing
    "T2": (Provenance.P1, True),   # flood-fill from a human seed point
    "T3": (Provenance.P5, False),  # algorithm output accepted by a human expert
}
#: which mask to keep when one annotator segmented one image twice. See note 5.
TOOL_ORDER = {"T1": 0, "T2": 1, "T3": 2}

#: load-integrity checks. 5,111 human masks in the release index; 221 are the extra masks of the 218
#: (image, annotator) pairs resolved by TOOL_ORDER, leaving one mask per annotator per image.
EXPECTED_HUMAN_MASKS = 4890
EXPECTED_IMAGES = 2394
EXPECTED_ANNOTATORS = 15


def load(root: Path | str) -> list[Annotation]:
    """Emit every IMA++ row. `root` holds `segs/` and the subset CSV, normally data/gtu-raw/ima."""
    import pandas as pd

    root = Path(root)
    frame = pd.read_csv(_subset_csv(root))
    human = frame[~frame.annotator.isin(FUSED_ANNOTATORS)].copy()

    unknown = set(human.tool) - set(TOOL_PROVENANCE)
    if unknown:
        raise ValueError(f"IMA++ tool values not covered by TOOL_PROVENANCE: {sorted(unknown)}")

    # note 5: one mask per (image, annotator), lowest tool first, then the first mskObjectID
    human["_tool_rank"] = human.tool.map(TOOL_ORDER)
    human = (
        human.sort_values(["ISIC_id", "annotator", "_tool_rank", "mskObjectID"])
        .drop_duplicates(["ISIC_id", "annotator"], keep="first")
    )

    rows = []
    for record in human.itertuples():
        provenance, seeded = TOOL_PROVENANCE[record.tool]
        path = root / "segs" / record.seg_filename
        if not path.is_file():
            raise FileNotFoundError(f"missing IMA++ mask: {path}")
        rows.append(
            Annotation(
                dataset=DATASET,
                task=TASK,
                image_id=record.ISIC_id,  # ISIC archive id, verbatim: the release key
                rater_id=record.annotator,  # A00..A15, stable across images
                mask_path=path,
                spacing=None,  # dermoscopy, no physical pixel size ships
                modality=Modality.DERMOSCOPY,
                annotation_type=AnnotationType.PIXEL,
                rater_provenance=provenance,  # note 4: T3 is P5, T1 and T2 are P1
                expertise=Expertise.UNKNOWN,  # skill level is per-segmentation, note 3
                licence=LICENCE,
                source_type=SourceType.RECOMPUTED,
                seeded=seeded,  # note 4: T2 works from a flood-fill seed
                cluster_id=None,  # one lesion image per row, nothing to nest
                label_value=None,  # masks are {0, 255}; every metric binarises on != 0
            )
        )
    return rows


def fused_paths(root: Path | str) -> dict[tuple[str, str], Path]:
    """(ISIC_id, method) -> the release's own MV / STAPLE consensus mask. Note 1: an external check
    on our leave-one-rater-out consensus (View B), never a rater."""
    import pandas as pd

    root = Path(root)
    frame = pd.read_csv(_subset_csv(root))
    fused = frame[frame.annotator.isin(FUSED_ANNOTATORS)]
    return {
        (r.ISIC_id, r.annotator): root / "fused" / r.seg_filename
        for r in fused.itertuples()
    }


def _subset_csv(root: Path) -> Path:
    """The subset index, from the raw folder or the original download beside it."""
    for candidate in (root / SUBSET_CSV, root.parents[1] / "IMA" / SUBSET_CSV):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"cannot find {SUBSET_CSV} for IMA++ under {root}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    root = (
        Path(args[0])
        if args
        else Path(__file__).resolve().parents[2] / "data/gtu-raw/ima"
    )
    loaded = load(root)
    images = {r.image_id for r in loaded}
    raters = sorted({r.rater_id for r in loaded})
    print(f"{len(loaded)} rows from {root}")
    print(f"  images={len(images)}  annotators={len(raters)}  {raters}")

    panels: dict[str, set[str]] = {}
    for row in loaded:
        panels.setdefault(row.image_id, set()).add(row.rater_id)
    sizes: dict[int, int] = {}
    for members in panels.values():
        sizes[len(members)] = sizes.get(len(members), 0) + 1
    print("  panel size per image:")
    for size, count in sorted(sizes.items()):
        note = "  <- no pair, excluded from the floor" if size < 2 else ""
        print(f"    {size} raters: {count} images{note}")
    print(
        f"  provenance={loaded[0].rater_provenance.value}  "
        f"modality={loaded[0].modality.value}  licence={loaded[0].licence}"
    )
    if len(loaded) != EXPECTED_HUMAN_MASKS:
        raise SystemExit(f"expected {EXPECTED_HUMAN_MASKS} human masks, found {len(loaded)}")
    fused = fused_paths(root)
    present = sum(1 for p in fused.values() if p.is_file())
    print(f"  fused consensus masks held aside (note 1): {present}/{len(fused)} on disk")
