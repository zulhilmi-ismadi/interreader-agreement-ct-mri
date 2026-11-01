"""Gleason 2019 loader. A recomputed histology source, and by image count the largest of them: six
pathologists graded prostate tissue microarray cores from the Vancouver Prostate Centre,
each pathologist's maps released as a separate archive, so rater attribution is unambiguous.

Layout: <root>/Maps{1..6}_T/<slide>_<core>_classimg_nonconvex.png, one 8-bit class map per
pathologist per core.

It anchors the high-disagreement end of the layer: these are Gleason-grade *region* maps, not nuclei,
and the grade boundary is a judgement call rather than a visible edge. That is the point of including
it.

Notes that matter:

1. **Six raters, uneven panel.** Every pathologist annotated a different subset — Maps1 244 cores,
   Maps2 141, Maps3 242, Maps4 244, Maps5 246, Maps6 65, over 247 distinct cores. Coverage per core:
   40 cores carry all six, 122 carry five, 79 four, 5 three, and exactly one core carries a single
   rater (excluded from the floor, since a pair needs two raters, and counted). The sparse-coverage
   rule governs the rest: pairs form only among raters present on that core.

2. **Rater ids are stable across cores**, unlike RIGA's. Each `Maps<n>_T` archive is one pathologist's
   complete output, so `r1..r6` mean the same person on every core they touched. That makes the
   per-rater table, the rater bootstrap and the drop-one-rater sensitivity analysis all valid
   here without the caveat RIGA carries.

3. **Label vocabulary is {0, 1, 3, 4, 5, 6}** and only 3/4/5 are unambiguous. 3, 4 and 5 are the
   Gleason grades and become the three tasks. 0 is the catch-all: it covers the glass around the core
   *and* tissue the pathologist did not call cancerous, so it is not a "benign" structure and no task
   is emitted for it. Measured on `slide001_core005`, rater 0-fractions run 0.33 to 0.65 against the
   ~0.22 a circular core inscribed in a square would give from glass alone, and where one rater has 0
   at the core centre another has grade 4 — so 0 conflates background with benign by construction.
   **Labels 1 and 6 are unresolved** (see `UNRESOLVED_LABELS`): both are drawn as real regions, 6
   covering up to 59 % of a core, but nothing in the release says what they are and the secondary
   sources disagree (one reading has 0/1/6 all benign). They are not loaded, because the challenge
   documentation does not define them; `--verify` reports how often each appears so the question stays visible.

4. **Cores nest in slides**, so `cluster_id` is the slide (cluster bootstrap over slides). 247
   cores over the slides named in the filenames.

5. **Shapes vary between cores but never within one.** Eight distinct sizes from 4608x4608 to
   5632x5632; all 247 cores checked at the PNG header, zero rater disagreements, so `gtu.io.align`
   never has to reconcile geometry here.

6. **No spacing ships with the maps**, so distances stay in pixels and the 2D NSD tolerance
   applies (tau = 2 px primary, 1 px and 4 px sensitivity) — same footing as CryoNuSeg.

Empty-mask caveat for whoever reads the task floors: most cores carry only one or two of the three
grades, so the grade-5 task in particular is dominated by pairs where neither rater used the label.
The empty-mask policy scores those as DSC = 1 and the both-empty count is reported per task, so the
floor restricted to cores where a grade actually appears stays recoverable from `per_image.csv`. Do not quote the grade-5 floor without that count beside it.

Licence: **unstated**. The challenge is account-gated and no LICENSE or README ships in any of the
eight archives (checked every entry). Computed, withheld from release until the
organisers confirm terms.
"""

from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
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

DATASET = "gleason2019"
LICENCE = UNSTATED_LICENCE

#: archive folder -> rater id. One archive is one pathologist's complete output (note 2).
RATER_FOLDERS = {f"Maps{n}_T": f"r{n}" for n in range(1, 7)}

#: label value in the class map -> task name, one binary task per class. See note 3 before adding to this.
GRADES = {3: "gleason3", 4: "gleason4", 5: "gleason5"}

#: Drawn as regions but of unknown meaning; deliberately not tasks. Note 3.
UNRESOLVED_LABELS = (1, 6)

#: Glass plus un-called tissue, never a structure. Note 3.
BACKGROUND_LABEL = 0

#: Load-integrity check: map counts per pathologist, as published in the archives.
EXPECTED_MAPS = {
    "Maps1_T": 244,
    "Maps2_T": 141,
    "Maps3_T": 242,
    "Maps4_T": 244,
    "Maps5_T": 246,
    "Maps6_T": 65,
}
EXPECTED_CORES = 247

_FILENAME = re.compile(
    r"^(?P<image_id>(?P<slide>slide\d+)_core\d+)_classimg_nonconvex\.png$"
)


def load(root: Path | str) -> list[Annotation]:
    """Emit every Gleason 2019 row. `root` holds the six Maps folders, normally
    data/gtu-raw/gleason2019."""
    root = Path(root)
    rows = []
    for folder, rater_id in RATER_FOLDERS.items():
        for path in _map_paths(root, folder):
            image_id, slide = _parse(path)
            for label, task in GRADES.items():
                rows.append(
                    Annotation(
                        dataset=DATASET,
                        task=task,
                        image_id=image_id,
                        rater_id=rater_id,
                        mask_path=path,
                        spacing=None,  # no physical pixel size ships; distances in pixels
                        modality=Modality.HISTOLOGY,
                        annotation_type=AnnotationType.PIXEL,
                        rater_provenance=Provenance.P1,
                        expertise=Expertise.EXPERT,
                        licence=LICENCE,
                        source_type=SourceType.RECOMPUTED,
                        seeded=False,  # graded from the slide, no algorithmic pre-segmentation
                        cluster_id=slide,  # cores nest in slides, so the bootstrap resamples slides
                        label_value=label,
                    )
                )
    return rows


def _map_paths(root: Path, folder: str) -> list[Path]:
    directory = root / folder
    if not directory.is_dir():
        raise FileNotFoundError(f"missing Gleason 2019 rater set: {directory}")
    paths = sorted(directory.glob("*.png"))
    expected = EXPECTED_MAPS[folder]
    if len(paths) != expected:
        raise ValueError(f"{directory}: expected {expected} maps, found {len(paths)}")
    return paths


def _parse(path: Path) -> tuple[str, str]:
    match = _FILENAME.match(path.name)
    if match is None:
        raise ValueError(f"unexpected Gleason 2019 filename: {path.name}")
    return match["image_id"], match["slide"]


def _verify(rows: list[Annotation]) -> None:
    """Read every map once and report what the loader is standing on: the label vocabulary, how
    often the unresolved labels appear, and how many cores actually carry each grade.

    The last number is the one to read before quoting a task floor. A grade absent from every rater
    on a core still contributes a both-empty pair scored 1.0 under the empty-mask policy."""
    import numpy as np

    from gtu.io import load_mask

    paths = sorted({row.mask_path for row in rows})
    slide_of = {row.image_id: row.cluster_id for row in rows}

    vocabulary: Counter[int] = Counter()
    grade_cores: dict[int, set[str]] = defaultdict(set)
    unresolved_maps: dict[int, int] = dict.fromkeys(UNRESOLVED_LABELS, 0)

    print(f"\nreading {len(paths)} maps (once each):")
    for index, path in enumerate(paths, 1):
        image_id, _ = _parse(path)
        present = {int(v) for v in np.unique(load_mask(path))}
        vocabulary.update(present)
        for label in present & set(GRADES):
            grade_cores[label].add(image_id)
        for label in present & set(UNRESOLVED_LABELS):
            unresolved_maps[label] += 1
        if index % 200 == 0:
            print(f"  {index}/{len(paths)}")

    print("\nlabel vocabulary (maps containing each value):")
    for value, count in sorted(vocabulary.items()):
        if value in GRADES:
            role = f"task {GRADES[value]}"
        elif value == BACKGROUND_LABEL:
            role = "background + un-called tissue, no task"
        elif value in UNRESOLVED_LABELS:
            role = "UNRESOLVED, no task"
        else:
            role = "UNEXPECTED - not in the known label vocabulary"
        print(f"  {value}: {count} maps  ({role})")

    print("\ncores carrying each grade (denominator for a restricted floor):")
    all_cores = len(slide_of)
    for label, task in GRADES.items():
        cores = grade_cores[label]
        slides = {slide_of[c] for c in cores}
        print(
            f"  {task}: {len(cores)}/{all_cores} cores over {len(slides)} slides "
            f"({len(cores) / all_cores:.1%}); the other "
            f"{all_cores - len(cores)} contribute both-empty pairs (scored 1.0)"
        )

    print("\nunresolved labels (note 3 - not loaded as tasks):")
    for label, count in sorted(unresolved_maps.items()):
        print(f"  {label}: drawn on {count} maps")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    root = (
        Path(args[0])
        if args
        else Path(__file__).resolve().parents[2] / "data/gtu-raw/gleason2019"
    )
    loaded = load(root)
    print(f"{len(loaded)} rows from {root}")

    cores = {r.image_id for r in loaded}
    coverage = Counter()
    for image_id in cores:
        coverage[len({r.rater_id for r in loaded if r.image_id == image_id})] += 1
    if len(cores) != EXPECTED_CORES:
        raise SystemExit(f"expected {EXPECTED_CORES} cores, found {len(cores)}")

    print(
        f"  cores={len(cores)}  slides={len({r.cluster_id for r in loaded})}  "
        f"raters={sorted({r.rater_id for r in loaded})}"
    )
    for task in sorted({r.task for r in loaded}):
        task_rows = [r for r in loaded if r.task == task]
        print(
            f"  {task}: {len(task_rows)} rows, "
            f"images={len({r.image_id for r in task_rows})}, "
            f"label_value={task_rows[0].label_value}, "
            f"floor_eligible={any(r.floor_eligible() for r in task_rows)}"
        )
    print("  rater coverage per core:")
    for n_raters, count in sorted(coverage.items(), reverse=True):
        note = "  <- excluded from the floor, fewer than two raters" if n_raters < 2 else ""
        print(f"    {n_raters} raters: {count} cores{note}")
    print(
        f"  provenance={loaded[0].rater_provenance.value}  "
        f"modality={loaded[0].modality.value}  licence={loaded[0].licence}  "
        f"spacing={loaded[0].spacing}"
    )

    if "--verify" in sys.argv:
        _verify(loaded)
