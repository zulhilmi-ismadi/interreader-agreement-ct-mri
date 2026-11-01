"""LiverHccSeg loader. Two board-certified radiologists segmented liver and HCC lesions independently
on multiphase contrast MRI, 17 patients from TCGA (gross-2023-liverhccseg).

Layout: <case>/<study_date>/rater{1,2}_{liver,tumor1,tumor2,tumor3}.nii.gz, alongside the four
acquisition phases (`pre`, `art`, `pv`, `del`) and the arterial coregistrations (`art_pre`, `art_pv`,
`art_del`). Only the rater masks are read here.

Notes that matter:

1. **Tasks are the source's own structure names**, not names of ours: `liver`, `tumor1`, `tumor2`,
   `tumor3`. That is deliberate. The release ships `segm_metrics.xlsx` with the authors' own
   rater1-vs-rater2 metrics **per case**, tidied to `data/derived/liverhccseg_published_percase.csv`,
   and its `structure` column uses exactly these four names. Keeping them makes the reproduction of
   published agreement statistics a direct join on (image_id, task, metric) with no mapping table in
   between. **This is the only dataset analysed here that publishes per-case agreement** — everywhere
   else the reproduction can only check a summary, so here it can be checked scan by scan.

2. **`tumor2` and `tumor3` exist on one case only.** TCGA-BC-A10Y carries three separately annotated
   lesions; every other case has one. So those two tasks have n_images = 1 and their CI is degenerate
   — an image bootstrap over a single image, with only 2 raters so the rater bootstrap cannot rescue
   it either. Report them as the single observations they are, or pool the lesion
   tasks downstream; do not present their CI as a spread.

3. **Anisotropy is extreme and it lands on NSD.** Slice pitch runs **1.75 mm to 10.8 mm** and in-plane
   0.742-1.406 mm, up to 14.6:1 on TCGA-BC-A10Y. NSD uses a 2 mm tolerance for MRI, which is under a
   fifth of one slice on the coarsest case, and the tolerance sensitivity grid (1/2/3 mm) does not
   reach it either. This is PDACVI's problem (0.45-8.00 mm) one step worse: report NSD for
   LiverHccSeg with the slice pitch beside it or the boundary numbers read as though they were
   isotropic.

4. **Geometry is clean.** All 33 (case, structure) units checked: rater1 and rater2 agree on shape and
   affine everywhere, every mask is int16 with values exactly {0, 1}, and none is empty. No alignment
   or dtype workaround is needed here, unlike Mesmer and CURVAS.

5. **A published duplicate this loader can settle.** TCGA-DD-A4NF and TCGA-DD-A4NH carry identical
   liver rows in `segm_metrics.xlsx`, agreeing to every printed decimal on all 12 metrics — two
   patients scanned 16 months apart cannot match to 4 dp on Hausdorff distance, so one row is
   duplicated in their supplement and the file does not say which is real. Our recompute distinguishes
   them. That is a reproduction *result*, not a reproduction failure; `--verify` flags the pair so it is not forgotten.

6. **One study date per case**, so `image_id` is the TCGA patient id alone and stays unique. The date
   is kept out of the id to hold the release key to the identifier the original dataset is indexed by,
   verbatim; it remains recoverable from the path.

Licence: neither zip carries a LICENSE or README covering the data; the Zenodo deposits state CC BY 4.0
(see `LICENCE` below), so the dataset's own terms are confirmed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nibabel

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

DATASET = "liverhccseg"
# Licence, from DataCite: the Zenodo deposits (10.5281/zenodo.7957515, .7957516, .8179129) all state
# Creative Commons Attribution 4.0 International, so the dataset's own terms are confirmed and do not
# rest on the article's CC BY.
LICENCE = "CC-BY-4.0"

#: the source's own rater folders/prefixes; ids kept verbatim
RATERS = ("rater1", "rater2")

#: the source's own structure names, which are also the published metrics file's `structure` values
TASKS = ("liver", "tumor1", "tumor2", "tumor3")

#: load-integrity check: units per structure, counted in the released files
EXPECTED_UNITS = {"liver": 17, "tumor1": 14, "tumor2": 1, "tumor3": 1}
EXPECTED_CASES = 17

#: the pair whose published liver rows are identical to 4 dp; note 5
PUBLISHED_DUPLICATE_CASES = ("TCGA-DD-A4NF", "TCGA-DD-A4NH")


def load(root: Path | str) -> list[Annotation]:
    """Emit every LiverHccSeg row. `root` holds one folder per TCGA case, normally
    data/gtu-raw/liverhccseg."""
    root = Path(root)
    cases = sorted(p for p in root.iterdir() if p.is_dir())
    if len(cases) != EXPECTED_CASES:
        raise ValueError(f"{root}: expected {EXPECTED_CASES} cases, found {len(cases)}")

    rows = []
    for case in cases:
        study = _study_dir(case)
        for task in TASKS:
            paths = {rater: study / f"{rater}_{task}.nii.gz" for rater in RATERS}
            present = {r: p for r, p in paths.items() if p.is_file()}
            if not present:
                continue  # most cases have one lesion; tumor2/tumor3 are TCGA-BC-A10Y only
            if len(present) != len(RATERS):
                raise ValueError(
                    f"{study}: {task} annotated by {sorted(present)} only — a lesion "
                    f"segmented by one rater cannot form a pair"
                )
            spacing = tuple(
                float(z) for z in nibabel.load(paths[RATERS[0]]).header.get_zooms()[:3]
            )
            for rater, path in present.items():
                rows.append(
                    Annotation(
                        dataset=DATASET,
                        task=task,
                        image_id=case.name,  # TCGA id, verbatim; one study per case (note 6)
                        rater_id=rater,
                        mask_path=path,
                        spacing=spacing,
                        modality=Modality.MRI,
                        annotation_type=AnnotationType.PIXEL,
                        rater_provenance=Provenance.P1,
                        expertise=Expertise.EXPERT,
                        licence=LICENCE,
                        source_type=SourceType.RECOMPUTED,
                        seeded=False,  # drawn from scratch, no algorithmic pre-segmentation
                        cluster_id=None,  # one scan per patient, nothing to nest
                        label_value=None,  # masks are already binary per structure
                    )
                )
    return rows


def _study_dir(case: Path) -> Path:
    """The single study-date folder inside a case (note 6)."""
    studies = sorted(p for p in case.iterdir() if p.is_dir())
    if len(studies) != 1:
        raise ValueError(
            f"{case}: expected exactly one study date, found {[p.name for p in studies]} — "
            f"image_id would no longer be unique"
        )
    return studies[0]


def _verify(rows: list[Annotation], root: Path) -> None:
    """Cross-check coverage against the authors' own per-case metrics file, and report the
    anisotropy that the fixed 2 mm NSD tolerance has to live with (note 3)."""
    import pandas as pd

    units = {(r.image_id, r.task) for r in rows}
    counts: dict[str, int] = {}
    for _, task in units:
        counts[task] = counts.get(task, 0) + 1
    print("\nunits per structure vs the counts in the released files:")
    for task in TASKS:
        expected = EXPECTED_UNITS[task]
        found = counts.get(task, 0)
        flag = "OK" if found == expected else "MISMATCH"
        print(f"  {task}: {found} vs expected {expected}  {flag}")

    # a project artifact, not a sibling of `root` — resolve from the repo, so a custom root still works
    published = (
        Path(__file__).resolve().parents[2]
        / "data/derived/liverhccseg_published_percase.csv"
    )
    if published.is_file():
        frame = pd.read_csv(published)
        their_units = set(zip(frame.patient_id, frame.structure))
        print(f"\nagainst {published.name} ({len(their_units)} published units):")
        print(f"  in both: {len(units & their_units)}")
        missing = sorted(their_units - units)
        extra = sorted(units - their_units)
        if missing:
            print(f"  published but not loaded: {missing}")
        if extra:
            print(f"  loaded but not published: {extra}")
        if not missing and not extra:
            print("  exact match - the reproduction joins on (image_id, task) with no mapping")
        duplicated = frame[
            frame.patient_id.isin(PUBLISHED_DUPLICATE_CASES) & (frame.structure == "liver")
        ]
        print(
            f"\n  note 5: {' and '.join(PUBLISHED_DUPLICATE_CASES)} publish identical liver rows "
            f"({duplicated.metric.nunique()} metrics); our recompute distinguishes them"
        )
    else:
        print(f"\npublished per-case file not found at {published}")

    spacings = {}
    for row in rows:
        spacings[row.image_id] = row.spacing
    pitches = sorted((s[2], case) for case, s in spacings.items())
    print("\nslice pitch (NSD uses tau = 2 mm for MRI):")
    print(f"  finest  {pitches[0][0]:.2f} mm  {pitches[0][1]}")
    print(f"  coarsest {pitches[-1][0]:.2f} mm  {pitches[-1][1]}  "
          f"- tau is {2 / pitches[-1][0]:.2f} of one slice here")
    coarse = [c for pitch, c in pitches if pitch > 2.0]
    print(f"  {len(coarse)}/{len(pitches)} cases have a slice pitch coarser than tau")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    root = (
        Path(args[0])
        if args
        else Path(__file__).resolve().parents[2] / "data/gtu-raw/liverhccseg"
    )
    loaded = load(root)
    print(f"{len(loaded)} rows from {root}")
    print(
        f"  cases={len({r.image_id for r in loaded})}  "
        f"raters={sorted({r.rater_id for r in loaded})}"
    )
    for task in TASKS:
        task_rows = [r for r in loaded if r.task == task]
        if not task_rows:
            continue
        print(
            f"  {task}: {len(task_rows)} rows, "
            f"images={len({r.image_id for r in task_rows})}, "
            f"floor_eligible={any(r.floor_eligible() for r in task_rows)}"
        )
    print(
        f"  provenance={loaded[0].rater_provenance.value}  "
        f"modality={loaded[0].modality.value}  licence={loaded[0].licence}"
    )
    if "--verify" in sys.argv:
        _verify(loaded, root)
