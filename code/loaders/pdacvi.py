"""PDACVI (CURVAS-PDAC, riera-2026-pdacvi) loader. Five independent expert raters outlined the
pancreatic-cancer lesion on 109 contrast CT studies - the largest expert panel among the datasets
analysed after C3RO, and the lowest-agreement source among them (the paper's own pairwise human DSC is 58.4 +/- 25.7).

Layout: <split>/<case>/annotation_{1..5}.nii.gz, one BINARY lesion mask per rater, plus two files
that are not rater masks and are excluded here:

  annotation_staple.nii.gz    the organisers' own STAPLE consensus of the five raters
  annotation_vascular.nii.gz  a single multi-class vascular map (labels 1-5), not per-rater

Notes that matter:

1. **The masks are binary, not multi-class.** Unlike CURVAS, which packs three organs into one label
   map, every PDACVI annotation is {0, 1}, so there is one task and `label_value` is 1.

   Setting it to 1 rather than leaving it None is a MEMORY choice, and it is only safe because the
   masks really are binary. With `label_value=None`, `compute_layer._task_mask` hands the raw array
   to the metrics, which binarise it themselves as `!= 0`; with `label_value=1` the loader hands over
   `array == 1`, a bool array 2x smaller than the int16 files and 4x smaller than the int32 the
   float64 files become. That shrinks the windowing temporaries and, more importantly, View B, which
   materialises one full-size array per rater. The two are identical **iff no mask carries a label
   other than 0 or 1** - a stray 2 would be foreground under `!= 0` and background under `== 1`, and
   nothing downstream would complain. So `--verify` scans all 546 files and asserts it. Run it after
   any re-download; it is the only thing standing between this optimisation and a silent change.

2. **Geometry varies case to case** - 512x512x561 at 0.709 x 0.709 x 1.0 mm in one case, 1024x1024x213
   at 0.391 x 0.391 x 3.0 mm in another - so spacing is read per case, never assumed. The 3 mm slice
   pitch matters for NSD: the tolerance is 2 mm for CT, which is below one slice thickness on such
   cases, and the tolerance sensitivity grid (1/2/3 mm) is what carries that.

3. **dtypes are inconsistent across cases** (`float64`, `int16`, `uint8` all observed for the same
   filename), which is why everything goes through `gtu.io.load_mask` rather than a bare comparison.

4. **The panel is 5 raters except on one case.** CURVASPDAC_00568 (validation split) ships a sixth,
   `annotation_6.nii.gz`. The sparse-coverage rule already handles uneven panels - pairs form among the
   raters present on that image - so it is loaded rather than trimmed, and `--report` prints the panel sizes.

5. **Not determined from the release, and deliberately not decided here:** the source documentation records a PANORAMA
   seed for one rater, which would make that rater P1 with `seeded=True`, but the release does not say
   WHICH of the five was seeded, and nothing in the file names or headers distinguishes them. Flagging
   all five would overstate it; flagging none understates it. This loader flags none by default and
   makes the choice one argument (`seeded_rater="r3"`), so that it can be set if the organisers
   identify the rater. The flag is a sensitivity stratum (seeded against unseeded), never an exclusion, so the
   floor itself is unaffected either way - only the seeded-stratum comparison is.

Licence: riera-2026-pdacvi does not print a dataset licence; the Zenodo deposit states CC BY 4.0 (see
`LICENCE` below), so the dataset's own terms are confirmed.

Published statistic to reproduce: pairwise human DSC 58.4 +/- 25.7 %.
"""

from __future__ import annotations

import os
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
    UNSTATED_LICENCE,
)

DATASET = "pdacvi"
# Licence, from DataCite: the Zenodo deposit "CURVAS-PDACVI dataset" (10.5281/zenodo.15401567,
# .15720085, .17552201) states Creative Commons Attribution 4.0 International on every version. The
# paper's own licence (CC BY-NC-SA) covers the article, not the deposit.
LICENCE = "CC-BY-4.0"
TASK = "lesion"

N_CASES = 109
EXPECTED_PANEL = 5

#: The masks are binary, so foreground is label 1. See note 1 in the module docstring: this is a
#: memory optimisation that is only equivalent while every mask is {0, 1}, which `--verify` checks.
LABEL_VALUE = 1
ALLOWED_LABELS = frozenset({0, 1})

SPLITS = ("training_set", "testing_set", "validation_set")

#: Files matching `annotation_<digits>` are rater masks. `annotation_staple` and
#: `annotation_vascular` deliberately do not match - see notes 1 and 5 in the module docstring.
_RATER_FILE = re.compile(r"^annotation_(?P<index>\d+)\.nii\.gz$")

#: Not a rater mask, but the organisers' consensus of the five. Kept addressable because it is a free
#: external check on our own leave-one-rater-out STAPLE consensus (View B), which no other dataset analysed here
#: provides; `staple_path` exposes it without letting it leak into the floor.
STAPLE_FILE = "annotation_staple.nii.gz"

#: Single multi-class vascular reference (labels 1-5), one per case rather than one per rater, so it
#: cannot define an agreement floor. Recorded here so a later reader does not mistake it for a task.
VASCULAR_FILE = "annotation_vascular.nii.gz"


def load(root: Path | str, seeded_rater: str | None = None) -> list[Annotation]:
    """Emit every PDACVI row. `root` holds the split folders, normally data/gtu-raw/pdacvi.

    `seeded_rater` is the rater id (e.g. "r3") to mark `seeded=True` (P1, seeded); None by default,
    because the release does not say which rater received the PANORAMA pre-segmentation (docstring note 5)."""
    root = Path(root)
    rows: list[Annotation] = []
    for split in SPLITS:
        split_root = root / split
        if not split_root.is_dir():
            continue
        for case_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
            paths = _rater_paths(case_dir)
            spacing = tuple(float(z) for z in nibabel.load(paths[0]).header.get_zooms()[:3])
            for path in paths:
                rater = f"r{int(_RATER_FILE.match(path.name)['index'])}"
                rows.append(
                    Annotation(
                        dataset=DATASET,
                        task=TASK,
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
                        seeded=(rater == seeded_rater),
                        cluster_id=None,  # one study per patient
                        label_value=LABEL_VALUE,  # binary masks; see docstring note 1 and --verify
                    )
                )
    return rows


def _labels_in(path: Path) -> tuple[str, list[int]]:
    """Every distinct label in one mask. Module level so a process pool can pickle it."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gtu.io import load_mask

    import numpy as np

    array = load_mask(path)
    return str(path), sorted(int(v) for v in np.unique(array))


def verify_binary(root: Path | str, workers: int = 2) -> list[tuple[str, list[int]]]:
    """Scan every rater mask and return the offenders whose labels are not a subset of {0, 1}.

    This is what licenses `label_value=1` (docstring note 1). It reads all 546 volumes, so it is
    minutes rather than seconds; it is a check to run once per download, not per compute."""
    from concurrent.futures import ProcessPoolExecutor

    paths = sorted(Path(root).rglob("annotation_*.nii.gz"))
    paths = [p for p in paths if _RATER_FILE.match(p.name)]
    offenders = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, (name, labels) in enumerate(pool.map(_labels_in, paths), start=1):
            if not set(labels) <= ALLOWED_LABELS:
                offenders.append((name, labels))
            if index % 50 == 0:
                print(f"    {index}/{len(paths)} scanned, {len(offenders)} offender(s)", flush=True)
    print(f"    {len(paths)}/{len(paths)} scanned, {len(offenders)} offender(s)", flush=True)
    return offenders


def staple_path(root: Path | str, split: str, case: str) -> Path:
    """The organisers' STAPLE consensus for one case. Not a rater mask; never loaded by `load`."""
    return Path(root) / split / case / STAPLE_FILE


def _rater_paths(case_dir: Path) -> list[Path]:
    paths = sorted(
        (p for p in case_dir.glob("annotation_*.nii.gz") if _RATER_FILE.match(p.name)),
        key=lambda p: int(_RATER_FILE.match(p.name)["index"]),
    )
    if len(paths) < 2:
        raise ValueError(f"{case_dir}: {len(paths)} rater masks, need >= 2 for a pair")
    return paths


if __name__ == "__main__":
    import collections

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    root = Path(args[0]) if args else Path(__file__).resolve().parents[2] / "data/gtu-raw/pdacvi"
    loaded = load(root)

    cases = {r.image_id for r in loaded}
    panels = collections.Counter()
    for image_id in cases:
        panels[len({r.rater_id for r in loaded if r.image_id == image_id})] += 1

    print(f"{len(loaded)} rows from {root}")
    print(f"  cases={len(cases)} (expected {N_CASES})  task={TASK}  "
          f"raters={sorted({r.rater_id for r in loaded})}")
    for split in SPLITS:
        present = sorted(p.name for p in (root / split).iterdir() if p.is_dir()) if (root / split).is_dir() else []
        print(f"  {split}: {len(present)} cases")
    print(f"  panel size -> n cases: {dict(sorted(panels.items()))}")
    print(f"  provenance={loaded[0].rater_provenance.value}  expertise={loaded[0].expertise.value}  "
          f"licence={loaded[0].licence}  floor_eligible={loaded[0].floor_eligible()}")

    if len(cases) != N_CASES:
        print(f"  *** case count {len(cases)} != published {N_CASES}")
    seeded = {r.rater_id for r in loaded if r.seeded}
    print(f"\n  NOTE: seeded raters = {sorted(seeded) or 'none flagged'}.")
    print("  The source documentation records a PANORAMA seed for one PDACVI rater; the release does")
    print("  not say which. Pass seeded_rater='rN' once confirmed. Affects the seeded stratum only, not the floor.")

    if "--verify" in sys.argv:
        print(f"\n  scanning all masks for labels outside {sorted(ALLOWED_LABELS)} "
              f"(licenses label_value={LABEL_VALUE}):")
        offenders = verify_binary(root, workers=int(os.environ.get("VERIFY_WORKERS", "2")))
        if offenders:
            print("  *** NOT BINARY - revert label_value to None before computing:")
            for name, labels in offenders[:10]:
                print(f"      {name}: labels={labels}")
            raise SystemExit(1)
        print(f"  OK: every mask is a subset of {sorted(ALLOWED_LABELS)}, so `array == 1` and the")
        print("  metrics' own `!= 0` binarisation select exactly the same voxels.")

    if "--spacing" in sys.argv:
        spacings = collections.Counter(r.spacing for r in loaded)
        print(f"\n  {len(spacings)} distinct voxel spacings across {len(cases)} cases; "
              f"slice pitch range "
              f"{min(s[2] for s in spacings):.2f}-{max(s[2] for s in spacings):.2f} mm")
