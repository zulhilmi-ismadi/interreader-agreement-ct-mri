"""C3RO loader (wahid-2023-crowdsourced-radiotherapy). The anchor source: 5 cancer sites, one case
each, with both an expert and a non-expert annotator panel on every ROI — so it is what supplies the
expertise-tier contrast (P1 vs P2, which are kept strictly separate and never merged).

Layout: Organized_files_v4/<Site>/Segmentations/{Expert,Non-Expert}/<annotator>/NIFTI/<annotator>_<ROI>.nii.gz

Two things the folder tree will mislead you about:

1. **Not every NIfTI is a study ROI.** H&N alone carries ~70 distinct structure names, most of them
   annotators' private working structures (`ns_artifact`, `couchsurface`, `refctv1p-auto`, `anillo`,
   `ctvn_7000=gtvn_l+5mm`). The 28 ROIs the study actually analyses are fixed by the paper's
   Supplementary Table 1, transcribed into ROIS below and used as a strict whitelist. `--verify`
   checks the resulting per-ROI rater counts against that table's own numbers.

2. **One case per site.** The rater panel varies, the case does not, so `image_id` is the site and
   the rater bootstrap (resampling raters with replacement) is the only usable interval here — there is nothing to resample
   over images.

Coverage is sparse: an annotator who skipped an ROI simply has no file, which is handled
by forming pairs only among the raters present on each image.

Licence: CC BY 4.0 (Figshare).
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
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

DATASET = "c3ro"
LICENCE = "CC-BY-4.0"

#: folder name -> site slug used in task names
SITES = {
    "Breast": "breast",
    "GI": "gi",
    "GYN": "gyn",
    "H&N": "hn",
    "Sarcoma": "sarcoma",
}

#: Supplementary Table 1 (41597_2023_2062_MOESM1): ROI -> (expert count, non-expert count).
#: These are the paper's own numbers and are asserted by --verify.
ROIS: dict[str, dict[str, tuple[int, int]]] = {
    "Breast": {
        "CTV_Ax": (8, 115),
        "CTV_Chestwall": (8, 117),
        "CTV_IMN": (8, 118),
        "CTV_Sclav_LN": (8, 119),
        "BrachialPlex_L": (6, 88),
        "Heart": (7, 121),
        "A_LAD": (7, 88),
    },
    "Sarcoma": {"GTV": (5, 60), "CTV": (5, 48), "Genitals": (4, 51)},
    "H&N": {
        "GTVp": (14, 59),
        "GTVn": (13, 60),
        "CTV1": (9, 45),
        "CTV2": (9, 49),
        "Brainstem": (13, 58),
        "Glnd_Submand_L": (13, 57),
        "Glnd_Submand_R": (12, 52),
        "Larynx": (12, 57),
        "Musc_Constrict": (11, 43),
        "Parotid_L": (13, 59),
        "Parotid_R": (13, 58),
    },
    "GYN": {
        "GTVn": (5, 42),
        "CTVn_4500": (5, 40),
        "CTVp_4500": (5, 41),
        "Bowel_Small": (4, 35),
    },
    "GI": {"CTV_4500": (4, 25), "CTV_5400": (4, 23), "Bag_Bowel": (4, 23)},
}

TIERS = {
    "Expert": (Provenance.P1, Expertise.EXPERT),
    "Non-Expert": (Provenance.P2, Expertise.NON_EXPERT),
}

_FILENAME = re.compile(r"^(?P<annotator>[^_]+)_(?P<roi>.+)\.nii\.gz$")


def load(root: Path | str) -> list[Annotation]:
    """Emit every C3RO row. `root` is the directory holding the site folders, normally
    data/gtu-raw/c3ro/Organized_files_v4."""
    root = Path(root)
    rows = []
    for site, slug in SITES.items():
        site_root = root / site / "Segmentations"
        if not site_root.is_dir():
            raise FileNotFoundError(f"missing C3RO site: {site_root}")
        spacing = None
        for tier, (provenance, expertise) in TIERS.items():
            for annotator_dir in sorted((site_root / tier).iterdir()):
                if not annotator_dir.is_dir():
                    continue
                for path in sorted((annotator_dir / "NIFTI").glob("*.nii.gz")):
                    match = _FILENAME.match(path.name)
                    if match is None or match["roi"] not in ROIS[site]:
                        continue  # annotator working structure, not a study ROI
                    if spacing is None:
                        spacing = tuple(
                            float(z) for z in nibabel.load(path).header.get_zooms()[:3]
                        )
                    rows.append(
                        Annotation(
                            dataset=DATASET,
                            task=f"{slug}_{match['roi']}",
                            image_id=slug,
                            rater_id=f"{tier[0].lower()}{match['annotator']}",
                            mask_path=path,
                            spacing=spacing,
                            modality=Modality.CT,
                            annotation_type=AnnotationType.PIXEL,
                            rater_provenance=provenance,
                            expertise=expertise,
                            licence=LICENCE,
                            source_type=SourceType.RECOMPUTED,
                            cluster_id=None,  # one case per site; nothing to nest
                        )
                    )
    return rows


def _verify(rows: list[Annotation]) -> None:
    """Check the whitelist against the paper's Supplementary Table 1 counts."""
    counted: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in rows:
        tier = "Expert" if row.rater_provenance is Provenance.P1 else "Non-Expert"
        counted[(row.image_id, row.task, tier)] += 1

    print("\nrater counts vs C3RO Supplementary Table 1:")
    mismatches = 0
    for site, slug in SITES.items():
        for roi, (expert, non_expert) in ROIS[site].items():
            task = f"{slug}_{roi}"
            got = (counted[(slug, task, "Expert")], counted[(slug, task, "Non-Expert")])
            ok = got == (expert, non_expert)
            mismatches += not ok
            flag = "OK" if ok else f"MISMATCH (table says {expert}/{non_expert})"
            print(f"  {task:26s} expert={got[0]:3d} non-expert={got[1]:4d}  {flag}")
    print(f"\n{mismatches} of {sum(len(v) for v in ROIS.values())} ROIs disagree with the table")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    root = (
        Path(args[0])
        if args
        else Path(__file__).resolve().parents[2] / "data/gtu-raw/c3ro/Organized_files_v4"
    )
    loaded = load(root)
    print(f"{len(loaded)} rows from {root}")
    print(f"  tasks={len({r.task for r in loaded})} raters={len({r.rater_id for r in loaded})}")
    if "--verify" in sys.argv:
        _verify(loaded)
