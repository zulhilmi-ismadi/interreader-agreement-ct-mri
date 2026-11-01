"""Recover the CURVAS annotation seed with TotalSegmentator, on correctly oriented images.

The CURVAS data descriptor states that its three radiologists were given one shared TotalSegmentator
output and asked to correct it. This script reproduces that seed so that `curvas_seed_screen.py` can
measure how far each radiologist moved from it.

**Orientation.** 89 of the 90 CURVAS `image.nii.gz` files carry qform_code = sform_code = 0. Readers then
fall back to a default affine that labels the second array axis anterior (LAS), but the anatomy shows the
arrays are stored LPS: the spine and kidneys sit at higher second-axis indices than the pancreas. The one
file with valid codes, UKCHLL115, is LPS. A model given the default affine therefore sees a body mirrored
front to back. Before inference, each image with missing codes is given an LPS affine built from its own
voxel sizes (the direction cosines of UKCHLL115) and written to a temporary file with valid codes. The
arrays are not changed, so the seed lands on the original grid and compares voxel for voxel with the
radiologists' outlines.

Writes one multilabel mask per case in CURVAS label order (1 pancreas, 2 kidney, 3 liver), the two
TotalSegmentator kidneys unioned, to data/derived/curvas_totalsegmentator_lps/, plus runs.json with the
version, arguments and orientation fix for every case. Cases run unsplit and one at a time, with one thread
for resampling and saving to bound memory: splitting
along z is known to lose volume on this data.

    python code/curvas_seed_inference.py \
        [--cases UKCHLL001,UKCHLL115] [--limit N]

Run in a Python environment with TotalSegmentator 2.18.0 installed. Resumable: a case whose output
already exists and reads in full is skipped.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "gtu-raw" / "curvas"
OUT = ROOT / "data" / "derived" / "curvas_totalsegmentator_lps"
MANIFEST = OUT / "runs.json"

#: CURVAS label order (code/loaders/curvas.py note 2) -> the TotalSegmentator ROIs that make it up.
ORGAN_ROIS = {1: ("pancreas",), 2: ("kidney_left", "kidney_right"), 3: ("liver",)}
ROI_SUBSET = sorted({r for rois in ORGAN_ROIS.values() for r in rois})
#: direction cosines of the arrays as stored (LPS), taken from UKCHLL115, the one file with valid codes
LPS = np.diag([-1.0, -1.0, 1.0])


def cases() -> list[tuple[str, str, Path]]:
    found = []
    for split in ("training_set", "validation_set", "testing_set"):
        for case_dir in sorted((RAW / split).iterdir()):
            img = case_dir / "image.nii.gz"
            if case_dir.is_dir() and img.exists():
                found.append((split, case_dir.name, img))
    return found


def oriented_copy(img_path: Path, tmp_path: Path) -> str:
    """Write the image with a valid LPS affine if its header has none; return what was done."""
    img = nib.load(img_path)
    if int(img.header["qform_code"]) > 0 or int(img.header["sform_code"]) > 0:
        return f"kept header ({''.join(nib.aff2axcodes(img.affine))})"
    zooms = np.asarray(img.header.get_zooms()[:3], dtype=float)
    affine = np.eye(4)
    affine[:3, :3] = LPS * zooms
    fixed = nib.Nifti1Image(np.asanyarray(img.dataobj), affine, header=img.header)
    fixed.set_qform(affine, code=1)
    fixed.set_sform(affine, code=1)
    nib.save(fixed, tmp_path)
    return "codes 0: LPS affine applied"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=None, help="comma-separated case ids (default: all 90)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from importlib.metadata import version as _pkg_version
    from totalsegmentator.python_api import totalsegmentator
    ts_version = _pkg_version("TotalSegmentator")

    OUT.mkdir(parents=True, exist_ok=True)
    todo = cases()
    if args.cases:
        wanted = set(args.cases.split(","))
        todo = [c for c in todo if c[1] in wanted]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} cases; TotalSegmentator {ts_version}, full resolution, no split, rois={ROI_SUBSET}",
          flush=True)

    runs = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    done = skipped = failed = 0
    for i, (split, case, img_path) in enumerate(todo, 1):
        out_path = OUT / f"{case}.nii.gz"
        if out_path.exists():
            try:
                np.asanyarray(nib.load(out_path).dataobj)          # reads the whole file: catches truncation
                skipped += 1
                continue
            except Exception:
                out_path.unlink(missing_ok=True)

        ref = nib.load(img_path)
        tmp_img = OUT / f"_{case}_lps.nii.gz"
        orientation = oriented_copy(img_path, tmp_img)
        source = tmp_img if tmp_img.exists() else img_path
        tmp = OUT / f"_{case}_rois"
        try:
            totalsegmentator(str(source), str(tmp), fast=False, roi_subset=list(ROI_SUBSET), ml=False,
                             quiet=True, force_split=False, nr_thr_saving=1, nr_thr_resamp=1)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  [{i}/{len(todo)}] {case}: FAILED {type(exc).__name__}: {exc}", flush=True)
            failed += 1
            tmp_img.unlink(missing_ok=True)
            continue

        combined = np.zeros(ref.shape, dtype=np.uint8)
        missing = []
        for label, rois in ORGAN_ROIS.items():
            organ = np.zeros(ref.shape, dtype=bool)
            for roi in rois:
                f = tmp / f"{roi}.nii.gz"
                if not f.exists():
                    missing.append(roi)
                    continue
                m = np.rint(nib.load(f).get_fdata()).astype(np.uint8) > 0
                if m.shape != ref.shape:
                    missing.append(roi)
                    continue
                organ |= m
            combined[organ] = label
        # saved with the original header, so the seed sits on the radiologists' grid
        nib.save(nib.Nifti1Image(combined, ref.affine, header=ref.header), out_path)
        for f in tmp.glob("*"):
            f.unlink()
        tmp.rmdir()
        tmp_img.unlink(missing_ok=True)

        vox = float(np.prod(ref.header.get_zooms()[:3])) / 1000.0
        # re-read, so that two processes given disjoint --cases do not overwrite each other's entries
        runs = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
        runs[case] = dict(split=split, orientation=orientation, fast=False, force_split=False,
                          rois=ROI_SUBSET, missing=missing, totalsegmentator=ts_version,
                          volume_cm3={ORGAN_ROIS[k][0].split("_")[0]: round(int((combined == k).sum()) * vox, 1)
                                      for k in ORGAN_ROIS},
                          when=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        MANIFEST.write_text(json.dumps(runs, indent=1, sort_keys=True))
        done += 1
        print(f"  [{i}/{len(todo)}] {case}  {orientation}  {runs[case]['volume_cm3']}"
              + (f"  MISSING {missing}" if missing else ""), flush=True)

    print(f"\nseeded {done}, already present {skipped}, failed {failed}", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
