"""RIGA loader (almazroa-2017-riga). Six ophthalmologists outlined the optic DISC and CUP on 750
colour fundus images - the largest rater x image product among the datasets analysed, and the only source where
one panel annotates two nested structures on the same pixels, so it shows how agreement differs
between two structures outlined by the same raters.

  MESSIDOR                                460 images
  Magrabia/{MagrabiaMale, MagrabiFemale}   95 images
  BinRushed/{BinRushed2,3,4} +
  BinRushedcorrected/BinRushed1-Corrected 195 images
                                          ---
                                          750

THE MARKS ARE DRAWN ON THE PICTURE, NOT SHIPPED AS MASKS. Each image base ships `<base>prime` (the
clean fundus photograph) and `<base>-1` ... `<base>-6` (the same photograph with one ophthalmologist's
two contours drawn on it). So the masks have to be recovered:

    contour = |markup - prime| exceeds THRESHOLD in any channel
    keep the two largest connected components (outer = disc, inner = cup, by bounding-box containment)
    disc, cup = fill_holes(outer), fill_holes(inner)

Colour is never assumed. The annotators did not use one ink - on a single MESSIDOR image the six
markups come back as several different colours, one of them a clean [46, 49, 146] blue and others
blended - so keying on a palette would silently lose raters. The difference against `prime` is
colour-blind and recovers all six.

FOUR THINGS THAT WOULD HAVE BROKEN THIS QUIETLY:

1. **BinRushed1's originals are missing from `BinRushed.zip`** - 50 image bases, every one of them in
   `BinRushed1`, ship markups with no `prime`, so a naive walk silently yields 700 images, not 750.
   `BinRushedcorrected/BinRushed1-Corrected` holds the same 50 bases WITH their primes (350 files =
   50 x 7), and is used for that subset alone. The rest of BinRushed comes from the uncorrected tree,
   which is complete.

2. **Extensions are mixed inside a single image.** `image1-3.tif` sits beside `image1-2.jpg` under the
   same base, so files are resolved by stem and read through PIL, never by a fixed suffix.

3. **JPEG makes the difference noisy, TIFF does not.** Where a pair is TIFF the difference is exactly
   the drawn pixels (`px>0 == px>30`, two components, always). Where the original is JPEG, compression
   perturbs ~1.57 M pixels at threshold 0 and the contour only emerges above ~30, with a handful of
   speckle components beside it. Hence THRESHOLD and MIN_COMPONENT_PX, and hence keeping the two
   LARGEST components rather than requiring exactly two.

4. **A rejected image is recorded, never guessed.** If the cleanup does not leave two nested contours,
   or the cup is not inside the disc, that (image, rater) is written to the skip table with its reason
   instead of contributing a mask. Images left with fewer than two raters form no pair and are dropped.

TWO-STEP USE, because the extraction is the expensive part and is done once:

    python code/loaders/riga.py --prepare     # writes _prepared/<image>/<rater>__{disc,cup}.tif
    python code/loaders/riga.py               # loads them

ASSUMPTION, and it matters more than it looks: **that `-1` names the same ophthalmologist on every
image is an assumption, not a verified fact.** Nothing in the file names carries a rater identity, and
the dataset readme (a PDF) does not settle it in machine-readable form. Every
per-rater statement - the per-rater table, the rater bootstrap and the drop-any-one-rater
sensitivity analysis - assumes rater ids are stable ACROSS images. The
pairwise floor itself does not: it only needs the six marks on one image to come from six different
people, which is certain. So the pairwise floor does not depend on the assumption, and the per-rater
views do. The loader prints the assumption whenever it runs.

Licence: CC BY-NC 4.0, stated on the Deep Blue Data record (doi:10.7302/Z23R0R29); see `LICENCE` below.

Published agreement statistic for the reproduction test: the agreement reported in almazroa-2017.
"""

from __future__ import annotations

import collections
import re
import sys
from pathlib import Path

import numpy as np

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

DATASET = "riga"
# Licence: the Deep Blue Data record (doi:10.7302/Z23R0R29) states
# http://creativecommons.org/licenses/by-nc/4.0/ in its licence field. Non-commercial, which
# licences.tsv records.
LICENCE = "CC-BY-NC-4.0"
PREPARED = "_prepared"
N_IMAGES = 750
N_RATERS = 6

DISC, CUP = "disc", "cup"
TASKS = (DISC, CUP)

#: Channel difference above which a pixel counts as drawn. Set by the JPEG noise floor (note 3):
#: TIFF pairs are unaffected (their difference is 0 or far above this), JPEG pairs need it.
THRESHOLD = 30
#: Connected components below this are compression speckle, not contour.
MIN_COMPONENT_PX = 100

#: BinRushed1 is taken from the corrected tree because only that copy ships the originals (note 1).
#: The image id is keyed on the LEAF FOLDER, not on the subset, because base names restart inside
#: each folder: MagrabiaMale and MagrabiFemale both number from image1, as do the four BinRushed
#: folders. Keying on the subset collapsed 749 images onto 558 ids and silently overwrote 97 of them.
#: Leaf folder names are unique across the release.
SUBSETS = (
    ("MESSIDOR", "MESSIDOR"),
    ("Magrabia/MagrabiaMale", "Magrabia"),
    ("Magrabia/MagrabiFemale", "Magrabia"),
    ("BinRushedcorrected/BinRushed1-Corrected", "BinRushed"),
    ("BinRushed/BinRushed2", "BinRushed"),
    ("BinRushed/BinRushed3", "BinRushed"),
    ("BinRushed/BinRushed4", "BinRushed"),
)

#: leaf folder -> the release subset it belongs to, for reporting only
FOLDER_SUBSET = {Path(relative).name: subset for relative, subset in SUBSETS}

_MARKUP = re.compile(r"^(?P<base>.+)-(?P<rater>[1-6])$", re.IGNORECASE)
_PRIME = re.compile(r"^(?P<base>.+)prime$", re.IGNORECASE)


def _read_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB")).astype(np.int16)


def _contours(markup: np.ndarray, prime: np.ndarray):
    """Recover (disc, cup) boolean masks, or (None, reason) if the marks do not resolve."""
    from scipy import ndimage

    if markup.shape != prime.shape:
        return None, f"shape mismatch {markup.shape} vs {prime.shape}"

    drawn = np.abs(markup - prime).max(axis=2) > THRESHOLD
    labels, count = ndimage.label(drawn, structure=np.ones((3, 3)))
    if count < 2:
        return None, f"{count} contour component(s) after threshold"

    sizes = ndimage.sum(drawn, labels, range(1, count + 1))
    keep = [i + 1 for i in np.argsort(-sizes)[:2] if sizes[i] >= MIN_COMPONENT_PX]
    if len(keep) < 2:
        return None, f"only {len(keep)} component(s) above {MIN_COMPONENT_PX} px"

    boxes = ndimage.find_objects(labels)
    first, second = (boxes[i - 1] for i in keep)
    if _contains(first, second):
        outer, inner = keep
    elif _contains(second, first):
        inner, outer = keep
    else:
        return None, "the two contours are not nested"

    # Fill inside the contour's own bounding box, not the whole photograph. `binary_fill_holes` is
    # the cost of this loader and it scales with the array it is handed: the disc occupies roughly
    # 250 x 250 px of a 1488 x 2240 image, so filling the crop is ~50x less work for an identical
    # result - the contour is closed and wholly inside the crop, so no hole can leak past its edge.
    # One pixel of padding keeps the fill from touching the crop border.
    disc = _fill_in_window(labels == outer, boxes[outer - 1], drawn.shape)
    cup = _fill_in_window(labels == inner, boxes[inner - 1], drawn.shape)
    if not cup.any() or not disc.any():
        return None, "a filled contour came back empty"
    if cup.sum() >= disc.sum():
        return None, "cup is not smaller than disc"
    if (cup & ~disc).sum() > 0.02 * cup.sum():
        return None, "cup lies outside disc by more than 2 %"
    return (disc, cup), "ok"


def _fill_in_window(component: np.ndarray, box, shape: tuple[int, int]) -> np.ndarray:
    """Fill one closed contour inside its bounding box and paste it back at full size."""
    from scipy import ndimage

    window = tuple(
        slice(max(s.start - 1, 0), min(s.stop + 1, size)) for s, size in zip(box, shape)
    )
    filled = np.zeros(shape, dtype=bool)
    filled[window] = ndimage.binary_fill_holes(component[window])
    return filled


def _contains(outer, inner) -> bool:
    """Does bounding box `outer` contain `inner`? Slices are (y, x) from find_objects."""
    return all(o.start <= i.start and o.stop >= i.stop for o, i in zip(outer, inner))


def _bases(folder: Path) -> dict[str, dict]:
    """Group one subset folder's files by image base: its prime and its six markups."""
    found: dict[str, dict] = collections.defaultdict(lambda: {"prime": None, "markups": {}})
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in (".tif", ".tiff", ".jpg", ".jpeg"):
            continue
        prime = _PRIME.match(path.stem)
        if prime:
            found[prime["base"].lower()]["prime"] = path
            continue
        markup = _MARKUP.match(path.stem)
        if markup:
            found[markup["base"].lower()]["markups"][int(markup["rater"])] = path
    return found


def prepare(root: Path | str) -> tuple[dict[str, int], list[dict]]:
    """Extract every rater's disc and cup mask. Run once; reruns skip images already written."""
    import tifffile

    root = Path(root)
    out_root = root / PREPARED
    counts: collections.Counter = collections.Counter()
    skipped: list[dict] = []

    for relative, subset in SUBSETS:
        folder = root / relative
        if not folder.is_dir():
            counts["subset_missing"] += 1
            continue
        for base, parts in sorted(_bases(folder).items()):
            if parts["prime"] is None:
                counts["image_no_prime"] += 1
                skipped.append({"subset": subset, "image": base, "rater": "", "reason": "no prime"})
                continue
            image_id = f"{folder.name}_{base}"
            destination = out_root / image_id
            if destination.is_dir() and len(list(destination.glob("*.tif"))) >= 2:
                counts["image_already_prepared"] += 1
                continue

            prime = _read_rgb(parts["prime"])
            written = 0
            for index in sorted(parts["markups"]):
                result, reason = _contours(_read_rgb(parts["markups"][index]), prime)
                if result is None:
                    counts["rater_rejected"] += 1
                    skipped.append(
                        {"subset": subset, "image": base, "rater": f"r{index}", "reason": reason}
                    )
                    continue
                disc, cup = result
                destination.mkdir(parents=True, exist_ok=True)
                for task, mask in ((DISC, disc), (CUP, cup)):
                    tifffile.imwrite(
                        destination / f"r{index}__{task}.tif", mask.astype(np.uint8), compression="zlib"
                    )
                written += 1
                counts["rater_masks"] += 1
            counts["image_prepared" if written else "image_empty"] += 1
    return dict(counts), skipped


def load(root: Path | str) -> list[Annotation]:
    """Emit every RIGA row. `root` is normally data/gtu-raw/riga; run `prepare` first."""
    root = Path(root)
    prepared = root / PREPARED
    if not prepared.is_dir():
        raise FileNotFoundError(
            f"{prepared} missing - run `python code/loaders/riga.py --prepare` first"
        )

    rows = []
    for folder in sorted(p for p in prepared.iterdir() if p.is_dir()):
        for task in TASKS:
            masks = sorted(folder.glob(f"*__{task}.tif"))
            if len(masks) < 2:
                continue  # no pair, no floor contribution
            for path in masks:
                rows.append(
                    Annotation(
                        dataset=DATASET,
                        task=task,
                        image_id=folder.name,
                        rater_id=path.stem.split("__")[0],
                        mask_path=path,
                        spacing=None,  # no physical pixel size ships with RIGA
                        modality=Modality.FUNDUS,
                        annotation_type=AnnotationType.PIXEL,
                        rater_provenance=Provenance.P1,
                        expertise=Expertise.EXPERT,
                        licence=LICENCE,
                        source_type=SourceType.RECOMPUTED,
                        cluster_id=None,  # one eye per image; images do not nest
                    )
                )
    return rows


if __name__ == "__main__":
    import pandas as pd

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    root = Path(args[0]) if args else Path(__file__).resolve().parents[2] / "data/gtu-raw/riga"

    if "--prepare" in sys.argv:
        stats, skipped = prepare(root)
        print(f"prepared {root / PREPARED}")
        for key, value in sorted(stats.items()):
            print(f"  {key:24s} {value}")
        if skipped:
            table = pd.DataFrame(skipped)
            out = Path(__file__).resolve().parents[2] / "results/riga_prepare_skipped.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(out, index=False)
            print(f"\n  {len(table)} (image, rater) marks rejected -> results/{out.name}")
            for reason, group in table.groupby("reason"):
                print(f"    {len(group):5d}  {reason}")

    loaded = load(root)
    images = {r.image_id for r in loaded}
    print(f"\n{len(loaded)} rows from {root}")
    print(f"  images={len(images)} (expected {N_IMAGES})  tasks={sorted({r.task for r in loaded})}  "
          f"raters={sorted({r.rater_id for r in loaded})}")
    for task in TASKS:
        rows = [r for r in loaded if r.task == task]
        panels = collections.Counter(
            len({r.rater_id for r in rows if r.image_id == i}) for i in {r.image_id for r in rows}
        )
        print(f"  {task:5s}: {len(rows):5d} rows over {len({r.image_id for r in rows})} images, "
              f"panel size -> n images {dict(sorted(panels.items()))}")
    folders = collections.Counter(i.rsplit("_", 1)[0] for i in images)
    subsets = collections.Counter(
        FOLDER_SUBSET.get(f, "?") for i in images for f in [i.rsplit("_", 1)[0]]
    )
    print(f"  folders: {dict(sorted(folders.items()))}")
    print(f"  subsets: {dict(sorted(subsets.items()))}")
    print(f"  licence={LICENCE}  provenance=P1  floor_eligible=True")
    print("\n  ASSUMPTION: rater ids assume the `-N` suffix is the same ophthalmologist on every")
    print("  image. Unverified: no rater identity ships with the masks. The pairwise floor")
    print("  does not depend on it; per-rater views (per-rater table, rater bootstrap, drop-one) do. Read per-rater")
    print("  values with this caveat.")
