"""Run the three admissible MONAI bundles on the CURVAS cases, scoring each against all three raters.

Companion to `per_rater_infer.py`, which does this for RIGA's 2D fundus photographs. CURVAS is
volumetric and its models are MONAI bundles rather than Hugging Face checkpoints.

**All 90 CURVAS cases ship `image.nii.gz`** — 20 in `training_set.zip`, 5 in `validation_set.zip` and
65 in `testing_set.zip` — so every case is scored. All three archives must be extracted with their
images: a tree into which only `training_set.zip` has been unpacked holds 20 image volumes beside 90
annotation sets, which looks like a release constraint but is an extraction artefact.

**Every model runs under its own authors' preprocessing, parsed from the bundle at run time.** The
three bundles do not agree on any of it: SwinUNETR wants a -175/250 HU window at 1.5x1.5x2.0 mm, the
two DiNTS models want -500/500 and -87/199 at 1 mm, and their sliding-window overlaps differ. Feeding
one model another's transform would make it look worse than it is, which is the error this experiment
must not commit. The values are therefore read out of each bundle's own `configs/inference.*` rather
than transcribed into this file, so they cannot drift from the published recipe; what is transcribed
here is only the mapping from each model's output channels onto the structures CURVAS annotates,
because that mapping is a decision and belongs in the open.

**What is excluded and why.** TotalSegmentator seeded the CURVAS annotations — the three radiologists
corrected its output — so scoring it against them would be circular. That exclusion has to reach models
*trained on TotalSegmentator's labels* as well, or it is cosmetic: `wholeBody_ct_segmentation`
(`data_source: TotalSegmentator`) and `pediatric_abdominal_ct_segmentation` (TotalSegmentator + TCIA +
BTCV) are both out on that ground. The three kept here declare training data independent of it.

    python code/per_rater_curvas.py --list
    python code/per_rater_curvas.py --model swin_unetr_btcv --task pancreas [--limit 2]

Writes results/per_rater/predictions_curvas_lps/<model>/<case>__<task>.nii.gz on each case's own voxel
grid, and `_model.json` recording the repository, revision, the recipe actually used and any
constructor argument that had to be dropped.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

CURVAS = ROOT / "data" / "gtu-raw" / "curvas"
#: CURVAS ships its cases in three archives. All three carry rater masks and an image volume; the
#: public leaderboard scored 60 of the 65 testing-split cases.
SPLITS = ("training_set", "validation_set", "testing_set")
OUT = ROOT / "results" / "per_rater"
#: Predictions on correctly oriented input (see `oriented_input`). A folder `predictions_curvas`, if
#: present, holds a run on images read with the default orientation; it is not used.
PREDICTIONS = OUT / "predictions_curvas_lps"
#: direction cosines of the CURVAS arrays as stored (LPS), from UKCHLL115, the one file with valid codes
LPS = np.diag([-1.0, -1.0, 1.0])


def oriented_input(image_path: Path, scratch: Path) -> Path:
    """Return a path to the image whose header states its true orientation.

    89 of the 90 CURVAS images carry qform_code = sform_code = 0, so readers fall back to an affine that
    labels the second array axis anterior, while the anatomy shows the arrays are stored LPS. A model given
    that affine sees the body mirrored front to back. Such an image is rewritten with an LPS affine from its
    own voxel sizes; the array is untouched, so a prediction mapped back through the same affine lands on
    the original grid. Images with valid codes are used as they are."""
    import nibabel

    image = nibabel.load(image_path)
    if int(image.header["qform_code"]) > 0 or int(image.header["sform_code"]) > 0:
        return image_path
    zooms = np.asarray(image.header.get_zooms()[:3], dtype=float)
    affine = np.eye(4)
    affine[:3, :3] = LPS * zooms
    fixed = nibabel.Nifti1Image(np.asanyarray(image.dataobj), affine, header=image.header)
    fixed.set_qform(affine, code=1)
    fixed.set_sform(affine, code=1)
    scratch.mkdir(parents=True, exist_ok=True)
    target = scratch / f"{image_path.parent.name}_lps.nii.gz"
    nibabel.save(fixed, target)
    return target
CACHE = ROOT / ".model-cache"

#: CURVAS packs all three organs into one label map. Order confirmed by voxel volume on all 20 cases
#: (median 122,120 < 458,452 < 2,331,058), which matches `code/loaders/curvas.py`.
CURVAS_LABELS = {"pancreas": 1, "kidney": 2, "liver": 3}

#: Only the channel mapping lives here; the preprocessing is read from each bundle. CURVAS labels one
#: kidney class while SwinUNETR emits left and right separately, so that structure is a union.
BUNDLES = {
    "swin_unetr_btcv": {
        "repo": "MONAI/swin_unetr_btcv_segmentation",
        "labels": {"pancreas": (11,), "kidney": (2, 3), "liver": (6,)},
        "note": "Swin UNETR trained on Beyond the Cranial Vault; 13 abdominal structures",
    },
    "dints_multi_organ": {
        "repo": "MONAI/multi_organ_segmentation",
        "labels": {"liver": (3,), "pancreas": (7,)},
        "note": "DiNTS trained at Aichi Cancer Center; eight abdominal structures, no kidney class",
    },
    "dints_pancreas": {
        "repo": "MONAI/pancreas_ct_dints_segmentation",
        "labels": {"pancreas": (1,)},
        "note": "DiNTS trained on Medical Segmentation Decathlon Task07; pancreas and its tumour",
    },
}


def cases() -> list[Path]:
    """The CURVAS cases that carry an image, across all three splits.

    Anything without an image cannot be given to a model. Ordered by split and then by name, so a run that
    is interrupted and resumed covers the cases in the same order. Case names are unique across the splits,
    which is what lets the predictions of all three sit in one folder keyed by name alone.
    """
    if not CURVAS.is_dir():
        raise SystemExit(f"{CURVAS} is missing")
    found: list[Path] = []
    for split in SPLITS:
        directory = CURVAS / split
        if not directory.is_dir():
            continue
        found += sorted(p for p in directory.iterdir()
                        if p.is_dir() and (p / "image.nii.gz").is_file())
    if not found:
        raise SystemExit(f"no case under {CURVAS} has an image.nii.gz in any of {', '.join(SPLITS)}")
    names = [c.name for c in found]
    if len(set(names)) != len(names):
        # the predictions of all three splits share one folder, keyed by case name
        raise SystemExit("case names are not unique across the splits; predictions would collide")
    return found


def split_of(case: Path) -> str:
    """Which archive a case came from. Recorded because only the testing split has a leaderboard."""
    return case.parent.name


def fetch(spec: dict) -> tuple[Path, str]:
    from huggingface_hub import snapshot_download

    CACHE.mkdir(parents=True, exist_ok=True)
    path = Path(snapshot_download(spec["repo"], cache_dir=str(CACHE)))
    return path, path.name


def read_recipe(bundle: Path) -> dict:
    """Pull the preprocessing and inferer settings out of the bundle's own inference config.

    Reading these rather than hard-coding them is the point: if a bundle is updated, the recipe used
    here changes with it, and a mismatch between what the authors published and what was run cannot
    creep in silently.
    """
    config = next((p for p in bundle.rglob("configs/inference.*") if p.suffix in (".json", ".yaml")), None)
    if config is None:
        raise SystemExit(f"{bundle.name} ships no configs/inference.json or .yaml")
    if config.suffix == ".json":
        parsed = json.loads(config.read_text(encoding="utf-8"))
    else:
        import yaml

        parsed = yaml.safe_load(config.read_text(encoding="utf-8"))

    recipe: dict = {"config_file": config.name, "network_def": parsed.get("network_def", {})}
    for transform in parsed.get("preprocessing", {}).get("transforms", []):
        target = str(transform.get("_target_", ""))
        if target.endswith("Spacingd"):
            recipe["pixdim"] = tuple(float(v) for v in transform["pixdim"])
        elif target.endswith("ScaleIntensityRanged"):
            recipe["a_min"] = float(transform["a_min"])
            recipe["a_max"] = float(transform["a_max"])
        elif target.endswith("Orientationd"):
            recipe["axcodes"] = transform.get("axcodes", "RAS")
    inferer = parsed.get("inferer", {})
    recipe["roi_size"] = tuple(int(v) for v in inferer.get("roi_size", (96, 96, 96)))
    recipe["overlap"] = float(inferer.get("overlap", 0.5))
    recipe["sw_batch_size"] = int(inferer.get("sw_batch_size", 1))
    recipe["output_classes"] = int(parsed.get("output_classes", 0)) or None

    missing = [key for key in ("pixdim", "a_min", "a_max") if key not in recipe]
    if missing:
        raise SystemExit(f"{bundle.name}: inference config does not state {missing}; refusing to guess")
    return recipe


def build_network(bundle: Path, recipe: dict) -> tuple[object, list[str]]:
    """Prefer the bundle's TorchScript; otherwise rebuild the architecture from its config.

    A rebuild is only trustworthy if the published weights land on it exactly, so the state dict is
    loaded and checked: anything missing or unexpected means the class has moved under the bundle and
    the run is stopped rather than reported. Constructor arguments the installed MONAI no longer
    accepts are dropped and returned, so the change is recorded beside the predictions instead of
    being invisible.
    """
    import torch

    script = bundle / "models" / "model.ts"
    if script.is_file():
        return torch.jit.load(str(script), map_location=device()).eval(), []

    import inspect

    import monai.networks.nets as nets

    definition = dict(recipe["network_def"])
    name = str(definition.pop("_target_")).rsplit(".", 1)[-1]
    factory = getattr(nets, name)
    accepted = set(inspect.signature(factory.__init__).parameters)
    dropped = sorted(k for k in definition if k not in accepted)
    network = factory(**{k: v for k, v in definition.items() if k in accepted})

    raw = torch.load(bundle / "models" / "model.pt", map_location="cpu", weights_only=True)
    state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    report = network.load_state_dict(state, strict=False)
    if report.missing_keys or report.unexpected_keys:
        raise SystemExit(
            f"{bundle.name}: {len(report.missing_keys)} missing and "
            f"{len(report.unexpected_keys)} unexpected keys after dropping {dropped}; "
            "the installed architecture no longer matches the published weights"
        )
    return network.eval().to(device()), dropped


def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def oriented_frame(affine: np.ndarray, shape: tuple, axcodes: str) -> tuple:
    """Where `Orientationd` leaves the case: the permutation, the shape and the affine it produces.

    That transform only permutes and flips axes — it resamples nothing — so the oriented shape is the
    case's own shape under the same permutation, and the oriented affine is nibabel's own composition,
    the one `as_reoriented` uses. Both are derived from the case's affine rather than assumed, so a case
    stored in any orientation is handled without a special case.
    """
    from nibabel.orientations import axcodes2ornt, inv_ornt_aff, io_orientation, ornt_transform

    forward = ornt_transform(io_orientation(affine), axcodes2ornt(tuple(axcodes)))
    order = np.argsort(forward[:, 0]).astype(int)
    oriented_shape = tuple(int(shape[k]) for k in order)
    oriented_affine = np.asarray(affine) @ inv_ornt_aff(forward, shape)
    return forward, oriented_shape, oriented_affine


def axis_map(spaced_affine: np.ndarray, oriented_affine: np.ndarray) -> tuple:
    """Per-axis scale and offset taking an oriented voxel index to a spaced one.

    Read off the two affines rather than recomputed from pixdim. `Spacingd`'s output size and origin
    follow from its own corner arithmetic, which does not reduce to any of the obvious formulas —
    `ceil(shape x zoom / pixdim)` and `round((shape - 1) x zoom / pixdim) + 1` each disagree with it on
    real cases, and `align_corners` True and False each describe a different convention again. None of
    that has to be guessed: MONAI records where it put the resampled volume in that volume's own affine,
    so composing the two affines gives the exact mapping. Both grids are in the same orientation by this
    point, so the composition is diagonal, which is checked rather than assumed.
    """
    mapping = np.linalg.inv(np.asarray(spaced_affine, dtype=float)) @ np.asarray(oriented_affine,
                                                                                dtype=float)
    linear = mapping[:3, :3]
    off_diagonal = np.abs(linear - np.diag(np.diag(linear))).max()
    if off_diagonal > 1e-4:
        raise SystemExit(
            "the resampled grid is rotated or sheared relative to the oriented case "
            f"(largest off-diagonal term {off_diagonal:.3g}); this inversion handles only the "
            "per-axis scaling Spacingd applies"
        )
    return tuple(float(v) for v in np.diag(linear)), tuple(float(v) for v in mapping[:3, 3])


def resample_axis(volume, axis: int, size: int, scale: float, offset: float):
    """Linear interpolation along one axis: `out[..., j, ...] = in[..., scale * j + offset, ...]`.

    Trilinear interpolation is separable, so undoing a per-axis scaling is three of these rather than
    one sampling grid. That is the whole economy of this inversion: `Invertd` materialises three
    coordinates for each of ~250 million voxels before it reads a single value, where this reads the
    volume three times and never builds a grid at all. It also expresses an arbitrary scale and offset
    per axis, which `interpolate` cannot: its two `align_corners` settings are two fixed conventions,
    and neither is the one `Spacingd` used.
    """
    import torch

    length = volume.shape[axis]
    position = (torch.arange(size, dtype=torch.float64) * scale + offset).clamp(0, length - 1)
    lower = position.floor()
    weight = (position - lower).to(volume.dtype)
    lower = lower.to(torch.long)
    upper = (lower + 1).clamp(max=length - 1)
    view = [1] * volume.ndim
    view[axis] = size
    weight = weight.reshape(view)
    return volume.index_select(axis, lower).mul_(1 - weight).add_(
        volume.index_select(axis, upper).mul_(weight))


def restore(probabilities, spaced_affine, affine: np.ndarray, shape: tuple, axcodes: str,
            chunk: int = 2):
    """Undo `Spacingd` and `Orientationd` and return the label map on the case's own voxel grid.

    This replaces MONAI's `Invertd`, which cannot afford these volumes: it inverts the chain by
    building an explicit sampling grid over the case's ~250 million voxels for every channel at once,
    which on one case needs about 16 GB of memory and over nine minutes. The same answer is available far more cheaply, because the two steps invert
    independently and neither needs a grid:

    * `Spacingd` is a per-axis scaling, so it is undone by three separable 1-D interpolations whose
      scale and offset come from the affines themselves (`axis_map`). The argmax over channels still
      runs at full resolution, after the interpolation, exactly as the bundles run it — no label map is
      ever interpolated. The channels are carried a few at a time while a running maximum and its
      channel index are kept across chunks, which gives bit-for-bit the same argmax as interpolating all
      fourteen at once while holding two of them in memory instead of fourteen.
    * `Orientationd` is a permutation and a flip, so it is undone exactly — not approximately — on the
      finished single-channel label map by nibabel's own `ornt_transform`, at no cost.
    """
    import torch
    from nibabel.orientations import apply_orientation, axcodes2ornt, io_orientation, ornt_transform

    forward, oriented_shape, oriented_affine = oriented_frame(affine, shape, axcodes)
    scales, offsets = axis_map(spaced_affine, oriented_affine)

    best_value = None
    best_channel = None
    for start in range(0, probabilities.shape[0], chunk):
        block = probabilities[start:start + chunk].to(torch.float32)
        for axis, size in enumerate(oriented_shape):
            block = resample_axis(block, axis + 1, size, scales[axis], offsets[axis])
        value, index = block.max(dim=0)
        del block
        index = (index + start).to(torch.int16)
        if best_value is None:
            best_value, best_channel = value, index
        else:
            better = value > best_value
            best_value = torch.where(better, value, best_value)
            best_channel = torch.where(better, index, best_channel)
            del better, value, index
    del best_value

    label = best_channel.numpy()
    backward = ornt_transform(axcodes2ornt(tuple(axcodes)), io_orientation(affine))
    return np.ascontiguousarray(apply_orientation(label, backward)), forward, backward


def run(model: str, tasks: list[str], limit: int | None, split: str | None = None) -> None:
    """Run one model over the cases and write a mask per structure it emits.

    The structures share a forward pass. A model produces one label map per case, and which structure
    a voxel belongs to is read out of that one map, so running the model again per structure would
    repeat the same arithmetic and return the same answer — `swin_unetr_btcv` alone would be run sixty
    times instead of twenty. Nothing about the recipe, the argmax or the channel mapping changes; only
    the number of times the identical forward pass is computed.
    """
    import torch
    from monai.inferers import SlidingWindowInferer
    from monai.transforms import (
        Compose,
        EnsureChannelFirstd,
        EnsureTyped,
        LoadImaged,
        Orientationd,
        ScaleIntensityRanged,
        Spacingd,
    )

    spec = BUNDLES[model]
    unsupported = [t for t in tasks if t not in spec["labels"]]
    if unsupported:
        raise SystemExit(
            f"'{model}' emits no {', '.join(unsupported)}; it is dropped for that structure "
            "and the drop is reported"
        )

    bundle, revision = fetch(spec)
    recipe = read_recipe(bundle)
    network, dropped = build_network(bundle, recipe)

    # the bundle's own chain, in its own order
    pre = Compose([
        LoadImaged(keys="image", image_only=False),
        EnsureChannelFirstd(keys="image"),
        Orientationd(keys="image", axcodes=recipe.get("axcodes", "RAS")),
        Spacingd(keys="image", pixdim=recipe["pixdim"], mode="bilinear"),
        ScaleIntensityRanged(keys="image", a_min=recipe["a_min"], a_max=recipe["a_max"],
                             b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys="image"),
    ])
    # the windows run on the GPU, but the volume they accumulate into stays in host memory: at the
    # DiNTS bundles' 1 mm the buffer is eight channels over ~232 million voxels, some 7.4 GB, which does
    # not fit beside the weights on a typical GPU. Where the output lands changes nothing about the result.
    inferer = SlidingWindowInferer(roi_size=recipe["roi_size"],
                                   sw_batch_size=recipe["sw_batch_size"],
                                   overlap=recipe["overlap"], sw_device=device(), device="cpu")
    axcodes = recipe.get("axcodes", "RAS")

    import nibabel

    folder = PREDICTIONS / model
    folder.mkdir(parents=True, exist_ok=True)
    found = cases()
    if split is not None:
        found = [c for c in found if split_of(c) == split]
    selected = found if limit is None else found[:limit]
    written = 0
    kept = 0
    for case in selected:
        # an interrupted run leaves finished cases on disk. They are kept rather than recomputed --
        # the forward pass is deterministic, so a recomputation would only cost time -- and the record
        # below counts them separately from the ones this run produced.
        if all((folder / f"{case.name}__{task}.nii.gz").is_file() for task in tasks):
            kept += 1
            print(f"  {case.name}: already on disk, kept")
            continue
        original = nibabel.load(case / "image.nii.gz")
        source = oriented_input(case / "image.nii.gz", folder / "_oriented")
        reference = nibabel.load(source)
        data = pre({"image": str(source)})
        with torch.no_grad():
            logits = inferer(data["image"].unsqueeze(0).to(device()), network)[0]

        # softmax before the interpolation and argmax after it, which is the order the bundles use. It is
        # computed in place (subtract the maximum, exponentiate, normalise, as torch.softmax does), so the
        # DiNTS buffer of ~7.4 GB is not held twice
        probabilities = logits.float().cpu()
        del logits
        probabilities.sub_(probabilities.amax(dim=0, keepdim=True)).exp_()
        probabilities.div_(probabilities.sum(dim=0, keepdim=True))
        # the resampled volume's own affine says where Spacingd put it, so the inversion is read off
        # the transform that actually ran rather than recomputed from the recipe
        label, _, _ = restore(probabilities, data["image"].affine, reference.affine,
                              reference.shape, axcodes)
        del probabilities
        if label.shape != reference.shape:
            raise SystemExit(
                f"{case.name}: prediction {label.shape} does not match the image {reference.shape}; "
                "the inversion did not return it to the original grid"
            )

        counts = []
        for task in tasks:
            mask = np.isin(label, spec["labels"][task])
            # saved with the original header, so the mask sits on the radiologists' grid
            nibabel.save(
                nibabel.Nifti1Image(mask.astype(np.uint8), original.affine, original.header),
                folder / f"{case.name}__{task}.nii.gz",
            )
            counts.append(f"{task} {int(mask.sum()):9d}")
        written += 1
        print(f"  {case.name}: " + "  ".join(counts) + " voxels")

    record_path = folder / "_model.json"
    record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.is_file() else {}
    record.update({"model": model, "repo": spec["repo"], "revision": revision,
                   "device": device(), "note": spec["note"],
                   "recipe": {k: v for k, v in recipe.items() if k != "network_def"},
                   # the installed MONAI is not the version the bundle was written against; any
                   # constructor argument that had to go is named here rather than left implicit
                   "constructor_args_dropped": dropped,
                   # Invertd is not what ran; the substitution is named here beside the predictions it
                   # produced, not left to the commit history
                   "inversion": "separable 1-D interpolation for Spacingd with scale and offset read "
                                "from the affines, nibabel ornt_transform for Orientationd; argmax at "
                                "full resolution after interpolation; agrees with MONAI Invertd on "
                                "100% of voxels on four test geometries"})
    # which split each scored case came from: the testing split is reported separately, because the
    # challenge leaderboard scored 60 of its 65 cases
    by_split: dict[str, int] = {}
    for case in selected:
        by_split[split_of(case)] = by_split.get(split_of(case), 0) + 1
    record["cases_by_split"] = dict(sorted(by_split.items()))
    record["case_splits"] = {case.name: split_of(case) for case in selected}
    for task in tasks:
        record.setdefault("tasks", {})[task] = {
            "channels_used": list(spec["labels"][task]), "cases": written + kept,
            "run_date": date.today().isoformat(),
            # named so the record does not imply one uninterrupted run
            "cases_kept_from_an_earlier_run": kept}
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"wrote {written} × {len(tasks)} masks ({kept} cases kept) to {folder.relative_to(ROOT)} "
          f"(revision {revision[:12]})")


#: With a spacing supplied the surface tolerance is read in millimetres, not voxels, as it is for every
#: volumetric source (2 mm primary, 1 and 3 mm as sensitivity). The number matches the RIGA arm's `TAU` so the two
#: arms are reported on the same grid position, not on the same physical distance — 2 px and 2 mm are
#: different quantities and the paper says which is which.
TAU_MM = 2


def raters(case: Path) -> list[str]:
    """The annotators present for a case. CURVAS ships three, but absence is checked, not assumed."""
    return sorted(p.stem.replace(".nii", "") for p in sorted(case.glob("annotation_*.nii.gz")))


def rater_mask(case: Path, rater: str, task: str) -> tuple[np.ndarray, tuple[float, float, float]]:
    """One rater's mask for one structure, with the case's own voxel spacing.

    CURVAS packs all three organs into a single label map, so a structure is one value out of it. The
    map is rounded rather than cast: it is int16 scaled across the dtype range with `scl_slope = nan`,
    so nibabel returns 0.9999999997 and a plain `== 1` matches nothing (`code/loaders/curvas.py` note 3).
    """
    import nibabel

    image = nibabel.load(case / f"{rater}.nii.gz")
    data = np.rint(np.asarray(image.dataobj)).astype(np.int16)
    spacing = tuple(float(z) for z in image.header.get_zooms()[:3])
    return data == CURVAS_LABELS[task], spacing


def score_volumes(prediction: np.ndarray, reference: np.ndarray, spacing) -> dict[str, float]:
    """The layer's own metric code, with spacing passed through so surface distances are in mm.

    Same functions as the RIGA arm and as the floors themselves, so a model-versus-rater number here
    sits on one scale with every other number in the paper.
    """
    from gtu import metrics
    from gtu.io import WindowedMask, align

    a, b = align(WindowedMask.from_array(prediction), WindowedMask.from_array(reference))
    return metrics.pairwise_scores(a, b, instance=False, spacing=spacing)


def score(task: str) -> None:
    """Score every CURVAS prediction against each of the three radiologists.

    Emits the same columns as the RIGA arm, so the ranking, bootstrap and margin functions in
    `per_rater_models.py` apply to this table unchanged rather than being written twice.
    """
    import pandas as pd

    import per_rater_models as prm

    rows = []
    for case in cases():
        for folder in sorted(p for p in PREDICTIONS.glob("*") if p.is_dir()):
            path = folder / f"{case.name}__{task}.nii.gz"
            if not path.is_file():
                continue
            import nibabel

            prediction = np.asarray(nibabel.load(path).dataobj) != 0
            for rater in raters(case):
                reference, spacing = rater_mask(case, rater, task)
                scores = score_volumes(prediction, reference, spacing)
                rows.append({
                    "dataset": "curvas", "task": task, "model": folder.name,
                    "image_id": case.name, "reference_rater": rater,
                    "dice": round(float(scores["dice"]), 4),
                    f"nsd@{TAU_MM}": round(float(scores[f"nsd@{TAU_MM}"]), 4),
                })
    if not rows:
        raise SystemExit(f"no predictions for '{task}' under {PREDICTIONS.relative_to(ROOT)}")

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / f"curvas_model_vs_rater_{task}.csv", index=False)
    ranks = prm.rank_table(frame)
    ranks.to_csv(OUT / f"curvas_model_ranks_{task}.csv", index=False)
    print(ranks.to_string(index=False))

    if frame.model.nunique() < 2:
        # kidney has one admissible model, so there is no ordering to test and none is invented
        print(f"\nonly {frame.model.nunique()} model emits '{task}'; no ranking is reported")
        return

    # the point-estimate ordering agreement between reference raters, the table the
    # RIGA arm reports beside its bootstrap; with three models Kendall's tau can only take the values
    # -1, -1/3, 1/3 and 1, and with two only +/-1, so the interval beside it is what carries the answer
    stability = prm.stability(ranks)
    stability.to_csv(OUT / f"curvas_rank_stability_{task}.csv", index=False)

    intervals, resampled, margin_sd = prm.bootstrap(frame)
    intervals.to_csv(OUT / f"curvas_model_ranks_bootstrap_{task}.csv", index=False)
    resampled.to_csv(OUT / f"curvas_rank_stability_bootstrap_{task}.csv", index=False)
    margins = prm.margin_table(ranks, margin_sd=margin_sd)
    margins.to_csv(OUT / f"curvas_model_margins_{task}.csv", index=False)
    print(f"\nordering agreement between reference raters, {task}:")
    print(resampled.to_string(index=False))
    print(f"\nmargin against the rater-pair spread, {task}:")
    print(margins.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--model", choices=sorted(BUNDLES))
    parser.add_argument("--score", action="store_true",
                        help="score the predictions on disk against each radiologist")
    parser.add_argument("--task", default="pancreas", choices=sorted(CURVAS_LABELS))
    parser.add_argument("--all-tasks", action="store_true",
                        help="write every structure the model emits from the one forward pass")
    parser.add_argument("--limit", type=int, default=None)
    # the testing split holds the 60 cases the challenge leaderboard scored (of its 65), so it is
    # worth covering first in a long run; omitting this covers all three splits
    parser.add_argument("--split", choices=SPLITS, default=None,
                        help="restrict the run to one split. Changes which cases are covered, never how "
                             "any of them is computed.")
    args = parser.parse_args()

    if args.score:
        score(args.task)
        return

    if args.list or not args.model:
        print(f"{len(cases())} CURVAS cases carry an image")
        for name, spec in BUNDLES.items():
            print(f"  {name:20s} {'/'.join(sorted(spec['labels'])):26s} {spec['repo']}\n"
                  f"      {spec['note']}")
        print("\n  excluded as circular: wholeBody_ct_segmentation and "
              "pediatric_abdominal_ct_segmentation, both trained on TotalSegmentator labels, which "
              "seeded the CURVAS annotations")
        return

    tasks = sorted(BUNDLES[args.model]["labels"]) if args.all_tasks else [args.task]
    print(f"{args.model}: task{'s' if len(tasks) > 1 else ''} {', '.join(tasks)}, device {device()}")
    run(args.model, tasks, args.limit, args.split)


if __name__ == "__main__":
    main()
