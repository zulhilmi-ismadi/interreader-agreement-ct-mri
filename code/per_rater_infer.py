"""Run off-the-shelf segmentation models on the images behind a multi-rater floor.

Companion to `per_rater_models.py`, which scores what this writes. The split is deliberate: inference
needs a GPU and downloaded weights, scoring needs neither, so the analysis can be re-run and checked by
anyone without repeating the model pass.

**What a model is allowed to see.** RIGA ships the raters' contours drawn ON the photograph; the only
clean input is the unmarked `<base>prime` image, which is what `riga_index.csv` records. A model never
sees a rater's mask, and nothing here is trained or fine-tuned: weights are downloaded at a pinned
revision, run once per image, and written out. The revision is recorded beside every prediction so the
release can say exactly what produced a number.

**Why these models.** Two published optic disc/cup models from different groups and architectures,
and one prompt-driven general model (MedSAM) as a deliberately weaker, different-family baseline. If a
model cannot be obtained or cannot emit the structure, it is dropped and the drop is reported with its
reason — never silently replaced.

    python code/per_rater_infer.py --list                 # show the model registry and resolve weights
    python code/per_rater_infer.py --model segformer_odoc --task disc [--limit 20]

Writes results/per_rater/predictions/<model>/<image_id>__<task>.tif (binary, at the image's own
resolution) and results/per_rater/predictions/<model>/_model.json (identifier, revision, device, date).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

OUT = ROOT / "results" / "per_rater"
PREDICTIONS = OUT / "predictions"
INDEX = OUT / "riga_index.csv"
#: weights are cached outside the repository and outside the release; only scores and code ship
CACHE = ROOT / ".model-cache"

#: The registry is the record of what was run. `labels` maps the model's own class indices onto the
#: two structures RIGA annotates; a model that cannot produce one of them is dropped for that task.
MODELS = {
    "segformer_odoc": {
        "repo": "pamixsun/segformer_for_optic_disc_cup_segmentation",
        "kind": "hf-semantic-segmentation",
        # config: 0 Background, 1 Optic disc, 2 Optic cup. RIGA's own masks nest the cup inside the
        # disc, so the disc is the union of the two classes — but only if the model carves the cup
        # out of the disc rather than painting the disc somewhere else. Checked on predictions
        # by geometry alone, reading no rater mask: on three images
        # every class-2 pixel fell inside the hole-filled class-1 region (1.000, 1.000, 1.000) and
        # class 1 was an annulus in each, with cup-to-disc area ratios of 0.21, 0.42 and 0.31.
        # The union is therefore the disc, and class 2 alone the cup.
        "labels": {"disc": (1, 2), "cup": (2,)},
        "note": "SegFormer trained for optic disc and cup segmentation on fundus photographs",
    },
    "segformer_refuge": {
        "repo": "SiddharthaM/segformer-refuge-disc-segmentation",
        "kind": "hf-semantic-segmentation",
        "labels": {"disc": (1,)},  # config: 0 unlabeled, 1 disc
        # This repo ships no preprocessor_config.json, so the processor cannot be read from it; the
        # base SegFormer processor is used instead, which is that model's own training-time transform.
        "processor": "nvidia/mit-b0",
        "note": "SegFormer trained on REFUGE by a different group from segformer_odoc; disc only",
        # Dropped from the set of models. Its `disc` class covers
        # 48-50% of the frame; scored against the fundus field of view — the illuminated circle inside
        # the black border — it gives Dice 0.938, 0.925, 0.938 on three images, while the optic disc is
        # under 1% of the frame. It segments the camera's field of view, not the structure its config
        # labels `disc`. The weights load cleanly (0 missing, 0 unexpected, 0 mismatched keys, and the
        # decode head comes from the checkpoint), so this is the model's real behaviour, not a loading
        # fault, and no preprocessing choice relocates a mask onto a different structure at 0.93
        # agreement. Kept in the registry so the record shows what was tried and why it went.
        "dropped": "segments the fundus field of view, not the optic disc",
    },
    "medsam": {
        "repo": "wanglab/medsam-vit-base",
        "kind": "sam-box-prompt",
        "labels": {"disc": (1,), "cup": (1,)},
        "note": "MedSAM prompted with a fixed box on the brightest region of the green channel; different family, weaker baseline",
    },
    "unet_drishti": {
        "repo": "Yashu13/od-cup-segmentation-glaucoma-detection",
        "kind": "smp-unet-binary",
        # Added to replace the dropped segformer_refuge. This repository ships bare .pth state dicts,
        # which are usable only when the training transform is known; unlike the ClementP family it
        # also ships the notebook that produced them, so nothing has to be guessed. From that notebook: the architecture is
        # smp.Unet(resnet34, in_channels=3, classes=1) built once per structure; images are read RGB,
        # resized to (256, 256) bilinear, transposed to CHW and divided by 255 with NO ImageNet mean
        # or std; the head is thresholded at sigmoid > 0.5 and reduced to its largest connected
        # component; and the weights were written with torch.save(model.state_dict()). Both files load
        # into that architecture with 0 missing and 0 unexpected keys, so the recipe is confirmed and
        # not reconstructed. Trained on Drishti-GS, which is neither RIGA nor the other model's data.
        "weights": {"disc": "model_od.pth", "cup": "model_cup.pth"},
        "encoder": "resnet34",
        "input_size": (256, 256),
        "labels": {"disc": (1,), "cup": (1,)},
        "note": "ResNet34 U-Net trained on Drishti-GS; separate disc and cup weight files",
    },
}


def load_index() -> pd.DataFrame:
    if not INDEX.is_file():
        raise SystemExit(f"{INDEX.relative_to(ROOT)} is missing; run per_rater_models.py --index first")
    return pd.read_csv(INDEX)


def resolve(model: str) -> dict:
    spec = MODELS.get(model)
    if spec is None:
        raise SystemExit(f"unknown model '{model}'; known: {', '.join(MODELS)}")
    if spec.get("dropped"):
        # A dropped model stays in the registry as a record, but must not be runnable: a prediction
        # made after the drop could only ever be used to reopen a decision that was taken on other
        # grounds.
        raise SystemExit(f"'{model}' was dropped from the panel: {spec['dropped']}")
    return spec


def fetch(spec: dict) -> tuple[str, str]:
    """Download a model at a pinned revision and return (local path, revision)."""
    from huggingface_hub import snapshot_download

    CACHE.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(spec["repo"], cache_dir=str(CACHE))
    revision = Path(path).name  # the snapshot hash the hub resolved to
    return path, revision


def _device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def run_semantic(spec: dict, index: pd.DataFrame, task: str, limit: int | None) -> tuple[int, str]:
    """Run a Hugging Face semantic-segmentation model over the unmarked images."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

    wanted = spec["labels"].get(task)
    if wanted is None:
        raise SystemExit(f"model cannot produce '{task}' — drop it for this task and report the reason")

    path, revision = fetch(spec)
    # Some repositories ship weights and a config but no preprocessor config. Reading the processor
    # from such a repo fails, and inventing a transform would misrepresent the model, so the registry
    # names the processor to use and that choice is recorded with the predictions.
    processor = AutoImageProcessor.from_pretrained(spec.get("processor", path))
    model = AutoModelForSemanticSegmentation.from_pretrained(path).eval().to(_device())

    written = 0
    rows = index if limit is None else index.head(limit)
    for _, row in rows.iterrows():
        image = Image.open(row.image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(_device())
        with torch.no_grad():
            logits = model(**inputs).logits
        # back to the image's own pixel grid: the masks were recovered at that resolution, and the
        # layer never resamples a mask
        logits = torch.nn.functional.interpolate(
            logits, size=(image.size[1], image.size[0]), mode="bilinear", align_corners=False
        )
        prediction = logits.argmax(dim=1)[0].cpu().numpy()
        mask = np.isin(prediction, wanted)
        _write(spec, row.image_id, task, mask)
        written += 1
    return written, revision


def run_sam_box(spec: dict, index: pd.DataFrame, task: str, limit: int | None) -> tuple[int, str]:
    """Run a SAM-family model prompted with a box, as MedSAM is meant to be used.

    MedSAM segments what it is pointed at; it does not know what an optic disc is. Pointing it at the
    right place is therefore part of the model, not a favour to it, and the prompt must be fixed by a
    rule rather than by anything a rater drew. The rule here: the brightest region of the green
    channel, which is where the optic nerve head sits in a fundus photograph, dilated to a square box.
    No rater mask is read, and the same rule is applied to every image.

    This is the deliberately weaker baseline of the three. A prompt-driven general model given a
    crude, automatic prompt should not beat a model trained for this structure, and if it does, that
    is itself worth reporting.
    """
    import torch
    from PIL import Image
    from transformers import AutoProcessor, SamModel

    path, revision = fetch(spec)
    processor = AutoProcessor.from_pretrained(spec.get("processor", path))
    model = SamModel.from_pretrained(path).eval().to(_device())

    written = 0
    rows = index if limit is None else index.head(limit)
    for _, row in rows.iterrows():
        image = Image.open(row.image_path).convert("RGB")
        box = _optic_nerve_box(np.array(image))
        inputs = processor(image, input_boxes=[[box]], return_tensors="pt").to(_device())
        with torch.no_grad():
            outputs = model(**inputs, multimask_output=False)
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
        )
        mask = masks[0][0, 0].numpy().astype(bool)
        _write(spec, row.image_id, task, mask)
        written += 1
    return written, revision


def _optic_nerve_box(image: np.ndarray, fraction: float = 0.18) -> list[int]:
    """A box around the brightest area of the green channel, the usual optic-disc cue in fundus.

    Deterministic, uses no annotation, and is reported with the results so a reader can see exactly
    how the prompt was chosen.
    """
    from scipy.ndimage import uniform_filter

    green = image[..., 1].astype(np.float32)
    smoothed = uniform_filter(green, size=max(image.shape[:2]) // 20 or 1)
    y, x = np.unravel_index(int(np.argmax(smoothed)), smoothed.shape)
    half = int(fraction * max(image.shape[:2]) / 2)
    height, width = image.shape[:2]
    return [max(0, x - half), max(0, y - half), min(width - 1, x + half), min(height - 1, y + half)]


def run_smp_unet_binary(spec: dict, index: pd.DataFrame, task: str, limit: int | None) -> tuple[int, str]:
    """Run a segmentation_models_pytorch U-Net whose training recipe is published in full.

    A bare state dict is only usable when the transform that produced it is known: the same weights
    fed differently normalised pixels are a different model, and scoring that would misrepresent it.
    Here the model author's notebook states every step, so each one below is theirs, not ours — RGB, resize
    to the training size, divide by 255 with no further normalisation, sigmoid at 0.5, largest
    connected component. The one thing the notebook does not cover is returning to the image's own
    grid, since it evaluated at 256x256; the logits are interpolated before thresholding, exactly as
    the semantic-segmentation path above does, so no mask is resampled after it is made binary.
    """
    import segmentation_models_pytorch as smp
    import torch
    from PIL import Image

    filename = spec["weights"].get(task)
    if filename is None:
        raise SystemExit(f"model cannot produce '{task}' — drop it for this task and report the reason")

    path, revision = fetch(spec)
    model = smp.Unet(in_channels=3, classes=1, encoder_name=spec["encoder"], encoder_weights=None)
    state = torch.load(Path(path) / filename, map_location="cpu", weights_only=True)
    # strict: a silently partial load would leave a randomly initialised head, which would look like
    # a bad model rather than a broken one
    model.load_state_dict(state)
    model = model.eval().to(_device())

    written = 0
    rows = index if limit is None else index.head(limit)
    for _, row in rows.iterrows():
        image = Image.open(row.image_path).convert("RGB")
        small = image.resize(tuple(spec["input_size"]), Image.BILINEAR)
        array = np.asarray(small).transpose(2, 0, 1).astype(np.float32) / 255.0
        tensor = torch.from_numpy(array)[None].to(_device())
        with torch.no_grad():
            logits = model(tensor)
        logits = torch.nn.functional.interpolate(
            logits, size=(image.size[1], image.size[0]), mode="bilinear", align_corners=False
        )
        mask = torch.sigmoid(logits)[0, 0].cpu().numpy() > 0.5
        _write(spec, row.image_id, task, _largest_component(mask))
        written += 1
    return written, revision


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep the largest connected region, as this model's own inference code does."""
    from scipy.ndimage import label

    if not mask.any():
        return mask
    labelled, count = label(mask)
    if count <= 1:
        return mask
    sizes = np.bincount(labelled.ravel())
    sizes[0] = 0
    return labelled == int(sizes.argmax())


def _write(spec: dict, image_id: str, task: str, mask: np.ndarray) -> None:
    from PIL import Image

    folder = PREDICTIONS / _folder_name(spec)
    folder.mkdir(parents=True, exist_ok=True)
    # A binary mask at fundus resolution is about 3.7 MB uncompressed, and the panel writes several
    # per image; deflate is lossless, so the mask that gets scored is bit-for-bit the mask the model
    # produced, and the set fits in a fraction of the space.
    Image.fromarray(mask.astype(np.uint8)).save(
        folder / f"{image_id}__{task}.tif", compression="tiff_deflate"
    )


def _folder_name(spec: dict) -> str:
    return next(name for name, value in MODELS.items() if value is spec)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="show the registry and resolve weights")
    parser.add_argument("--model", choices=sorted(MODELS))
    parser.add_argument("--task", default="disc", choices=("disc", "cup"))
    parser.add_argument("--limit", type=int, default=None, help="run on the first N images only")
    args = parser.parse_args()

    if args.list or not args.model:
        print(f"{'model':18s} {'task support':16s} repo")
        for name, spec in MODELS.items():
            print(f"{name:18s} {'/'.join(spec['labels']):16s} {spec['repo']}\n    {spec['note']}")
        return

    spec = resolve(args.model)
    index = load_index()
    print(f"{args.model}: {len(index) if args.limit is None else args.limit} images, task '{args.task}', "
          f"device {_device()}")
    if spec["kind"] == "hf-semantic-segmentation":
        written, revision = run_semantic(spec, index, args.task, args.limit)
    elif spec["kind"] == "sam-box-prompt":
        written, revision = run_sam_box(spec, index, args.task, args.limit)
    elif spec["kind"] == "smp-unet-binary":
        written, revision = run_smp_unet_binary(spec, index, args.task, args.limit)
    else:
        raise SystemExit(f"kind '{spec['kind']}' is not implemented yet; see the model registry")

    # One folder holds both tasks, so the record is keyed by task: running `cup` after `disc` must add
    # to the provenance, not erase the disc run whose masks are sitting in the same folder.
    folder = PREDICTIONS / args.model
    record_path = folder / "_model.json"
    record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.is_file() else {}
    record.update(
        {"model": args.model, "repo": spec["repo"], "revision": revision,
         # the processor is part of what produced these masks: for a repo that ships no
         # preprocessor config it comes from elsewhere, and that must be visible, not implied
         "processor": spec.get("processor", spec["repo"]),
         "device": _device(), "note": spec["note"]}
    )
    record.setdefault("tasks", {})[args.task] = {
        "labels_used": list(spec["labels"].get(args.task, ())),
        "images": written,
        "run_date": date.today().isoformat(),
    }
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"wrote {written} masks to {folder.relative_to(ROOT)} (revision {revision})")


if __name__ == "__main__":
    main()
