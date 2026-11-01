"""Reproduction of the published Mesmer agreement statistic with the authors' own Figure 3d recipe.

    python code/validate_mesmer_fig3d.py

Greenwald et al. 2022 report human-human agreement only as a swarm plot (Fig. 3d); no value is printed.
The reproduction target is therefore the recipe, not a number: re-run the authors' `Figure_3.py` Figure 3d
loop on the same four tissues and five annotators, and set its per-pair F1 beside the layer's instance F1
(one-to-one Hungarian matching at IoU >= 0.5) for the same rater pairs.

Sources, pinned:
  * recipe: github.com/vanvalenlab/publication-figures @ d487de885b07b6b290f792ba58e960f58f6ec4ec,
    `2021-Greenwald_Miller_et_al-Mesmer/Figure_3.py` (Figure 3d block) and `figures.py`
    (`calculate_human_f1_scores`), cloned to data/external/publication-figures
  * matching code: deepcell-toolbox 0.9.1, the version that repository's requirements.txt pins,
    tag 0.9.1 (e2f6e9ebb49b24b57c1308b81cffe78550133712), cloned to data/external/deepcell-toolbox-0.9.1

Changes from running the recipe verbatim, all forced by the 2021 stack not installing under
Python 3.14 (deepcell-toolbox 0.9.1 needs a Cython build and scikit-image < 0.19):
  1. `metrics.py` is loaded on its own rather than through the package `__init__`, which imports
     functions removed from scikit-image (`register_translation`) and OpenCV, neither used here.
  2. The Cython `compute_overlap` (a bounding-box prefilter) is replaced by `_compute_overlap` below,
     a line-for-line NumPy translation, including the `+ 1` pixel-inclusive box convention.
  3. `sklearn.metrics.confusion_matrix` and `erode_edges` are imported by `metrics.py` but used only on
     pixel-statistic and plotting paths this recipe never calls; they are stubbed to raise if called.
  4. `Metrics.calc_object_stats` is replaced by what it does per image, `relabel_sequential` then
     `ObjectAccuracy`, because it accumulates with `DataFrame.append`, removed in pandas 2. The F1
     arithmetic is copied from `figures.calculate_human_f1_scores` unchanged.
  5. Float-typed label files (geneva in two tissues) are cast to integers after checking that every
     value is integral, as `gtu.io.load_mask` does, since `relabel_sequential` rejects floats.
The matching rule itself (cutoff1 = 0.4, cutoff2 = 0.1, linear assignment, loner graph) is untouched.

Writes results/e5_mesmer_fig3d.csv (per pair) and prints per-tissue and overall means.
"""
from __future__ import annotations

import importlib.util
import itertools
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/gtu-raw/mesmer"
TOOLBOX = ROOT / "data/external/deepcell-toolbox-0.9.1/deepcell_toolbox"
OUT = ROOT / "results/e5_mesmer_fig3d.csv"

#: Figure_3.py, Figure 3d block: the folders and the names the authors gave them
FOLDERS = ["DCIS_2328", "Eliot_Point17", "P101_T3_T4_Point2", "cHL_Point8908"]
FOLDER_NAMES = ["DCIS_MIBI", "Colon_IF", "Esophagus_MIBI", "Hodgekins_Vectra"]


def _compute_overlap(boxes: np.ndarray, query_boxes: np.ndarray) -> np.ndarray:
    """NumPy translation of deepcell_toolbox/compute_overlap.pyx (Fast R-CNN, S. Karayev)."""
    overlaps = np.zeros((boxes.shape[0], query_boxes.shape[0]), dtype=np.float64)
    for k in range(query_boxes.shape[0]):
        box_area = (query_boxes[k, 2] - query_boxes[k, 0] + 1) * (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        for n in range(boxes.shape[0]):
            iw = min(boxes[n, 2], query_boxes[k, 2]) - max(boxes[n, 0], query_boxes[k, 0]) + 1
            if iw > 0:
                ih = min(boxes[n, 3], query_boxes[k, 3]) - max(boxes[n, 1], query_boxes[k, 1]) + 1
                if ih > 0:
                    ua = np.float64(
                        (boxes[n, 2] - boxes[n, 0] + 1) * (boxes[n, 3] - boxes[n, 1] + 1)
                        + box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def _unused(*_args, **_kwargs):
    raise NotImplementedError("stubbed: not on the Figure 3d path (see module docstring, item 3)")


def load_deepcell_metrics():
    """Load deepcell-toolbox 0.9.1 metrics.py with the shims described in the docstring."""
    package = types.ModuleType("deepcell_toolbox")
    package.__path__ = [str(TOOLBOX)]
    package.erode_edges = _unused
    overlap = types.ModuleType("deepcell_toolbox.compute_overlap")
    overlap.compute_overlap = _compute_overlap
    overlap_3d = types.ModuleType("deepcell_toolbox.compute_overlap_3D")
    overlap_3d.compute_overlap_3D = _unused
    sys.modules.update({
        "deepcell_toolbox": package,
        "deepcell_toolbox.compute_overlap": overlap,
        "deepcell_toolbox.compute_overlap_3D": overlap_3d,
    })
    if importlib.util.find_spec("sklearn") is None:
        sklearn = types.ModuleType("sklearn")
        sklearn_metrics = types.ModuleType("sklearn.metrics")
        sklearn_metrics.confusion_matrix = _unused
        sys.modules.update({"sklearn": sklearn, "sklearn.metrics": sklearn_metrics})
    spec = importlib.util.spec_from_file_location("deepcell_toolbox.metrics", TOOLBOX / "metrics.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["deepcell_toolbox.metrics"] = module
    spec.loader.exec_module(module)
    return module


def _read_label(path: Path) -> np.ndarray:
    import tifffile

    image = tifffile.imread(path)
    if not np.issubdtype(image.dtype, np.integer):
        if not np.all(np.equal(np.mod(image, 1), 0)):
            raise ValueError(f"non-integral label values in {path}")
        image = image.astype(np.int64)
    return image


def authors_pair_f1(dc_metrics, img1: np.ndarray, img2: np.ndarray) -> float:
    """figures.calculate_human_f1_scores for one pair, with calc_object_stats inlined (docstring 4)."""
    from skimage.segmentation import relabel_sequential

    true_relabel, _, _ = relabel_sequential(img1)
    pred_relabel, _, _ = relabel_sequential(img2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o = dc_metrics.ObjectAccuracy(true_relabel, pred_relabel, cutoff1=0.4, cutoff2=0.1, seg=False)
    recall = o.correct_detections / o.n_true
    precision = o.correct_detections / o.n_pred
    return 2 * precision * recall / (precision + recall)


def main() -> None:
    dc_metrics = load_deepcell_metrics()
    pairs = pd.read_parquet(ROOT / "results/layer/pairs.parquet")
    layer = pairs[pairs.dataset == "mesmer"].set_index(["image_id", "rater_a", "rater_b"])["instance_f1@0.5"]

    rows = []
    for folder, name in zip(FOLDERS, FOLDER_NAMES):
        directory = RAW / folder / "annotations"
        # Figure_3.py lists files containing '.tiff' (ark io_utils.list_files): this takes the five
        # annotators and excludes `rgb_bri_labels.tif`, a visualisation. Sorted for reproducibility.
        files = sorted(p for p in directory.iterdir() if ".tiff" in p.name)
        names = [p.name.split("_labels")[0] for p in files]
        images = [_read_label(p) for p in files]
        for (i, a), (j, b) in itertools.combinations(enumerate(names), 2):
            f1_recipe = authors_pair_f1(dc_metrics, images[i], images[j])
            key = (folder, *sorted((a, b)))
            f1_layer = float(layer.get(key, np.nan))
            rows.append({"tissue": folder, "tissue_name": name, "rater_a": key[1], "rater_b": key[2],
                         "f1_authors_recipe": f1_recipe, "f1_layer_iou05": f1_layer,
                         "difference": f1_layer - f1_recipe})
        print(f"  {folder:18s} {len(names)} annotators: {', '.join(names)}")

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT, index=False)
    missing = frame.f1_layer_iou05.isna().sum()
    print(f"\nMesmer reproduction, {len(frame)} rater pairs over {frame.tissue.nunique()} tissues "
          f"({missing} pairs without a layer value)")
    for tissue, part in frame.groupby("tissue", sort=False):
        print(f"  {tissue:18s} recipe F1 {part.f1_authors_recipe.mean():.3f}   "
              f"layer F1@0.5 {part.f1_layer_iou05.mean():.3f}")
    print(f"  {'all pairs':18s} recipe F1 {frame.f1_authors_recipe.mean():.3f}   "
          f"layer F1@0.5 {frame.f1_layer_iou05.mean():.3f}   "
          f"mean difference {frame.difference.mean():+.3f}")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
