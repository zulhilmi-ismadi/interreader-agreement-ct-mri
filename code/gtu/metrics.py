"""Agreement metrics. Every definition, tolerance, threshold, seed and degenerate-case rule used by the
layer is defined here, in one place: the overlap, boundary and instance metrics, the reference-substitution
spread and the bootstrap intervals. Every downstream script calls these functions rather than restating them."""

from __future__ import annotations

import itertools

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

# NSD tolerances: 2 px for 2D images and 2 mm for CT and MRI volumes, with 1 and 4 px or 1 and 3 mm as
# sensitivity values. Set by modality, never fitted to the data.
TAU_PRIMARY_PX = 2.0
TAU_SENSITIVITY_PX = (1.0, 4.0)
TAU_PRIMARY_MM = 2.0
TAU_SENSITIVITY_MM = (1.0, 3.0)

# Instance matching: one-to-one at IoU >= 0.5, with 0.7 as the sensitivity threshold.
INSTANCE_IOU_PRIMARY = 0.5
INSTANCE_IOU_SENSITIVITY = 0.7

# Bootstrap: 1,000 resamples, percentile 95% interval, this seed.
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 20260903


def tau_grid(volumetric: bool) -> tuple[float, ...]:
    """Primary tolerance first, then the two sensitivity values used by the tolerance sensitivity analysis."""
    if volumetric:
        return (TAU_PRIMARY_MM,) + TAU_SENSITIVITY_MM
    return (TAU_PRIMARY_PX,) + TAU_SENSITIVITY_PX


def binarise(mask: np.ndarray) -> np.ndarray:
    """Foreground of a pixel mask, or the union of all instances of a label image."""
    return np.asarray(mask) > 0


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = binarise(a), binarise(b)
    _check_shapes(a, b)
    total = a.sum() + b.sum()
    if total == 0:
        return 1.0  # both empty: two raters who both mark nothing agree
    return float(2.0 * np.logical_and(a, b).sum() / total)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = binarise(a), binarise(b)
    _check_shapes(a, b)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0  # both empty: two raters who both mark nothing agree
    return float(np.logical_and(a, b).sum() / union)


def surface(mask: np.ndarray) -> np.ndarray:
    """Foreground voxels with a background neighbour. Connectivity 1; the array border counts as
    background, so a structure running off the field of view has surface along that edge."""
    return _surface_of_binary(binarise(mask))


def _surface_of_binary(mask: np.ndarray) -> np.ndarray:
    eroded = ndimage.binary_erosion(
        mask, ndimage.generate_binary_structure(mask.ndim, 1), border_value=0
    )
    return mask & ~eroded


def _union_window(a: np.ndarray, b: np.ndarray, pad: int = 1) -> tuple[slice, ...]:
    """Smallest box containing both masks, padded by one voxel and clipped to the array.

    Cropping to this window leaves every distance in `surface_distances` unchanged: both surfaces lie
    inside it, so each one's nearest counterpart does too. The padding preserves the border rule of
    `surface` — where the structure does not reach the array edge the pad supplies the background
    margin that makes it a surface, and where it does reach the edge the clip keeps the crop edge on
    the array edge, which is the same situation the uncropped array was in.

    C3RO is why this exists: an ROI can occupy 0.1% of a 512x512x140 volume, and the full-volume
    distance transform dominates the runtime by three orders of magnitude."""
    coordinates = np.argwhere(a | b)
    low = np.maximum(coordinates.min(axis=0) - pad, 0)
    high = np.minimum(coordinates.max(axis=0) + pad + 1, a.shape)
    return tuple(slice(int(lo), int(hi)) for lo, hi in zip(low, high))


def surface_distances(
    a: np.ndarray, b: np.ndarray, spacing: tuple[float, ...] | None = None
) -> np.ndarray | None:
    """Pooled two-directional surface distances, or None where either mask is empty."""
    a_binary, b_binary = binarise(a), binarise(b)
    _check_shapes(a_binary, b_binary)
    if not a_binary.any() or not b_binary.any():
        return None

    window = _union_window(a_binary, b_binary)
    surf_a = _surface_of_binary(a_binary[window])
    surf_b = _surface_of_binary(b_binary[window])
    to_b = ndimage.distance_transform_edt(~surf_b, sampling=spacing)
    to_a = ndimage.distance_transform_edt(~surf_a, sampling=spacing)
    return np.concatenate([to_b[surf_a], to_a[surf_b]])


def hd95(a: np.ndarray, b: np.ndarray, spacing: tuple[float, ...] | None = None) -> float:
    """95th percentile of the pooled two-directional surface distances (symmetric).
    Both empty -> 0; exactly one empty -> NaN, excluded from aggregation."""
    if _both_empty(a, b):
        return 0.0
    distances = surface_distances(a, b, spacing)
    if distances is None:
        return float("nan")
    return float(np.percentile(distances, 95))


def nsd(
    a: np.ndarray, b: np.ndarray, tau: float, spacing: tuple[float, ...] | None = None
) -> float:
    """Normalised surface Dice at tolerance tau. Both empty -> 1; exactly one empty -> 0
    (note this differs from HD95, which is NaN when one mask is empty)."""
    if _both_empty(a, b):
        return 1.0
    distances = surface_distances(a, b, spacing)
    if distances is None:
        return 0.0
    return float((distances <= tau).mean())


def nsd_from_distances(
    distances: np.ndarray | None, tau: float, *, both_empty: bool = False
) -> float:
    """NSD at tau from an already-computed pooled distance array, applying the empty-mask rules of `nsd`."""
    if both_empty:
        return 1.0
    if distances is None:
        return 0.0
    return float((distances <= tau).mean())


def tau95_for_task(
    pair_distances: list[np.ndarray | None],
    both_empty_flags: list[bool] | None = None,
    *,
    tolerance: float = 1e-3,
) -> float:
    """Smallest tau at which the MEAN pairwise NSD for a task reaches 0.95 — "the tolerance a
    benchmark would need". Descriptive only, never fed back into NSD.

    This is a task-level quantity, not a per-pair one: it inverts the mean of the pairs' NSD curves.
    (For a single pair it would collapse to that pair's HD95, which is why it is not a pair metric.)"""
    if both_empty_flags is None:
        both_empty_flags = [False] * len(pair_distances)
    if not pair_distances:
        return float("nan")

    def mean_nsd(tau: float) -> float:
        return float(
            np.mean(
                [
                    nsd_from_distances(d, tau, both_empty=e)
                    for d, e in zip(pair_distances, both_empty_flags)
                ]
            )
        )

    finite = [d.max() for d in pair_distances if d is not None and d.size]
    if not finite:
        return float("nan")
    high = float(max(finite))
    if mean_nsd(high) < 0.95:
        return float("nan")  # unreachable: one-empty pairs cap the mean below 0.95

    low = 0.0
    while high - low > tolerance:
        middle = (low + high) / 2
        if mean_nsd(middle) >= 0.95:
            high = middle
        else:
            low = middle
    return high


def instance_iou_matrix(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IoU of every instance of `a` against every instance of `b`. Instance ids come from np.unique
    and need not be contiguous (CryoNuSeg ids have gaps)."""
    a, b = np.asarray(a), np.asarray(b)
    _check_shapes(a, b)
    a_ids = np.unique(a)
    a_ids = a_ids[a_ids != 0]
    b_ids = np.unique(b)
    b_ids = b_ids[b_ids != 0]
    n, m = len(a_ids), len(b_ids)
    if n == 0 or m == 0:
        return np.zeros((n, m)), a_ids, b_ids

    a_dense = np.searchsorted(a_ids, a) + 1
    a_dense[a == 0] = 0
    b_dense = np.searchsorted(b_ids, b) + 1
    b_dense[b == 0] = 0

    joint = np.bincount(
        a_dense.ravel().astype(np.int64) * (m + 1) + b_dense.ravel(),
        minlength=(n + 1) * (m + 1),
    ).reshape(n + 1, m + 1)
    intersection = joint[1:, 1:]
    union = joint.sum(axis=1)[1:, None] + joint.sum(axis=0)[None, 1:] - intersection
    matrix = np.divide(
        intersection, union, out=np.zeros_like(intersection, dtype=float), where=union > 0
    )
    return matrix, a_ids, b_ids


def _match_instances(
    a: np.ndarray, b: np.ndarray, threshold: float
) -> tuple[np.ndarray, int, int]:
    """One-to-one Hungarian matching on IoU, keeping pairs at IoU >= threshold.
    Returns the matched IoUs and the counts of unmatched instances in a and in b."""
    matrix, a_ids, b_ids = instance_iou_matrix(a, b)
    n, m = len(a_ids), len(b_ids)
    if n == 0 or m == 0:
        return np.empty(0), n, m
    eligible = np.where(matrix >= threshold, matrix, 0.0)
    rows, cols = linear_sum_assignment(-eligible)
    matched = eligible[rows, cols]
    matched = matched[matched >= threshold]
    return matched, n - len(matched), m - len(matched)


def instance_f1(a: np.ndarray, b: np.ndarray, threshold: float = INSTANCE_IOU_PRIMARY) -> float:
    """Instance F1 under one-to-one matching at the IoU threshold. Both empty -> 1; exactly one empty -> 0."""
    matched, false_negatives, false_positives = _match_instances(a, b, threshold)
    true_positives = len(matched)
    if true_positives == 0 and false_negatives == 0 and false_positives == 0:
        return 1.0
    return float(
        2 * true_positives / (2 * true_positives + false_positives + false_negatives)
    )


def panoptic_quality(
    a: np.ndarray, b: np.ndarray, threshold: float = INSTANCE_IOU_PRIMARY
) -> float:
    """Segmentation quality times recognition quality (a secondary instance metric):
    sum of matched IoU over (TP + FP/2 + FN/2). Both empty -> 1; exactly one empty -> 0."""
    matched, false_negatives, false_positives = _match_instances(a, b, threshold)
    true_positives = len(matched)
    if true_positives == 0 and false_negatives == 0 and false_positives == 0:
        return 1.0
    denominator = true_positives + 0.5 * false_positives + 0.5 * false_negatives
    return float(matched.sum() / denominator) if denominator else 0.0


def pairwise_scores(
    a: np.ndarray,
    b: np.ndarray,
    *,
    instance: bool,
    spacing: tuple[float, ...] | None = None,
    volumetric: bool = False,
) -> dict[str, float]:
    """Every agreement metric for one unordered rater pair on one image."""
    both_empty = _both_empty(a, b)
    distances = None if both_empty else surface_distances(a, b, spacing)

    scores: dict[str, float] = {
        "dice": dice(a, b),
        "iou": iou(a, b),
        "hd95": 0.0
        if both_empty
        else (float("nan") if distances is None else float(np.percentile(distances, 95))),
    }
    for tau in tau_grid(volumetric):
        scores[f"nsd@{tau:g}"] = nsd_from_distances(distances, tau, both_empty=both_empty)

    if instance:
        for threshold in (INSTANCE_IOU_PRIMARY, INSTANCE_IOU_SENSITIVITY):
            scores[f"instance_f1@{threshold:g}"] = instance_f1(a, b, threshold)
        scores["panoptic_quality"] = panoptic_quality(a, b)
    return scores


def reference_substitution_spread(
    scores: dict[tuple[str, str], float], *, statistic: str = "range"
) -> float:
    """Reference-substitution spread. `scores` maps (prediction_rater, reference_rater) -> metric value for ONE
    image. For each rater taken as the prediction, take the spread of its scores across the other
    raters used as reference; return the mean of those spreads over raters.

    Undefined (NaN) with fewer than 3 raters, since every prediction needs >= 2 references."""
    predictions: dict[str, list[float]] = {}
    for (prediction, _reference), value in scores.items():
        if np.isfinite(value):
            predictions.setdefault(prediction, []).append(value)

    spreads = []
    for values in predictions.values():
        if len(values) < 2:
            continue
        array = np.asarray(values, dtype=float)
        spreads.append(
            float(array.max() - array.min()) if statistic == "range" else float(array.std(ddof=1))
        )
    return float(np.mean(spreads)) if spreads else float("nan")


def rss_scores_for_image(
    masks: dict[str, np.ndarray], metric, **kwargs
) -> dict[tuple[str, str], float]:
    """Score every ordered (prediction, reference) rater pair on one image, for use with
    `reference_substitution_spread`. `metric` is symmetric, so each unordered pair is computed once."""
    scores: dict[tuple[str, str], float] = {}
    for first, second in itertools.combinations(sorted(masks), 2):
        value = metric(masks[first], masks[second], **kwargs)
        scores[(first, second)] = value
        scores[(second, first)] = value
    return scores


def bootstrap_ci(
    values: np.ndarray,
    *,
    clusters: np.ndarray | None = None,
    n_samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """95% percentile CI of the mean over images, resampling images or, where images nest in a higher
    unit, whole clusters. NaNs are dropped; the caller reports their count."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    values = values[finite]
    if values.size == 0:
        return (float("nan"), float("nan"))

    # The RNG draws by index, so an unsorted input would make the interval depend on the order rows
    # happened to arrive in — which differed between a fresh run and a checkpoint-resumed one, since
    # Windows sorts Path objects case-insensitively. Sorting first makes the interval a function of
    # the multiset of values and the seed alone.
    rng = np.random.default_rng(seed)
    if clusters is None:
        values = np.sort(values, kind="stable")
        means = values[rng.integers(0, values.size, size=(n_samples, values.size))].mean(axis=1)
    else:
        clusters = np.asarray(clusters)[finite]
        order = np.lexsort((values, clusters))
        values, clusters = values[order], clusters[order]
        unique = np.unique(clusters)
        members = [np.flatnonzero(clusters == c) for c in unique]
        means = np.empty(n_samples)
        for i in range(n_samples):
            picked = rng.integers(0, len(unique), size=len(unique))
            means[i] = values[np.concatenate([members[p] for p in picked])].mean()
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def rater_bootstrap_ci(
    per_pair: dict[tuple[str, str], float],
    raters: list[str],
    *,
    n_samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """95% CI from resampling raters with replacement and recomputing the pairs among them
    (for sources with too few images to bootstrap over images)."""
    rng = np.random.default_rng(seed)
    lookup = {frozenset(pair): value for pair, value in per_pair.items()}
    means = []
    for _ in range(n_samples):
        drawn = [raters[i] for i in rng.integers(0, len(raters), size=len(raters))]
        values = [
            lookup[frozenset((x, y))]
            for x, y in itertools.combinations(drawn, 2)
            if x != y and frozenset((x, y)) in lookup
        ]
        values = [v for v in values if np.isfinite(v)]
        if values:
            means.append(float(np.mean(values)))
    if not means:
        return (float("nan"), float("nan"))
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def _both_empty(a: np.ndarray, b: np.ndarray) -> bool:
    a, b = binarise(a), binarise(b)
    _check_shapes(a, b)
    return not a.any() and not b.any()


def _check_shapes(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(f"raters disagree on image shape: {a.shape} vs {b.shape}")
