"""View B: leave-one-rater-out against a consensus of the rest.

Both consensus constructions are here, reported side by side: STAPLE
(SimpleITK STAPLEImageFilter, default parameters, probability thresholded at 0.5) and majority vote
(ties resolved to foreground).

View B is a semantic view. STAPLE and majority vote both produce a binary consensus, so instance
metrics have no counterpart here; instance sources contribute their foreground union.
"""

from __future__ import annotations

import numpy as np

from . import metrics

STAPLE_THRESHOLD = 0.5


def majority_vote(masks: list[np.ndarray]) -> np.ndarray:
    """Foreground where at least half the raters say foreground (ties -> foreground)."""
    if not masks:
        raise ValueError("majority vote needs at least one mask")
    stack = np.stack([metrics.binarise(m) for m in masks])
    return stack.sum(axis=0) >= np.ceil(len(masks) / 2)


def staple_consensus(masks: list[np.ndarray], threshold: float = STAPLE_THRESHOLD) -> np.ndarray:
    """STAPLE probability map thresholded at 0.5.

    STAPLE's assumptions, in warfield-2004's own words:
    conditional independence of raters, initialisation sensitivity (its Brain Phantom experiment
    converged to different solutions depending on initialisation), sensitivity to the global prior,
    and correlated structural errors across a majority of raters not being recoverable."""
    import SimpleITK as sitk

    if len(masks) < 2:
        raise ValueError("STAPLE needs at least two raters")
    images = [sitk.GetImageFromArray(metrics.binarise(m).astype(np.uint8)) for m in masks]
    staple = sitk.STAPLEImageFilter()
    staple.SetForegroundValue(1.0)
    probability = sitk.GetArrayFromImage(staple.Execute(images))
    return probability >= threshold


def view_b_scores(
    masks: dict[str, np.ndarray],
    *,
    spacing: tuple[float, ...] | None = None,
    volumetric: bool = False,
) -> dict[str, dict[str, float]]:
    """Hold out each rater in turn, score it against the consensus of the others.

    Returns {consensus_method: {metric: mean over held-out raters}}. Empty when fewer than 3 raters
    annotated the image: with two raters the held-out rater would be scored against a single other rater,
    which is the pairwise view, so View B is undefined there."""
    rater_ids = sorted(masks)
    if len(rater_ids) < 3:
        return {}

    per_method: dict[str, list[dict[str, float]]] = {"staple": [], "majority": []}
    for held_out in rater_ids:
        others = [masks[r] for r in rater_ids if r != held_out]
        for method, consensus in (
            ("staple", staple_consensus(others)),
            ("majority", majority_vote(others)),
        ):
            per_method[method].append(
                metrics.pairwise_scores(
                    masks[held_out],
                    consensus,
                    instance=False,  # consensus is binary; see the module docstring
                    spacing=spacing,
                    volumetric=volumetric,
                )
            )

    return {
        method: {
            metric: float(np.nanmean([s[metric] for s in scores]))
            if any(np.isfinite(s[metric]) for s in scores)
            else float("nan")
            for metric in scores[0]
        }
        for method, scores in per_method.items()
    }
