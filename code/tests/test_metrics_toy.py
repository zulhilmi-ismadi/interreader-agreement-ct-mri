"""Hand-computable validation of every metric against its definition.
Real data are scored only when this passes. It covers perfect overlap, disjoint, one-empty,
both-empty, 1-px shift and known instance counts.

Runs under pytest or standalone: python code/tests/test_metrics_toy.py"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gtu import metrics


# --------------------------------------------------------------------------- method constants


def test_method_constants():
    """NSD tolerance grids, instance-match thresholds and the bootstrap settings. These define the
    method and are not tunables."""
    assert metrics.tau_grid(volumetric=False) == (2.0, 1.0, 4.0)
    assert metrics.tau_grid(volumetric=True) == (2.0, 1.0, 3.0)
    assert metrics.INSTANCE_IOU_PRIMARY == 0.5
    assert metrics.INSTANCE_IOU_SENSITIVITY == 0.7
    assert metrics.BOOTSTRAP_SAMPLES == 1000
    assert metrics.BOOTSTRAP_SEED == 20260903


# --------------------------------------------------------------------------- overlap


def test_dice_and_iou_hand_computed():
    a = np.zeros((10, 10), dtype=np.uint8)
    b = np.zeros((10, 10), dtype=np.uint8)
    a[0:5, :] = 1  # 50 px
    b[2:7, :] = 1  # 50 px, overlap rows 2-4 = 30 px, union rows 0-6 = 70 px
    assert metrics.dice(a, b) == 2 * 30 / 100
    assert metrics.iou(a, b) == 30 / 70


def test_disjoint_masks_score_zero():
    a = np.zeros((10, 10), dtype=np.uint8)
    b = np.zeros((10, 10), dtype=np.uint8)
    a[0:3, 0:3] = 1
    b[7:10, 7:10] = 1
    assert metrics.dice(a, b) == 0.0
    assert metrics.iou(a, b) == 0.0


def test_dice_iou_monotone_relation():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = rng.integers(0, 2, size=(16, 16))
        b = rng.integers(0, 2, size=(16, 16))
        d = metrics.dice(a, b)
        assert np.isclose(metrics.iou(a, b), d / (2 - d))


def test_identity_gives_perfect_scores():
    a = np.zeros((16, 16), dtype=np.uint16)
    a[4:9, 4:9] = 1
    a[11:14, 2:5] = 2
    assert metrics.dice(a, a) == 1.0
    assert metrics.iou(a, a) == 1.0
    assert metrics.hd95(a, a) == 0.0
    assert metrics.nsd(a, a, tau=1.0) == 1.0
    assert metrics.instance_f1(a, a, 0.5) == 1.0
    assert metrics.instance_f1(a, a, 0.7) == 1.0
    assert metrics.panoptic_quality(a, a) == 1.0


def test_symmetry():
    rng = np.random.default_rng(1)
    a = (rng.random((24, 24)) > 0.6).astype(np.uint8)
    b = (rng.random((24, 24)) > 0.6).astype(np.uint8)
    assert metrics.dice(a, b) == metrics.dice(b, a)
    assert metrics.iou(a, b) == metrics.iou(b, a)
    assert metrics.hd95(a, b) == metrics.hd95(b, a)
    assert metrics.nsd(a, b, tau=2.0) == metrics.nsd(b, a, tau=2.0)


# --------------------------------------------------------------------------- boundary


def test_hd95_and_nsd_hand_computed():
    """Two single pixels three apart: every pooled surface distance is exactly 3."""
    a = np.zeros((11, 11), dtype=np.uint8)
    b = np.zeros((11, 11), dtype=np.uint8)
    a[5, 5] = 1
    b[5, 8] = 1
    assert metrics.hd95(a, b) == 3.0
    assert metrics.nsd(a, b, tau=2.0) == 0.0
    assert metrics.nsd(a, b, tau=3.0) == 1.0


def test_one_pixel_shift():
    """The 1-px shift case. A square shifted by one column: every surface
    point is within 1 px of the other surface, so NSD at the primary tolerance is 1."""
    a = np.zeros((20, 20), dtype=np.uint8)
    b = np.zeros((20, 20), dtype=np.uint8)
    a[5:15, 5:15] = 1
    b[5:15, 6:16] = 1
    assert metrics.nsd(a, b, tau=metrics.TAU_PRIMARY_PX) == 1.0
    assert metrics.hd95(a, b) <= 2.0
    assert 0.8 < metrics.dice(a, b) < 1.0


def test_bbox_crop_is_exact_for_an_interior_structure():
    """The crop in surface_distances is an optimisation only: padding it out must not move a value."""
    small = np.zeros((40, 40), dtype=np.uint8)
    other = np.zeros((40, 40), dtype=np.uint8)
    small[18:22, 18:22] = 1
    other[19:24, 19:24] = 1

    big = np.zeros((400, 400), dtype=np.uint8)
    big_other = np.zeros((400, 400), dtype=np.uint8)
    big[18:22, 18:22] = 1
    big_other[19:24, 19:24] = 1

    assert metrics.hd95(small, other) == metrics.hd95(big, big_other)
    assert metrics.nsd(small, other, tau=2.0) == metrics.nsd(big, big_other, tau=2.0)


def test_bbox_crop_preserves_the_border_rule():
    """A structure running off the array edge has surface along that edge; the crop must keep that."""
    a = np.zeros((20, 20), dtype=np.uint8)
    b = np.zeros((20, 20), dtype=np.uint8)
    a[0:5, 0:5] = 1  # touches two array edges
    b[0:6, 0:6] = 1
    distances = metrics.surface_distances(a, b)
    surf_a = metrics.surface(a)
    # the corner at the array edge is surface under the border rule
    assert surf_a[0, 0]
    assert distances is not None and distances.size == surf_a.sum() + metrics.surface(b).sum()


def test_spacing_scales_distances():
    a = np.zeros((11, 11), dtype=np.uint8)
    b = np.zeros((11, 11), dtype=np.uint8)
    a[5, 5] = 1
    b[5, 8] = 1
    assert metrics.hd95(a, b, spacing=(1.0, 2.0)) == 6.0


# --------------------------------------------------------------------------- degenerate cases


def test_degenerate_cases_follow_empty_mask_policy():
    """The empty-mask policy exactly: both empty -> DSC = IoU = NSD = 1, HD95 = 0;
    exactly one empty -> DSC = IoU = NSD = 0, HD95 = NaN."""
    empty = np.zeros((8, 8), dtype=np.uint8)
    filled = np.zeros((8, 8), dtype=np.uint8)
    filled[2:5, 2:5] = 1

    assert metrics.dice(empty, empty) == 1.0
    assert metrics.iou(empty, empty) == 1.0
    assert metrics.nsd(empty, empty, tau=1.0) == 1.0
    assert metrics.hd95(empty, empty) == 0.0
    assert metrics.instance_f1(empty, empty) == 1.0
    assert metrics.panoptic_quality(empty, empty) == 1.0

    assert metrics.dice(empty, filled) == 0.0
    assert metrics.iou(empty, filled) == 0.0
    assert metrics.nsd(empty, filled, tau=1.0) == 0.0
    assert np.isnan(metrics.hd95(empty, filled))
    assert metrics.instance_f1(empty, filled) == 0.0
    assert metrics.panoptic_quality(empty, filled) == 0.0


def test_pairwise_scores_applies_degenerate_rules():
    empty = np.zeros((8, 8), dtype=np.uint16)
    filled = np.zeros((8, 8), dtype=np.uint16)
    filled[2:5, 2:5] = 1

    both = metrics.pairwise_scores(empty, empty, instance=True)
    assert both["dice"] == 1.0 and both["nsd@2"] == 1.0 and both["hd95"] == 0.0

    one = metrics.pairwise_scores(empty, filled, instance=True)
    assert one["dice"] == 0.0 and one["nsd@2"] == 0.0 and np.isnan(one["hd95"])


# --------------------------------------------------------------------------- instance


def test_instance_f1_and_pq_hand_computed():
    """Three instances each; two match exactly, one pair is disjoint. TP=2 FP=1 FN=1."""
    a, b = _three_instance_pair()
    assert metrics.instance_f1(a, b, 0.5) == 2 * 2 / (2 * 2 + 1 + 1)
    # PQ = sum of matched IoU / (TP + FP/2 + FN/2) = 2.0 / 3.0
    assert np.isclose(metrics.panoptic_quality(a, b), 2.0 / 3.0)


def test_instance_ids_need_not_be_contiguous():
    """CryoNuSeg label ids have gaps (89 instances, max id 157). Relabelling must not move scores."""
    a, b = _three_instance_pair()
    sparse_a = np.where(a == 1, 5, np.where(a == 2, 17, np.where(a == 3, 99, 0))).astype(np.uint16)
    sparse_b = np.where(b == 1, 4, np.where(b == 2, 60, np.where(b == 3, 255, 0))).astype(np.uint16)
    assert metrics.instance_f1(sparse_a, sparse_b, 0.5) == metrics.instance_f1(a, b, 0.5)
    assert metrics.panoptic_quality(sparse_a, sparse_b) == metrics.panoptic_quality(a, b)


def test_instance_f1_respects_threshold():
    """One pair at IoU 7/13 = 0.538: matched at the primary 0.5, not at the 0.7 sensitivity."""
    a = np.zeros((1, 20), dtype=np.uint16)
    b = np.zeros((1, 20), dtype=np.uint16)
    a[0, 0:10] = 1
    b[0, 3:13] = 1
    matrix, _, _ = metrics.instance_iou_matrix(a, b)
    assert np.isclose(matrix[0, 0], 7 / 13)
    assert metrics.instance_f1(a, b, metrics.INSTANCE_IOU_PRIMARY) == 1.0
    assert metrics.instance_f1(a, b, metrics.INSTANCE_IOU_SENSITIVITY) == 0.0


def test_instance_iou_matrix_shape_and_values():
    a = np.zeros((10, 10), dtype=np.uint16)
    b = np.zeros((10, 10), dtype=np.uint16)
    a[0:2, 0:2] = 7
    b[0:2, 0:2] = 1
    b[5:7, 5:7] = 2
    matrix, a_ids, b_ids = metrics.instance_iou_matrix(a, b)
    assert matrix.shape == (1, 2)
    assert list(a_ids) == [7] and list(b_ids) == [1, 2]
    assert matrix[0, 0] == 1.0 and matrix[0, 1] == 0.0


def test_known_instance_counts():
    """Known instance counts."""
    a = np.zeros((30, 30), dtype=np.uint16)
    for i in range(5):
        a[i * 6 : i * 6 + 4, 0:4] = i + 1
    matrix, a_ids, _ = metrics.instance_iou_matrix(a, a)
    assert len(a_ids) == 5
    assert np.allclose(np.diag(matrix), 1.0)


# --------------------------------------------------------------------------- reference-substitution spread


def test_reference_substitution_spread_hand_computed():
    """The reference-substitution spread with three raters and scores AB=0.9, AC=0.7, BC=0.8.
    Prediction A sees {0.9, 0.7} -> range 0.2; B sees {0.9, 0.8} -> 0.1; C sees {0.7, 0.8} -> 0.1.
    RSS = mean(0.2, 0.1, 0.1)."""
    scores = {
        ("A", "B"): 0.9, ("B", "A"): 0.9,
        ("A", "C"): 0.7, ("C", "A"): 0.7,
        ("B", "C"): 0.8, ("C", "B"): 0.8,
    }
    assert np.isclose(metrics.reference_substitution_spread(scores), (0.2 + 0.1 + 0.1) / 3)


def test_rss_undefined_for_two_raters():
    """RSS needs >= 3 raters, so 2-rater tasks have no spread and do not enter the leaderboard
    comparison."""
    scores = {("A", "B"): 0.9, ("B", "A"): 0.9}
    assert np.isnan(metrics.reference_substitution_spread(scores))


def test_rss_scores_for_image_is_symmetric_and_complete():
    masks = {}
    for i, name in enumerate(["A", "B", "C"]):
        m = np.zeros((12, 12), dtype=np.uint8)
        m[2 : 8 + i, 2:8] = 1
        masks[name] = m
    scores = metrics.rss_scores_for_image(masks, metrics.dice)
    assert len(scores) == 6  # 3 unordered pairs, both directions
    assert scores[("A", "B")] == scores[("B", "A")]
    assert np.isfinite(metrics.reference_substitution_spread(scores))


# --------------------------------------------------------------------------- tau95


def test_tau95_for_task_hand_computed():
    """Distances 0..99: the mean NSD reaches 0.95 at tau = 94 (values 0..94 are 95 of 100)."""
    assert np.isclose(metrics.tau95_for_task([np.arange(100.0)]), 94.0, atol=1e-2)


def test_tau95_unreachable_when_a_pair_is_one_empty():
    """A one-empty pair contributes NSD = 0 at every tau, capping the mean below 0.95."""
    assert np.isnan(metrics.tau95_for_task([np.arange(100.0), None], [False, False]))


# --------------------------------------------------------------------------- uncertainty


def test_bootstrap_ci_on_constants_is_degenerate():
    low, high = metrics.bootstrap_ci(np.full(50, 0.8))
    assert np.isclose(low, 0.8) and np.isclose(high, 0.8)


def test_bootstrap_ci_is_seeded_and_reproducible():
    values = np.linspace(0.5, 0.9, 40)
    assert metrics.bootstrap_ci(values) == metrics.bootstrap_ci(values)


def test_bootstrap_ci_does_not_depend_on_row_order():
    """The interval must be reproducible. Row order is an accident of how the
    per-image table was assembled - a fresh run and a checkpoint-resumed run ordered it differently -
    so the interval must depend only on the values and the seed."""
    values = np.array([28.2, 30.5, 38.2, 11.4, 19.7])
    shuffled = np.array([11.4, 28.2, 38.2, 19.7, 30.5])
    assert metrics.bootstrap_ci(values) == metrics.bootstrap_ci(shuffled)


def test_cluster_bootstrap_does_not_depend_on_row_order():
    values = np.array([0.1, 0.2, 0.8, 0.9, 0.5, 0.4])
    clusters = np.array(["a", "a", "b", "b", "c", "c"])
    order = [3, 0, 5, 2, 1, 4]
    assert metrics.bootstrap_ci(values, clusters=clusters) == metrics.bootstrap_ci(
        values[order], clusters=clusters[order]
    )


def test_bootstrap_ci_drops_nans():
    values = np.array([0.8, np.nan, 0.8, np.nan])
    assert metrics.bootstrap_ci(values) == (0.8, 0.8)


def test_cluster_bootstrap_uses_clusters():
    """Two clusters with disjoint values: resampling clusters must be able to reach either mean."""
    values = np.concatenate([np.zeros(10), np.ones(10)])
    clusters = np.array(["a"] * 10 + ["b"] * 10)
    low, high = metrics.bootstrap_ci(values, clusters=clusters)
    assert low == 0.0 and high == 1.0


def test_rater_bootstrap_ci():
    """The rater bootstrap, for sources with too few images to bootstrap over images."""
    per_pair = {("A", "B"): 0.9, ("A", "C"): 0.7, ("B", "C"): 0.8}
    low, high = metrics.rater_bootstrap_ci(per_pair, ["A", "B", "C"])
    assert 0.7 <= low <= high <= 0.9


# --------------------------------------------------------------------------- misc


def test_mismatched_shapes_raise():
    try:
        metrics.dice(np.zeros((4, 4)), np.zeros((4, 5)))
    except ValueError as exc:
        assert "shape" in str(exc)
    else:
        raise AssertionError("expected a ValueError on mismatched rater shapes")


def test_pairwise_scores_reports_every_metric():
    a = np.zeros((32, 32), dtype=np.uint16)
    b = np.zeros((32, 32), dtype=np.uint16)
    a[4:12, 4:12] = 1
    b[5:13, 5:13] = 1
    scores = metrics.pairwise_scores(a, b, instance=True)
    for key in (
        "dice", "iou", "hd95", "nsd@2", "nsd@1", "nsd@4",
        "instance_f1@0.5", "instance_f1@0.7", "panoptic_quality",
    ):
        assert key in scores, key
    assert 0.0 < scores["dice"] < 1.0


def test_pairwise_scores_uses_mm_grid_for_volumetric():
    a = np.zeros((16, 16), dtype=np.uint8)
    b = np.zeros((16, 16), dtype=np.uint8)
    a[4:10, 4:10] = 1
    b[5:11, 5:11] = 1
    scores = metrics.pairwise_scores(a, b, instance=False, spacing=(1.0, 1.0), volumetric=True)
    assert "nsd@3" in scores and "nsd@4" not in scores


def _three_instance_pair() -> tuple[np.ndarray, np.ndarray]:
    a = np.zeros((20, 20), dtype=np.uint16)
    b = np.zeros((20, 20), dtype=np.uint16)
    a[1:4, 1:4] = 1
    b[1:4, 1:4] = 1
    a[6:9, 6:9] = 2
    b[6:9, 6:9] = 2
    a[12:15, 12:15] = 3  # only in a
    b[16:19, 16:19] = 3  # only in b
    return a, b


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as exc:  # noqa: BLE001 - standalone runner reports and continues
                failures += 1
                print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
