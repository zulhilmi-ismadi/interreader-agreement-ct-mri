"""Hand-computable validation of View B (leave-one-rater-out against a consensus of the rest).

Runs under pytest or standalone: python code/tests/test_consensus_toy.py"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gtu import consensus


def test_majority_vote_hand_computed():
    """Three raters: a pixel carried by 2 is foreground, a pixel carried by 1 is not."""
    a = np.zeros((4, 4), dtype=np.uint8)
    b = np.zeros((4, 4), dtype=np.uint8)
    c = np.zeros((4, 4), dtype=np.uint8)
    a[0, 0] = b[0, 0] = 1  # 2 of 3
    a[1, 1] = 1  # 1 of 3
    a[2, 2] = b[2, 2] = c[2, 2] = 1  # 3 of 3

    voted = consensus.majority_vote([a, b, c])
    assert voted[0, 0]
    assert not voted[1, 1]
    assert voted[2, 2]
    assert voted.sum() == 2


def test_majority_vote_ties_go_to_foreground():
    """Majority vote resolves ties to foreground, so an even split is foreground."""
    a = np.zeros((4, 4), dtype=np.uint8)
    b = np.zeros((4, 4), dtype=np.uint8)
    c = np.zeros((4, 4), dtype=np.uint8)
    d = np.zeros((4, 4), dtype=np.uint8)
    a[0, 0] = b[0, 0] = 1  # 2 of 4 -> tie -> foreground
    a[1, 1] = 1  # 1 of 4 -> background

    voted = consensus.majority_vote([a, b, c, d])
    assert voted[0, 0]
    assert not voted[1, 1]


def test_majority_vote_of_identical_masks_is_that_mask():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:5, 2:5] = 1
    assert np.array_equal(consensus.majority_vote([mask, mask, mask]), mask > 0)


def test_staple_of_identical_masks_is_that_mask():
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:10, 4:10] = 1
    result = consensus.staple_consensus([mask, mask, mask])
    assert result.dtype == bool
    assert np.array_equal(result, mask > 0)


def test_staple_recovers_the_shape_two_of_three_raters_agree_on():
    base = np.zeros((24, 24), dtype=np.uint8)
    base[6:18, 6:18] = 1
    agree_a, agree_b = base.copy(), base.copy()
    outlier = np.zeros((24, 24), dtype=np.uint8)
    outlier[10:14, 10:14] = 1  # much smaller

    result = consensus.staple_consensus([agree_a, agree_b, outlier])
    overlap = np.logical_and(result, base > 0).sum() / (base > 0).sum()
    assert overlap > 0.9


def test_staple_needs_two_raters():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:4, 2:4] = 1
    try:
        consensus.staple_consensus([mask])
    except ValueError as exc:
        assert "two raters" in str(exc)
    else:
        raise AssertionError("expected a ValueError for a single-rater consensus")


def test_view_b_undefined_below_three_raters():
    """View B applies only when >= 3 raters annotated the image."""
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:5, 2:5] = 1
    assert consensus.view_b_scores({"r1": mask, "r2": mask}) == {}


def test_view_b_reports_both_consensus_methods():
    masks = {}
    for i, name in enumerate(["r1", "r2", "r3", "r4"]):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8 : 20 + i, 8:20] = 1
        masks[name] = mask

    scores = consensus.view_b_scores(masks)
    assert set(scores) == {"staple", "majority"}
    for method in scores.values():
        for key in ("dice", "iou", "hd95", "nsd@2"):
            assert key in method, key
        assert 0.0 < method["dice"] <= 1.0
    # consensus is binary, so no instance metrics in View B
    assert not any(k.startswith("instance_f1") for k in scores["staple"])


def test_view_b_on_identical_raters_is_perfect():
    mask = np.zeros((16, 16), dtype=np.uint16)
    mask[4:12, 4:12] = 1
    scores = consensus.view_b_scores({name: mask for name in ("r1", "r2", "r3")})
    for method in scores.values():
        assert method["dice"] == 1.0
        assert method["hd95"] == 0.0


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
