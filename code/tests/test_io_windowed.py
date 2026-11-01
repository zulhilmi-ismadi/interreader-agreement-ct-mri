"""WindowedMask must be a pure storage optimisation: every metric has to give the same value on a
windowed pair as on the same pair at full size. Runs standalone or under pytest."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gtu import metrics
from gtu.io import WindowedMask, align


def _assert_same_scores(a: np.ndarray, b: np.ndarray, instance: bool = False) -> None:
    windowed = align(WindowedMask.from_array(a), WindowedMask.from_array(b))
    direct = metrics.pairwise_scores(a, b, instance=instance)
    through_window = metrics.pairwise_scores(*windowed, instance=instance)
    for key, value in direct.items():
        other = through_window[key]
        assert (np.isnan(value) and np.isnan(other)) or value == other, (
            f"{key}: full={value} windowed={other}"
        )


def test_round_trip_preserves_values():
    array = np.zeros((40, 40), dtype=np.uint16)
    array[10:14, 10:14] = 7
    array[20:22, 20:22] = 99
    assert np.array_equal(WindowedMask.from_array(array).to_array(), array)


def test_window_is_much_smaller_than_the_array():
    array = np.zeros((400, 400), dtype=np.uint8)
    array[100:110, 100:110] = 1
    windowed = WindowedMask.from_array(array)
    assert windowed.data.size < array.size / 100


def test_scores_match_for_interior_structures():
    a = np.zeros((200, 200), dtype=np.uint8)
    b = np.zeros((200, 200), dtype=np.uint8)
    a[80:100, 80:100] = 1
    b[85:108, 84:101] = 1
    _assert_same_scores(a, b)


def test_scores_match_for_disjoint_structures():
    """Far-apart masks give a large joint window; values must still agree."""
    a = np.zeros((200, 200), dtype=np.uint8)
    b = np.zeros((200, 200), dtype=np.uint8)
    a[10:20, 10:20] = 1
    b[170:190, 170:190] = 1
    _assert_same_scores(a, b)


def test_scores_match_when_structures_touch_the_array_border():
    """The border rule is the case most likely to break under windowing."""
    a = np.zeros((60, 60), dtype=np.uint8)
    b = np.zeros((60, 60), dtype=np.uint8)
    a[0:20, 0:20] = 1
    b[0:24, 0:18] = 1
    _assert_same_scores(a, b)


def test_scores_match_for_instance_masks():
    a = np.zeros((80, 80), dtype=np.uint16)
    b = np.zeros((80, 80), dtype=np.uint16)
    for i in range(4):
        a[10 + i * 12 : 16 + i * 12, 10:16] = i + 1
        b[11 + i * 12 : 17 + i * 12, 10:16] = (i + 1) * 5
    _assert_same_scores(a, b, instance=True)


def test_scores_match_with_one_empty_mask():
    a = np.zeros((50, 50), dtype=np.uint8)
    b = np.zeros((50, 50), dtype=np.uint8)
    a[10:20, 10:20] = 1
    _assert_same_scores(a, b)


def test_scores_match_with_both_empty():
    a = np.zeros((50, 50), dtype=np.uint8)
    b = np.zeros((50, 50), dtype=np.uint8)
    _assert_same_scores(a, b)


def test_align_rejects_mismatched_shapes():
    try:
        align(
            WindowedMask.from_array(np.ones((4, 4), dtype=np.uint8)),
            WindowedMask.from_array(np.ones((4, 5), dtype=np.uint8)),
        )
    except ValueError as exc:
        assert "shape" in str(exc)
    else:
        raise AssertionError("expected a ValueError on mismatched shapes")


def test_scores_match_on_a_3d_volume():
    a = np.zeros((60, 60, 30), dtype=np.uint8)
    b = np.zeros((60, 60, 30), dtype=np.uint8)
    a[20:30, 20:30, 10:16] = 1
    b[22:33, 21:29, 11:18] = 1
    windowed = align(WindowedMask.from_array(a), WindowedMask.from_array(b))
    spacing = (1.0, 1.0, 2.5)
    assert metrics.hd95(a, b, spacing) == metrics.hd95(*windowed, spacing)
    assert metrics.nsd(a, b, 2.0, spacing) == metrics.nsd(*windowed, 2.0, spacing)


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
