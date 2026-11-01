"""`load_mask` handles float label images block by block, for memory (gtu/io.py). That changes no
metric definition, so it must change no value either: this pins the blocked code against the
exact whole-array expression it replaces.

Runs under pytest or standalone: python code/tests/test_io_load_mask_equivalence.py"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gtu import io


def _legacy(array: np.ndarray) -> np.ndarray:
    """The implementation as it stood before the blocking change, verbatim."""
    if np.issubdtype(array.dtype, np.floating):
        rounded = np.rint(array)
        finite = np.isfinite(array)
        if finite.any() and np.all(np.abs(array[finite] - rounded[finite]) < 1e-6):
            array = rounded.astype(np.int32)
    return array


def _new(array: np.ndarray) -> np.ndarray:
    if np.issubdtype(array.dtype, np.floating):
        if io._every_finite_value_is_integral(array):
            array = io._round_to_int32(array)
    return array


def _same(name: str, array: np.ndarray) -> None:
    with np.errstate(invalid="ignore"):
        expected, actual = _legacy(array.copy()), _new(array.copy())
    assert expected.dtype == actual.dtype, f"{name}: dtype {expected.dtype} vs {actual.dtype}"
    assert expected.shape == actual.shape, f"{name}: shape {expected.shape} vs {actual.shape}"
    both_nan = np.isnan(expected) & np.isnan(actual) if expected.dtype.kind == "f" else False
    assert np.all((expected == actual) | both_nan), f"{name}: values differ"
    print(f"PASS  {name:46s} -> {actual.dtype}")


CASES = {
    # the real reason the tolerance exists: CURVAS's scaled int16 labels
    "curvas_style_0.9999999997": np.array([[0.0, 0.9999999997], [2.0000000001, 3.0]]),
    "clean_integral_floats": np.arange(24, dtype=np.float64).reshape(2, 3, 4),
    "non_integral_stays_float": np.array([[0.0, 0.5], [1.0, 2.0]]),
    "just_outside_tolerance": np.array([[0.0, 1.0 + 2e-6]]),
    "just_inside_tolerance": np.array([[0.0, 1.0 + 5e-7]]),
    "nan_mixed_with_integers": np.array([[0.0, np.nan], [1.0, 2.0]]),
    "all_nan": np.full((2, 3), np.nan),
    "inf_mixed": np.array([[0.0, np.inf], [1.0, -np.inf]]),
    "empty": np.zeros((0, 4)),
    "single_voxel": np.array([[7.0]]),
    "zero_dimensional": np.array(3.0),
    "negative_labels": np.array([[-3.0, -1.0], [0.0, 5.0]]),
    "already_int": np.arange(12, dtype=np.int32).reshape(3, 4),
    "float32_integral": np.arange(12, dtype=np.float32).reshape(3, 4),
    "large_values": np.array([[2.0**30, -(2.0**30)]]),
}


def test_matches_legacy_on_edge_cases() -> None:
    for name, array in CASES.items():
        _same(name, array)


def test_matches_legacy_when_blocking_actually_splits() -> None:
    """Force several blocks so a bug that only shows across block boundaries cannot hide."""
    original = io._BLOCK_BYTES
    try:
        io._BLOCK_BYTES = 64  # a handful of voxels per block
        rng = np.random.default_rng(20260908)
        _same("blocked_integral", rng.integers(0, 5, size=(40, 8)).astype(np.float64))
        mixed = rng.integers(0, 5, size=(40, 8)).astype(np.float64)
        mixed[37, 5] = 1.25  # non-integral in the LAST block only
        _same("blocked_non_integral_late", mixed)
        early = rng.integers(0, 5, size=(40, 8)).astype(np.float64)
        early[0, 0] = 0.5  # non-integral in the FIRST block; must bail and still agree
        _same("blocked_non_integral_early", early)
        nanned = rng.integers(0, 5, size=(40, 8)).astype(np.float64)
        nanned[:20] = np.nan  # whole leading blocks non-finite, so any_finite must accumulate
        _same("blocked_leading_all_nan", nanned)
    finally:
        io._BLOCK_BYTES = original


def test_matches_legacy_on_real_volumes() -> None:
    """The cases that actually motivated the change, if the data is on disk."""
    root = Path(__file__).resolve().parents[2] / "data/gtu-raw"
    candidates = [
        ("curvas", next(root.glob("curvas/*/*/annotation_1.nii.gz"), None)),
        ("pdacvi", next(root.glob("pdacvi/*/*/annotation_1.nii.gz"), None)),
    ]
    import nibabel

    for name, path in candidates:
        if path is None:
            print(f"SKIP  {name}: not on disk")
            continue
        raw = np.asanyarray(nibabel.load(path).dataobj)
        _same(f"real_{name}_{raw.dtype}", raw)


if __name__ == "__main__":
    failures = 0
    for test in [
        test_matches_legacy_on_edge_cases,
        test_matches_legacy_when_blocking_actually_splits,
        test_matches_legacy_on_real_volumes,
    ]:
        try:
            test()
        except AssertionError as error:
            failures += 1
            print(f"FAIL  {test.__name__}: {error}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
