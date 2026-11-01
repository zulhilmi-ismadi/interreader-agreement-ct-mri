"""Mask loading. The canonical schema stores `mask_path`, so everything that needs an
array goes through here — one place to handle the per-source dtype quirks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def load_mask(path: Path | str) -> np.ndarray:
    """Read a mask at native resolution, never resampled.

    Float label images are cast to integer when every value is integral: Mesmer ships some annotators'
    label images as float32/float64, and comparing float label ids risks silent instance miscounts."""
    path = Path(path)
    suffix = "".join(path.suffixes).lower()

    if suffix.endswith((".tif", ".tiff")):
        import tifffile

        array = tifffile.imread(path)
    elif suffix.endswith(".png"):
        # Gleason 2019 ships its class maps as 8-bit single-channel PNG. skimage is already a
        # dependency; it returns the label plane unchanged, no palette expansion.
        from skimage import io as skimage_io

        array = skimage_io.imread(path)
    elif suffix.endswith((".nii", ".nii.gz")):
        import nibabel

        array = np.asanyarray(nibabel.load(path).dataobj)
    else:
        raise ValueError(f"unsupported mask format: {path}")

    if array.ndim > 2 and array.shape[-1] in (3, 4):
        raise ValueError(
            f"{path.name} looks like an RGB image, not a label mask "
            f"(shape {array.shape}); exclude it in the loader"
        )

    if np.issubdtype(array.dtype, np.floating):
        # Tolerant, not exact: CURVAS stores labels as int16 spanning the full dtype range with a
        # nan scl_slope, so nibabel hands back 0.9999999997 rather than 1. Comparing such a label
        # with == would silently match nothing. 1e-6 is far below any real label spacing.
        if _every_finite_value_is_integral(array):
            array = _round_to_int32(array)
    return array


#: Bytes of `array` covered by one block in the two helpers below. Big enough that per-block
#: overhead is noise, small enough that a block's temporaries are megabytes rather than gigabytes.
_BLOCK_BYTES = 64 << 20

#: Half a rounding unit would be 0.5; this is far below any real label spacing and well above the
#: float64 error in CURVAS's scaled int16 labels.
_INTEGRAL_TOLERANCE = 1e-6


def _blocks(array: np.ndarray):
    """Slices along the leading axis, each spanning about `_BLOCK_BYTES` of `array`.

    These two helpers exist for memory, not speed. Written as whole-array expressions - `rounded =
    np.rint(array)`, then `np.abs(array[finite] - rounded[finite])` - the integrality test holds the
    float64 original, a float64 copy, a bool mask and three more full-size temporaries at once, about
    40 bytes per voxel. On a 512 x 512 x 953 CURVAS volume that is ~10 GB for a test whose answer is
    one bit, and it was what capped the volumetric worker count at three. Blocking keeps the peak at
    the original plus the int32 result."""
    if array.ndim == 0 or array.size == 0:
        yield Ellipsis
        return
    per_row = max(array[0].nbytes, 1)
    rows = max(_BLOCK_BYTES // per_row, 1)
    for start in range(0, array.shape[0], rows):
        yield slice(start, min(start + rows, array.shape[0]))


def _every_finite_value_is_integral(array: np.ndarray) -> bool:
    """True when at least one value is finite and every finite value is a whole number.

    Exactly the condition the whole-array form tested, evaluated block by block: `np.rint` is
    elementwise, so rounding a block equals the matching slice of rounding everything, and `all` over
    the blocks equals `all` over the array. Bails on the first non-integral block."""
    any_finite = False
    for block in _blocks(array):
        chunk = array[block]
        finite = np.isfinite(chunk)
        if not finite.any():
            continue
        any_finite = True
        values = chunk[finite]
        if not np.all(np.abs(values - np.rint(values)) < _INTEGRAL_TOLERANCE):
            return False
    return any_finite


def _round_to_int32(array: np.ndarray) -> np.ndarray:
    """`np.rint(array).astype(np.int32)` without materialising the rounded float array.

    A non-finite value casts to whatever the platform gives, exactly as `.astype` did; this function
    is only reached when every finite value is integral, and it preserves that edge case rather than
    quietly repairing it."""
    out = np.empty(array.shape, dtype=np.int32)
    for block in _blocks(array):
        out[block] = np.rint(array[block])
    return out


@dataclass(frozen=True)
class WindowedMask:
    """A mask stored only over the box that contains it.

    C3RO needs this: an ROI can occupy 0.1% of a 512x512x140 volume, so holding one rater panel as
    full arrays costs 4.4 GB where the windows cost 36 MB. Values are preserved, not binarised, so
    instance label images survive the round trip.

    The window is padded by one voxel and clipped to the array, exactly as `metrics._union_window`
    is, so a pair materialised by `align` scores identically to the same pair at full size — including
    the border rule, since a structure reaching the array edge still reaches its window's edge."""

    shape: tuple[int, ...]
    window: tuple[slice, ...]
    data: np.ndarray

    @classmethod
    def from_array(cls, array: np.ndarray, pad: int = 1) -> WindowedMask:
        array = np.asarray(array)
        coordinates = np.argwhere(array != 0)
        if coordinates.size == 0:
            empty = tuple(slice(0, 0) for _ in array.shape)
            return cls(array.shape, empty, array[empty])
        low = np.maximum(coordinates.min(axis=0) - pad, 0)
        high = np.minimum(coordinates.max(axis=0) + pad + 1, array.shape)
        window = tuple(slice(int(lo), int(hi)) for lo, hi in zip(low, high))
        return cls(array.shape, window, array[window].copy())

    @property
    def is_empty(self) -> bool:
        return self.data.size == 0

    def to_array(self) -> np.ndarray:
        """Rebuild the full-size array. Only for tests and single-mask inspection."""
        full = np.zeros(self.shape, dtype=self.data.dtype)
        if not self.is_empty:
            full[self.window] = self.data
        return full


def align(first: WindowedMask, second: WindowedMask) -> tuple[np.ndarray, np.ndarray]:
    """Materialise two masks into their smallest common window, ready to hand to the metrics."""
    if first.shape != second.shape:
        raise ValueError(f"raters disagree on image shape: {first.shape} vs {second.shape}")

    if first.is_empty and second.is_empty:
        zero = np.zeros((0,) * len(first.shape), dtype=first.data.dtype)
        return zero, zero.copy()

    present = [m for m in (first, second) if not m.is_empty]
    low = [min(m.window[axis].start for m in present) for axis in range(len(first.shape))]
    high = [max(m.window[axis].stop for m in present) for axis in range(len(first.shape))]
    joint = tuple(slice(lo, hi) for lo, hi in zip(low, high))
    size = tuple(hi - lo for lo, hi in zip(low, high))

    materialised = []
    for mask in (first, second):
        buffer = np.zeros(size, dtype=mask.data.dtype)
        if not mask.is_empty:
            offset = tuple(
                slice(mask.window[axis].start - joint[axis].start,
                      mask.window[axis].stop - joint[axis].start)
                for axis in range(len(size))
            )
            buffer[offset] = mask.data
        materialised.append(buffer)
    return materialised[0], materialised[1]
