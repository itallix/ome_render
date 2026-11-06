from __future__ import annotations

from typing import Sequence

import numpy as np
import zarr

from ..models import OmeImageInfo, Region


def read_array_region(
    array: zarr.Array,
    axes: str,
    info: OmeImageInfo,
    *,
    level: int,
    region: Region | None,
    z: int,
    t: int,
    channels: Sequence[int] | None,
) -> np.ndarray:
    if not 0 <= z < info.size_z:
        raise IndexError(f"z {z} out of range (0..{info.size_z - 1})")
    if not 0 <= t < info.size_t:
        raise IndexError(f"t {t} out of range (0..{info.size_t - 1})")
    chans = list(range(info.size_c)) if channels is None else list(channels)
    if not chans:
        raise IndexError("at least one channel is required")
    for c in chans:
        if not 0 <= c < info.size_c:
            raise IndexError(f"channel {c} out of range (0..{info.size_c - 1})")

    lv = info.levels[level]
    r = region or Region(0, 0, lv.width, lv.height)
    if not (0 <= r.x < lv.width and 0 <= r.y < lv.height and r.width > 0 and r.height > 0):
        raise IndexError(f"region outside level {level} ({lv.width}x{lv.height})")
    x1, y1 = min(r.x + r.width, lv.width), min(r.y + r.height, lv.height)

    planes = []
    for c in chans:
        coordinates = {
            "T": t, "Z": z, "C": c, "S": c,
            "Y": slice(r.y, y1), "X": slice(r.x, x1),
        }
        plane = np.asarray(array[tuple(coordinates[axis] for axis in axes)])
        if axes.index("X") < axes.index("Y"):
            plane = plane.T
        planes.append(plane)
    return np.stack(planes)
