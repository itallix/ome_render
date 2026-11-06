from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Sequence

import numpy as np
import tifffile
import zarr

from ..models import ChannelInfo, LevelInfo, OmeImageInfo, Region
from ..storage import MediaStorage
from .regions import read_array_region

_SUPPORTED_AXES = set("TZCSYX")


class _TiffHandle:
    def __init__(self, tif: tifffile.TiffFile, series_index: int):
        if not 0 <= series_index < len(tif.series):
            raise IndexError(f"series {series_index} out of range (0..{len(tif.series) - 1})")
        self._tif = tif
        self._series = tif.series[series_index]
        self._series_index = series_index
        self._arrays: dict[int, zarr.Array] = {}

        axes = self._series.axes
        if not set(axes) <= _SUPPORTED_AXES:
            raise ValueError(f"unsupported axes {axes!r}")
        if "C" in axes and "S" in axes:
            raise NotImplementedError("both C and S axes present")
        self._axes = axes
        self._chan_axis = "C" if "C" in axes else ("S" if "S" in axes else None)
        self.info = self._build_info()

    # ---- metadata -------------------------------------------------------
    def _build_info(self) -> OmeImageInfo:
        s = self._series
        dims = dict(zip(self._axes, s.shape))
        w0, h0 = dims["X"], dims["Y"]
        levels = []
        for i, lv in enumerate(s.levels):
            d = dict(zip(lv.axes, lv.shape))
            levels.append(LevelInfo(i, d["X"], d["Y"], (w0 / d["X"], h0 / d["Y"])))
        kf = s.keyframe
        tile = (kf.tilewidth, kf.tilelength) if getattr(kf, "is_tiled", False) else None
        size_c = dims.get(self._chan_axis, 1) if self._chan_axis else 1
        return OmeImageInfo(
            size_x=w0, size_y=h0,
            size_z=dims.get("Z", 1), size_c=size_c, size_t=dims.get("T", 1),
            pixel_type=np.dtype(s.dtype).name,
            levels=tuple(levels), tile_size=tile,
            channels=self._parse_channels(size_c),
        )

    def _parse_channels(self, size_c: int) -> tuple[ChannelInfo, ...]:
        names: list[tuple[str | None, str | None]] = []
        xml = self._tif.ome_metadata
        if xml:
            try:
                images = ET.fromstring(xml).findall("{*}Image")
                for ch in images[self._series_index].findall("{*}Pixels/{*}Channel"):
                    color = ch.get("Color")
                    hex_color = f"{(int(color) & 0xFFFFFFFF) >> 8:06X}" if color else None
                    names.append((ch.get("Name"), hex_color))
            except (ET.ParseError, IndexError, ValueError):
                names = []
        if len(names) != size_c:  # missing, or RGB-style samples
            names = [(None, None)] * size_c
        return tuple(ChannelInfo(i, n or f"Channel {i}", c) for i, (n, c) in enumerate(names))

    # ---- pixels ---------------------------------------------------------
    def _array(self, level: int) -> zarr.Array:
        if level not in self._arrays:
            store = self._series.aszarr(level=level)  # lazy, tile/chunk-wise reads
            self._arrays[level] = zarr.open(store, mode="r")
        return self._arrays[level]

    def read_region(
        self,
        *,
        level: int = 0,
        region: Region | None = None,
        z: int = 0,
        t: int = 0,
        channels: Sequence[int] | None = None,
    ) -> np.ndarray:
        info = self.info
        if not 0 <= level < len(info.levels):
            raise IndexError(f"level {level} out of range (0..{len(info.levels) - 1})")
        return read_array_region(
            self._array(level), self._axes, info,
            level=level, region=region, z=z, t=t, channels=channels,
        )

    # ---- lifecycle ------------------------------------------------------
    def close(self) -> None:
        self._arrays.clear()
        self._tif.close()

    def __enter__(self) -> "_TiffHandle":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class OmeTiffReader:
    def probe(self, storage: MediaStorage) -> bool:
        path = storage.main_path
        if not path.is_file():
            return False
        try:
            with tifffile.TiffFile(path) as tif:
                return bool(tif.is_ome)
        except Exception:
            return False

    def open(self, storage: MediaStorage, series: int = 0) -> _TiffHandle:
        tif = tifffile.TiffFile(storage.main_path)
        try:
            return _TiffHandle(tif, series)
        except Exception:
            tif.close()
            raise
