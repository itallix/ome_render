from __future__ import annotations

import json
import re
from pathlib import PurePosixPath
from typing import Sequence

import numpy as np
import zarr

from ..models import ChannelInfo, LevelInfo, OmeImageInfo, Region
from ..storage import MediaStorage
from .regions import read_array_region


def _attributes(storage: MediaStorage) -> dict:
    path = storage.main_path
    if not path.is_dir():
        return {}
    for name in ("zarr.json", ".zattrs"):
        metadata = path / name
        if metadata.is_file():
            try:
                document = json.loads(metadata.read_text())
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid Zarr metadata in {metadata.name}") from exc
            if not isinstance(document, dict):
                raise ValueError(f"{metadata.name} must contain an object")
            attributes = document.get("attributes", {}) if name == "zarr.json" else document
            if not isinstance(attributes, dict):
                raise ValueError("Zarr attributes must be an object")
            return attributes
    return {}


def _ome_attributes(attributes: dict) -> dict:
    ome = attributes.get("ome", attributes)
    if not isinstance(ome, dict):
        raise ValueError("OME attributes must be an object")
    return ome


class _ZarrHandle:
    def __init__(self, group: zarr.Group, attributes: dict, series: int):
        self._group = group
        self._closed = False
        ome = _ome_attributes(attributes)
        multiscales = ome.get("multiscales")
        if not isinstance(multiscales, list) or not multiscales:
            raise ValueError("OME-Zarr requires a non-empty multiscales list")
        if not 0 <= series < len(multiscales):
            raise IndexError(f"series {series} out of range (0..{len(multiscales) - 1})")
        image = multiscales[series]
        if not isinstance(image, dict):
            raise ValueError("multiscales entries must be objects")
        version = ome.get("version", image.get("version"))
        if version not in ("0.4", "0.5"):
            raise ValueError(f"unsupported OME-Zarr version {version!r}; expected 0.4 or 0.5")
        axes = image.get("axes")
        if not isinstance(axes, list) or not all(
            isinstance(axis, dict) and isinstance(axis.get("name"), str) for axis in axes
        ):
            raise ValueError("OME-Zarr axes must be named axis objects")
        names = [axis["name"] for axis in axes]
        if len(set(names)) != len(names) or not {"x", "y"} <= set(names) or not set(names) <= set("tczyx"):
            raise ValueError(f"unsupported OME-Zarr axes {names!r}")
        self._axes = "".join(names).upper()
        datasets = image.get("datasets")
        if not isinstance(datasets, list) or not datasets:
            raise ValueError("OME-Zarr requires at least one pyramid dataset")
        self._arrays: list[zarr.Array] = []
        for dataset in datasets:
            path = dataset.get("path") if isinstance(dataset, dict) else None
            if not isinstance(path, str) or not path or path.startswith("/") or ".." in PurePosixPath(path).parts:
                raise ValueError(f"invalid OME-Zarr dataset path {path!r}")
            try:
                array = group[path]
            except KeyError as exc:
                raise ValueError(f"OME-Zarr dataset {path!r} is missing") from exc
            if not isinstance(array, zarr.Array) or array.ndim != len(names):
                raise ValueError(f"OME-Zarr dataset {path!r} does not match its axes")
            if any(size <= 0 for size in array.shape) or array.dtype.kind not in "buif":
                raise ValueError(f"unsupported OME-Zarr shape or pixel type in {path!r}")
            self._arrays.append(array)
        self.info = self._build_info(ome.get("omero", attributes.get("omero", {})))

    def _build_info(self, omero: dict) -> OmeImageInfo:
        first = self._arrays[0]
        dims = dict(zip(self._axes, first.shape))
        levels = []
        for i, array in enumerate(self._arrays):
            shape = dict(zip(self._axes, array.shape))
            if array.dtype != first.dtype or any(
                shape[axis] != dims[axis] for axis in self._axes if axis not in "XY"
            ):
                raise ValueError("OME-Zarr levels must share pixel type and T/Z/C dimensions")
            if shape["X"] > dims["X"] or shape["Y"] > dims["Y"]:
                raise ValueError("OME-Zarr level 0 must be full resolution")
            levels.append(LevelInfo(i, shape["X"], shape["Y"], (dims["X"] / shape["X"], dims["Y"] / shape["Y"])))
        if not isinstance(omero, dict):
            raise ValueError("OME-Zarr omero metadata must be an object")
        channels = omero.get("channels")
        size_c = dims.get("C", 1)
        if channels is not None and (not isinstance(channels, list) or len(channels) != size_c):
            raise ValueError("OME-Zarr channel metadata does not match the channel dimension")
        channel_info = []
        for i in range(size_c):
            channel = channels[i] if channels is not None else {}
            if not isinstance(channel, dict):
                raise ValueError("OME-Zarr channels must be objects")
            name, color = channel.get("label", f"Channel {i}"), channel.get("color")
            if not isinstance(name, str) or (color is not None and (
                not isinstance(color, str) or re.fullmatch(r"[0-9a-fA-F]{6}", color) is None
            )):
                raise ValueError(f"invalid OME-Zarr channel metadata at index {i}")
            channel_info.append(ChannelInfo(i, name, color.upper() if color else None))
        chunks = dict(zip(self._axes, first.chunks))
        return OmeImageInfo(
            size_x=dims["X"], size_y=dims["Y"], size_z=dims.get("Z", 1),
            size_c=size_c, size_t=dims.get("T", 1), pixel_type=np.dtype(first.dtype).name,
            levels=tuple(levels), tile_size=(chunks["X"], chunks["Y"]), channels=tuple(channel_info),
        )

    def read_region(
        self, *, level: int = 0, region: Region | None = None,
        z: int = 0, t: int = 0, channels: Sequence[int] | None = None,
    ) -> np.ndarray:
        if self._closed:
            raise ValueError("OME-Zarr handle is closed")
        if not 0 <= level < len(self._arrays):
            raise IndexError(f"level {level} out of range (0..{len(self._arrays) - 1})")
        return read_array_region(
            self._arrays[level], self._axes, self.info,
            level=level, region=region, z=z, t=t, channels=channels,
        )

    def close(self) -> None:
        if not self._closed:
            self._arrays.clear()
            self._group.store.close()
            self._closed = True

    def __enter__(self) -> "_ZarrHandle":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class OmeZarrReader:
    def probe(self, storage: MediaStorage) -> bool:
        return "multiscales" in _ome_attributes(_attributes(storage))

    def open(self, storage: MediaStorage, series: int = 0) -> _ZarrHandle:
        attributes = _attributes(storage)
        group = zarr.open_group(storage.main_path, mode="r", use_consolidated=False)
        try:
            return _ZarrHandle(group, attributes, series)
        except Exception:
            group.store.close()
            raise
