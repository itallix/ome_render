from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Region:
    """Pixel box at a given pyramid level (x/y = top-left)."""

    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class LevelInfo:
    level: int
    width: int
    height: int
    scale: tuple[float, float]  # (x, y) relative to level 0


@dataclass(frozen=True)
class ChannelInfo:
    index: int
    name: str
    color: str | None  # "RRGGBB" or None


@dataclass(frozen=True)
class OmeImageInfo:
    size_x: int
    size_y: int
    size_z: int
    size_c: int
    size_t: int
    pixel_type: str  # numpy dtype name for now; normalise to OME names at ingest
    levels: tuple[LevelInfo, ...]  # >= 1; non-pyramidal = single level
    tile_size: tuple[int, int] | None  # native (width, height) hint
    channels: tuple[ChannelInfo, ...]
