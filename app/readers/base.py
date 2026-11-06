from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from ..models import OmeImageInfo, Region
from ..storage import MediaStorage


class OmeImageHandle(Protocol):
    """An open image. Parsed once at open; reads are cheap."""

    info: OmeImageInfo

    def read_region(
        self,
        *,
        level: int = 0,
        region: Region | None = None,  # None = full plane
        z: int = 0,
        t: int = 0,
        channels: Sequence[int] | None = None,  # None = all
    ) -> np.ndarray:
        """Return (C, H, W) in the order of `channels`, whatever the on-disk axis order.
        Absent Z/T axes only accept 0. Raises IndexError for out-of-range coordinates."""
        ...

    def close(self) -> None: ...
    def __enter__(self) -> "OmeImageHandle": ...
    def __exit__(self, *exc) -> None: ...


class OmeReader(Protocol):
    """One implementation per container format."""

    def probe(self, storage: MediaStorage) -> bool:
        """Cheap content sniff (don't trust extensions)."""
        ...

    def open(self, storage: MediaStorage, series: int = 0) -> OmeImageHandle: ...
