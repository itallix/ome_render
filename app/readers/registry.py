from __future__ import annotations

from ..storage import MediaStorage
from .base import OmeImageHandle, OmeReader
from .ome_tiff import OmeTiffReader
from .ome_zarr import OmeZarrReader

READERS: list[OmeReader] = [OmeTiffReader(), OmeZarrReader()]


def open_image(storage: MediaStorage, series: int = 0) -> OmeImageHandle:
    for reader in READERS:
        if reader.probe(storage):
            return reader.open(storage, series)
    raise ValueError(f"no reader can handle {storage.main_path}")
