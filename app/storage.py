from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MediaStorage:
    """Where a media's bytes live. `entrypoint` is relative to `root`; None for single-file media."""

    root: Path
    entrypoint: Path | None = None

    @property
    def is_directory(self) -> bool:
        return self.entrypoint is not None

    @property
    def main_path(self) -> Path:
        """The file/dir a reader should open first."""
        return self.root if self.entrypoint is None else self.root / self.entrypoint
