from __future__ import annotations

import io
import re
from dataclasses import dataclass

import numpy as np
from PIL import Image

# "<channel>|<start>:<end>$<RRGGBB>", comma separated
_SPEC = re.compile(r"^(\d+)\|(-?\d+(?:\.\d+)?):(-?\d+(?:\.\d+)?)\$([0-9a-fA-F]{6})$")


@dataclass(frozen=True)
class ChannelSpec:
    index: int
    start: float
    end: float
    rgb: tuple[int, int, int]


def parse_channels(raw: str) -> list[ChannelSpec]:
    specs = []
    for part in raw.split(","):
        m = _SPEC.match(part.strip())
        if not m:
            raise ValueError(f"bad channel spec {part!r}; expected idx|start:end$RRGGBB")
        idx, start, end, color = int(m[1]), float(m[2]), float(m[3]), m[4]
        if end <= start:
            raise ValueError(f"channel {idx}: end must be greater than start")
        rgb = tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))
        specs.append(ChannelSpec(idx, start, end, rgb))  # type: ignore[arg-type]
    if not specs:
        raise ValueError("no channels requested")
    return specs


def composite(planes: np.ndarray, order: list[int], specs: list[ChannelSpec]) -> np.ndarray:
    """planes: (C,H,W) for channel indices `order`. Additive blend -> (H,W,3) uint8."""
    pos = {c: i for i, c in enumerate(order)}
    out = np.zeros(planes.shape[1:] + (3,), dtype=np.float32)
    for s in specs:
        x = planes[pos[s.index]].astype(np.float32)
        x = np.clip((x - s.start) / (s.end - s.start), 0.0, 1.0)  # window: start->black, end->full
        out += x[..., None] * (np.array(s.rgb, dtype=np.float32) / 255.0)
    return (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def encode(rgb: np.ndarray, fmt: str) -> tuple[bytes, str]:
    buf = io.BytesIO()
    if fmt == "jpeg":
        Image.fromarray(rgb).save(buf, "JPEG", quality=90)
        return buf.getvalue(), "image/jpeg"
    Image.fromarray(rgb).save(buf, "PNG")
    return buf.getvalue(), "image/png"
