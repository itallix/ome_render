from __future__ import annotations

import os
import re
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Response

from .models import Region
from .readers.registry import open_image
from .render import composite, encode, parse_channels
from .storage import MediaStorage

MEDIA_ROOT = Path(os.environ.get("MEDIA_ROOT", "media")).resolve()
MAX_PIXELS = 4096 * 4096

app = FastAPI(title="ome-render")


def resolve_storage(media: str) -> MediaStorage:
    """Map a media id to storage. Swap for a DB lookup (media.id -> ome_images row) later."""
    if not re.fullmatch(r"[\w.\-]+", media):
        raise HTTPException(400, "invalid media id")
    path = MEDIA_ROOT / media
    if path.is_file():
        return MediaStorage(root=path)
    if path.is_dir():
        if (path / ".zattrs").is_file() or (path / "zarr.json").is_file():
            return MediaStorage(root=path, entrypoint=Path("."))
        for pattern in ("*.companion.ome", "*.ome.tif", "*.ome.tiff"):
            hit = next(iter(sorted(path.glob(pattern))), None)
            if hit:
                return MediaStorage(root=path, entrypoint=Path(hit.name))
    raise HTTPException(404, f"media {media!r} not found")


def _open(media: str, series: int):
    storage = resolve_storage(media)
    try:
        return open_image(storage, series)
    except IndexError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(415, str(e))


@app.get("/info")
def info(media: str, series: int = 0):
    with _open(media, series) as img:
        return asdict(img.info)


@app.get("/render")
def render(
    media: str,
    c: str = Query(..., description="idx|start:end$RRGGBB[,...]"),
    z: int = 0,
    t: int = 0,
    level: int = 0,
    series: int = 0,
    x: int | None = None,
    y: int | None = None,
    w: int | None = None,
    h: int | None = None,
    fmt: str = Query("png", pattern="^(png|jpeg)$"),
):
    try:
        specs = parse_channels(c)
    except ValueError as e:
        raise HTTPException(400, str(e))

    region = None
    if None not in (x, y, w, h):
        region = Region(x, y, w, h)  # type: ignore[arg-type]
    elif any(v is not None for v in (x, y, w, h)):
        raise HTTPException(400, "x, y, w, h must be given together")

    order = sorted({s.index for s in specs})
    with _open(media, series) as img:
        try:
            lv = img.info.levels[level] if 0 <= level < len(img.info.levels) else None
            if lv is not None:
                rw, rh = (region.width, region.height) if region else (lv.width, lv.height)
                if rw * rh > MAX_PIXELS:
                    raise HTTPException(413, f"{rw}x{rh} exceeds {MAX_PIXELS} pixels; use a higher level or a region")
            planes = img.read_region(level=level, region=region, z=z, t=t, channels=order)
        except IndexError as e:
            raise HTTPException(400, str(e))

    body, mime = encode(composite(planes, order, specs), fmt)
    return Response(body, media_type=mime, headers={"Cache-Control": "public, max-age=3600"})
