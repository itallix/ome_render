# ome-render

Small FastAPI backend that renders OME-TIFF and local OME-Zarr images as composited PNG/JPEG.
Both formats use `probe` / `open -> handle.read_region` with lazy pixel-region reads.

```
uv sync                                              # creates .venv from uv.lock (incl. dev group)
uv run python -m tests.sample media/sample.ome.tif   # synthetic Z=16, C=3, 3-level pyramid
MEDIA_ROOT=media uv run uvicorn app.main:app --reload
uv run pytest
```

```
GET /render?media=sample.ome.tif&z=12&t=0&level=2&c=0|120:2400$3B6CFF,1|300:9800$27C46A,2|40:1500$E8503A
GET /info?media=sample.ome.tif
```

`c` = comma-separated `channel|start:end$RRGGBB`: window start->black, end->full colour, additive blend.
Optional: `x,y,w,h` (pixels at that level, all four together), `series`, `fmt=png|jpeg`.

## OME-Zarr

Place an image store directly under `MEDIA_ROOT`, for example `media/sample.ome.zarr/`, then use the same endpoints:

```
GET /info?media=sample.ome.zarr
GET /render?media=sample.ome.zarr&z=0&t=0&level=0&c=0|0:65535$FFFFFF
```

Supports NGFF 0.4 metadata in Zarr v2 `.zattrs` and NGFF 0.5 metadata in Zarr v3 `zarr.json`
(`attributes.ome`). Stores are identified by metadata, not their directory suffix. Image axes must be unique
`t`, `c`, `z`, `y`, `x` axes with X/Y present; omitted T/Z/C dimensions have size 1. XY pyramid arrays must share
dtype and T/Z/C dimensions. OMERO channel labels/colors are exposed by `/info`.

`series` selects an entry in the image group's `multiscales` list (default 0), not a well or plate. Region coordinates
are pixels in the selected pyramid level, not physical coordinates; NGFF physical transforms are not applied.
Level scale factors in `/info` are spatial size ratios relative to level 0, matching the TIFF reader.
Regions clip at the right/bottom edge. Reads decode only intersecting chunks and return `(C,H,W)` in requested
channel order. Remote stores, ZIP archives, and plate/well traversal are not supported.
