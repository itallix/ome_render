"""Synthetic pyramidal OME-TIFF: Z=16, C=3, 256x256 uint16, tiled, 3 levels.
Run: python -m tests.sample media/sample.ome.tif"""
import sys
from pathlib import Path

import numpy as np
import tifffile


def make_sample(path: Path | str) -> None:
    Z, C, N = 16, 3, 256
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)
    data = np.zeros((Z, C, N, N), np.uint16)
    centers = [(64, 64), (128, 128), (192, 192)]
    peaks = [3000, 12000, 1800]
    for z in range(Z):
        for c, ((cy, cx), peak) in enumerate(zip(centers, peaks)):
            r2 = (yy - cy) ** 2 + (xx - cx) ** 2
            data[z, c] = (peak * np.exp(-r2 / (2 * 30.0**2)) * (1 - abs(z - 8) / 10)).clip(0).astype(np.uint16)

    opts = dict(photometric="minisblack", tile=(64, 64), compression="zlib")
    meta = {"axes": "ZCYX", "Channel": {"Name": ["DAPI", "GFP", "RFP"]}}
    with tifffile.TiffWriter(path, ome=True) as tif:
        tif.write(data, subifds=2, metadata=meta, **opts)
        lvl = data
        for _ in range(2):
            lvl = lvl[..., ::2, ::2]
            tif.write(np.ascontiguousarray(lvl), subfiletype=1, metadata=None, **opts)


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "media/sample.ome.tif")
    out.parent.mkdir(parents=True, exist_ok=True)
    make_sample(out)
    print("wrote", out)
