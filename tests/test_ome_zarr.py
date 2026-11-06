import io
import json

import numpy as np
import pytest
import zarr
from fastapi.testclient import TestClient
from PIL import Image

from app.models import Region
from app.readers.ome_zarr import OmeZarrReader
from app.readers.registry import open_image
from app.storage import MediaStorage
from tests.sample import make_sample


def make_zarr(path, *, version="0.4", axes="tczyx"):
    shape = {"t": 2, "c": 3, "z": 2, "y": 8, "x": 10}
    data = np.arange(np.prod([shape[a] for a in axes]), dtype=np.uint16).reshape(
        tuple(shape[a] for a in axes)
    )
    group = zarr.open_group(path, mode="w", zarr_format=2 if version == "0.4" else 3)
    group.create_array("0", data=data, chunks=tuple(min(shape[a], 4) for a in axes))
    small = data[tuple(slice(None, None, 2) if a in "xy" else slice(None) for a in axes)]
    group.create_array("1", data=small)
    image = {
        "axes": [{"name": a, "type": "channel" if a == "c" else "time" if a == "t" else "space"} for a in axes],
        "datasets": [
            {"path": str(i), "coordinateTransformations": [
                {"type": "scale", "scale": [2**i if a in "xy" else 1 for a in axes]}
            ]} for i in range(2)
        ],
    }
    omero = {"channels": [
        {"label": name, "color": color}
        for name, color in zip(("DAPI", "GFP", "RFP"), ("0000ff", "00ff00", "ff0000"))
    ]} if "c" in axes else {}
    if version == "0.4":
        image["version"] = version
        group.attrs.update({"multiscales": [image], "omero": omero})
    else:
        group.attrs["ome"] = {"version": version, "multiscales": [image], "omero": omero}
    group.store.close()
    return data


@pytest.mark.parametrize("version", ["0.4", "0.5"])
@pytest.mark.parametrize("axes", ["tczyx", "xyczt", "yx"])
def test_read_regions(tmp_path, version, axes):
    path = tmp_path / "sample.ome.zarr"
    data = make_zarr(path, version=version, axes=axes)
    with open_image(MediaStorage(path)) as handle:
        assert (handle.info.size_x, handle.info.size_y) == (10, 8)
        assert [(lv.width, lv.height, lv.scale) for lv in handle.info.levels] == [
            (10, 8, (1, 1)), (5, 4, (2, 2))
        ]
        assert handle.info.pixel_type == "uint16"
        t, z = (1, 1) if "t" in axes else (0, 0)
        channels = [2, 0, 2] if "c" in axes else [0]
        result = handle.read_region(t=t, z=z, channels=channels, region=Region(2, 3, 20, 20))
        expected = []
        for c in channels:
            selection = {"t": t, "z": z, "c": c, "y": slice(3, 8), "x": slice(2, 10)}
            pixels = data[tuple(selection[a] for a in axes)]
            expected.append(pixels.T if axes.index("x") < axes.index("y") else pixels)
        np.testing.assert_array_equal(result, np.stack(expected))
        full = handle.read_region(t=t, z=z, channels=channels)
        np.testing.assert_array_equal(
            handle.read_region(t=t, z=z, channels=channels, level=1), full[:, ::2, ::2]
        )
        if "c" in axes:
            assert [ch.name for ch in handle.info.channels] == ["DAPI", "GFP", "RFP"]
            assert [ch.color for ch in handle.info.channels] == ["0000FF", "00FF00", "FF0000"]
        else:
            assert (handle.info.size_z, handle.info.size_t, handle.info.size_c) == (1, 1, 1)
    handle.close()
    with pytest.raises(ValueError, match="closed"):
        handle.read_region()


@pytest.mark.parametrize("kwargs", [
    {"level": -1}, {"level": 2}, {"z": 2}, {"t": -1},
    {"channels": [3]}, {"channels": []},
    {"region": Region(-1, 0, 1, 1)}, {"region": Region(10, 0, 1, 1)},
    {"region": Region(0, 0, 0, 1)},
])
def test_invalid_selection(tmp_path, kwargs):
    path = tmp_path / "sample.zarr"
    make_zarr(path)
    with open_image(MediaStorage(path)) as handle, pytest.raises(IndexError):
        handle.read_region(**kwargs)


def test_absent_axes_only_accept_zero(tmp_path):
    path = tmp_path / "sample.zarr"
    make_zarr(path, axes="yx")
    with open_image(MediaStorage(path)) as handle:
        for kwargs in ({"z": 1}, {"t": 1}, {"channels": [1]}):
            with pytest.raises(IndexError):
                handle.read_region(**kwargs)


def test_series_selection(tmp_path):
    path = tmp_path / "sample.zarr"
    make_zarr(path)
    group = zarr.open_group(path, mode="a")
    second = np.full((4, 6), 123, dtype=np.uint16)
    group.create_array("second", data=second)
    images = group.attrs["multiscales"]
    images.append({
        "version": "0.4", "axes": [{"name": "y"}, {"name": "x"}],
        "datasets": [{"path": "second"}],
    })
    group.attrs["multiscales"] = images
    # OMERO channel settings are group-wide; omit them for series with different C sizes.
    del group.attrs["omero"]
    group.store.close()
    with open_image(MediaStorage(path), series=1) as handle:
        np.testing.assert_array_equal(handle.read_region(), second[None])
    for series in (-1, 2):
        with pytest.raises(IndexError):
            open_image(MediaStorage(path), series=series)


@pytest.fixture
def client(tmp_path, monkeypatch):
    import app.main as main
    monkeypatch.setattr(main, "MEDIA_ROOT", tmp_path)
    for version in ("0.4", "0.5"):
        make_zarr(tmp_path / f"sample-{version}.ome.zarr", version=version)
    return TestClient(main.app)


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_endpoints(client, version):
    media = f"sample-{version}.ome.zarr"
    info = client.get("/info", params={"media": media})
    assert info.status_code == 200
    assert (info.json()["size_x"], info.json()["size_z"], info.json()["size_t"]) == (10, 2, 2)
    params = {
        "media": media, "c": "0|0:1000$FF0000", "z": 1, "t": 1,
        "x": 8, "y": 6, "w": 4, "h": 4,
    }
    response = client.get("/render", params=params)
    assert response.status_code == 200
    image = Image.open(io.BytesIO(response.content))
    assert image.size == (2, 2)
    # t=1, c=0, z=1, y=6, x=8 -> 628, mapped to 160 red.
    assert image.getpixel((0, 0)) == (160, 0, 0)
    jpeg = client.get("/render", params={**params, "fmt": "jpeg"})
    assert jpeg.status_code == 200 and jpeg.headers["content-type"] == "image/jpeg"
    assert client.get("/render", params={**params, "series": 9}).status_code == 404
    assert client.get("/render", params={**params, "z": 9}).status_code == 400
    assert client.get("/render", params={**params, "level": 9}).status_code == 400


@pytest.mark.parametrize("change", [
    {"axes": [{"name": "q"}, {"name": "x"}]},
    {"datasets": [{"path": "missing"}]},
    {"datasets": [{"path": "../outside"}]},
    {"datasets": [{"path": "0"}], "axes": [{"name": "y"}, {"name": "x"}]},
    {"version": "9.0"}, {"datasets": []},
])
def test_malformed_metadata(client, tmp_path, change):
    group = zarr.open_group(tmp_path / "sample-0.4.ome.zarr", mode="a")
    images = group.attrs["multiscales"]
    images[0].update(change)
    group.attrs["multiscales"] = images
    group.store.close()
    response = client.get("/info", params={"media": "sample-0.4.ome.zarr"})
    assert response.status_code == 415


def test_non_ome_and_invalid_json(client, tmp_path):
    path = tmp_path / "plain.zarr"
    zarr.open_group(path, mode="w", zarr_format=2).store.close()
    assert not OmeZarrReader().probe(MediaStorage(path))
    assert client.get("/info", params={"media": path.name}).status_code == 415
    (path / ".zattrs").write_text(json.dumps({"unrelated": True}))
    assert client.get("/info", params={"media": path.name}).status_code == 415
    (path / ".zattrs").write_text("{")
    assert client.get("/info", params={"media": path.name}).status_code == 415


def test_region_reads_are_lazy(tmp_path, monkeypatch):
    path = tmp_path / "sample.zarr"
    make_zarr(path)
    selections = []
    original = zarr.Array.__getitem__

    def record_selection(array, selection):
        selections.append(selection)
        return original(array, selection)

    monkeypatch.setattr(zarr.Array, "__getitem__", record_selection)
    with open_image(MediaStorage(path)) as handle:
        assert selections == []
        handle.read_region(t=1, z=1, channels=[2], region=Region(2, 3, 4, 2))
    assert selections == [(1, 2, 1, slice(3, 5), slice(2, 6))]


@pytest.mark.parametrize("metadata", [
    {"multiscales": []},
    {"multiscales": [None]},
    {"multiscales": [{"version": "0.4", "axes": [{"name": "y"}, {"name": "y"}]}]},
    {"ome": []},
])
def test_invalid_structure(client, tmp_path, metadata):
    group = zarr.open_group(tmp_path / "sample-0.4.ome.zarr", mode="a")
    group.attrs.clear()
    group.attrs.update(metadata)
    group.store.close()
    assert client.get("/info", params={"media": "sample-0.4.ome.zarr"}).status_code == 415


def test_failure_closes_store(tmp_path, monkeypatch):
    path = tmp_path / "sample.zarr"
    make_zarr(path)
    original = zarr.storage.LocalStore.close
    closed = []

    def close(store):
        closed.append(store)
        original(store)

    monkeypatch.setattr(zarr.storage.LocalStore, "close", close)
    with pytest.raises(IndexError):
        OmeZarrReader().open(MediaStorage(path), series=9)
    assert len(closed) == 1


@pytest.mark.parametrize("shape,dtype", [
    ((2, 3, 1, 4, 5), np.uint16),
    ((2, 3, 2, 4, 5), np.uint8),
    ((2, 3, 2, 12, 5), np.uint16),
])
def test_incompatible_levels(client, tmp_path, shape, dtype):
    group = zarr.open_group(tmp_path / "sample-0.4.ome.zarr", mode="a")
    del group["1"]
    group.create_array("1", data=np.zeros(shape, dtype=dtype))
    group.store.close()
    assert client.get("/info", params={"media": "sample-0.4.ome.zarr"}).status_code == 415


def test_tiff_zarr_render_equivalence(tmp_path, monkeypatch):
    import app.main as main
    import tifffile

    monkeypatch.setattr(main, "MEDIA_ROOT", tmp_path)
    tiff_path = tmp_path / "sample.ome.tif"
    make_sample(tiff_path)
    group = zarr.open_group(tmp_path / "sample.ome.zarr", mode="w", zarr_format=2)
    with tifffile.TiffFile(tiff_path) as tif:
        for i, level in enumerate(tif.series[0].levels):
            group.create_array(str(i), data=level.asarray())
    group.attrs["multiscales"] = [{
        "version": "0.4", "axes": [{"name": a} for a in "zcyx"],
        "datasets": [{"path": str(i)} for i in range(3)],
    }]
    group.store.close()
    client = TestClient(main.app)
    params = {"c": "0|0:3000$0000FF,1|0:12000$00FF00", "z": 8, "level": 1}
    tiff = client.get("/render", params={**params, "media": tiff_path.name})
    ome_zarr = client.get("/render", params={**params, "media": "sample.ome.zarr"})
    assert tiff.status_code == ome_zarr.status_code == 200
    assert tiff.content == ome_zarr.content
