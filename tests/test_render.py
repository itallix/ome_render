import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tests.sample import make_sample

C = "0|120:2400$3B6CFF,1|300:9800$27C46A,2|40:1500$E8503A"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("media")
    make_sample(root / "sample.ome.tif")
    import app.main as m
    m.MEDIA_ROOT = root
    return TestClient(m.app)


def test_info(client):
    d = client.get("/info", params={"media": "sample.ome.tif"}).json()
    assert (d["size_x"], d["size_z"], d["size_c"], d["size_t"]) == (256, 16, 3, 1)
    assert [lv["width"] for lv in d["levels"]] == [256, 128, 64]
    assert [ch["name"] for ch in d["channels"]] == ["DAPI", "GFP", "RFP"]


def test_render_example_request(client):
    r = client.get(f"/render?media=sample.ome.tif&z=12&t=0&level=2&c={C}")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    img = Image.open(io.BytesIO(r.content))
    assert img.size == (64, 64)
    r_, g_, b_ = img.getpixel((32, 32))
    # GFP peak at z=12 is 7200 -> (7200-300)/(9800-300)=0.726 of 0x27C46A green (196) ~ 142
    assert abs(g_ - 142) <= 6 and g_ > r_ and g_ > b_


def test_region_and_jpeg(client):
    r = client.get("/render", params={"media": "sample.ome.tif", "c": C, "level": 0,
                                      "x": 192, "y": 192, "w": 128, "h": 128, "fmt": "jpeg"})
    assert r.status_code == 200
    assert Image.open(io.BytesIO(r.content)).size == (64, 64)  # clamped at the edge


@pytest.mark.parametrize("params", [
    {"c": "9|0:10$FFFFFF"},                 # channel out of range
    {"c": C, "z": 99},                      # z out of range
    {"c": C, "level": 7},                   # level out of range
    {"c": "0|10:5$FFFFFF"},                 # end <= start
    {"c": "garbage"},
    {"c": C, "x": 0, "y": 0},               # partial region
])
def test_bad_requests(client, params):
    assert client.get("/render", params={"media": "sample.ome.tif", **params}).status_code == 400


def test_missing_media(client):
    assert client.get("/render", params={"media": "nope.ome.tif", "c": C}).status_code == 404
