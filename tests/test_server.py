import base64
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from adapters.synthetic.generator import SynthConfig, generate
from core.server import create_app
from protocol.io import read_ndjson
from protocol.models import Overrides, Snapshot
from protocol.ts import generate as gen_ts, DEFAULT_OUT

FIXTURE = Path("tests/fixtures/toy_scene.ndjson")


@pytest.fixture()
def client(tmp_path):
    return TestClient(create_app(gt_dir=tmp_path / "gt"))


def _post(client, snap: Snapshot) -> Overrides:
    r = client.post("/correct", content=snap.model_dump_json(by_alias=True), headers={"content-type": "application/json"})
    assert r.status_code == 200, r.text
    ov = Overrides.model_validate(r.json())
    assert ov.frame == snap.frame
    assert {p.id for p in ov.palette_overrides} <= {o.id for o in snap.objects}
    return ov


def test_toy_and_synthetic_same_endpoint(client):
    toy = list(read_ndjson(FIXTURE, Snapshot))
    synth = list(generate(SynthConfig(n_objects=8, n_frames=10), seed=0))
    assert len(toy) >= 10 and len(synth) == 10
    assert any(o.emissive_rgb[0] > 1 or o.emissive_rgb[1] > 1 for s in toy for o in s.objects)
    for s in toy + synth:
        ov = _post(client, s)
        assert len(ov.palette_overrides) == len(s.objects)


def test_rejects_invalid_snapshot(client):
    assert client.post("/correct", json={"objects": []}).status_code == 422


def test_cors_for_vite(client):
    r = client.options("/correct", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_gt_written(client, tmp_path):
    w, h = 4, 2
    buf = lambda v: base64.b64encode(bytes([v]) * (w * h * 4)).decode()
    body = {"frame": 7, "width": w, "height": h, "legend": {"instance": {"car": 1}},
            "buffers": {"instance": buf(1), "class": buf(2), "state": buf(3), "color": buf(4)}}
    r = client.post("/gt", json=body)
    assert r.status_code == 200
    z = np.load(tmp_path / "gt" / "gt_000007.npz")
    assert z["instance"].shape == (h, w, 4) and set(z.files) == {"instance", "class", "state", "color"}
    bad = dict(body, buffers={"instance": buf(1)[:-4]})
    assert client.post("/gt", json=bad).status_code == 422
    assert client.post("/gt", json=dict(body, buffers={"../x": buf(1)})).status_code == 422


def test_ts_types_in_sync():
    assert Path(DEFAULT_OUT).read_text() == gen_ts(), "run `python -m protocol.ts`"
