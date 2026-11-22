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


OBS = "tr-0-000"


def _post(client, snap: Snapshot, obs=OBS) -> Overrides:
    snap = snap.model_copy(update={"observer_id": obs})
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


def _infeasible_snap() -> Snapshot:
    snap = next(iter(generate(SynthConfig(n_objects=4, n_frames=1), seed=1)))
    bb = snap.objects[0].bbox
    o = [x.model_copy(update={"cls": c, "albedo_rgb": (0.2, 0.3, 0.2), "bbox": bb.model_copy(update={"x": 0.4 + 0.1 * i, "y": 0.4, "w": 0.1, "h": 0.1})})
         for i, (x, c) in enumerate(zip(snap.objects[:2], ("sky", "grass")))]
    return snap.model_copy(update={"objects": o})


def test_real_solver_diagnostics_and_cues(client):
    synth = next(iter(generate(SynthConfig(n_objects=8, n_frames=1), seed=2)))
    ov = _post(client, synth)
    assert ov.diagnostics.solver_tier in ("exact", "allocated")
    assert {"scene", "solve", "pairs_cues", "total"} <= set(ov.diagnostics.stage_latency_ms)
    assert any(p.albedo != o.albedo_rgb for p, o in zip(ov.palette_overrides, synth.objects))
    for cp in ov.confusion_pairs:
        assert cp.de00_before >= 0 and cp.de00_after >= 0
    snap = _infeasible_snap()
    client.app.state.corrector.cfg_extra = {"delta_mem": 0.0}   # memory colours frozen -> pair cannot be separated
    ov = _post(client, snap, "he-0-000")  # held-out observer id is accepted
    bad = [p for p in ov.confusion_pairs if not p.feasible]
    assert bad and ov.cues and {c.id for c in ov.cues} <= {o.id for o in snap.objects}
    assert "infeasible_pairs" in ov.diagnostics.fallback_flags


def test_cache_by_signature(client):
    snaps = list(generate(SynthConfig(n_objects=6, n_frames=1), seed=3))
    s = snaps[0]
    a = _post(client, s)
    n = client.app.state.corrector.solves
    b = _post(client, s.model_copy(update={"frame": 5}))
    assert client.app.state.corrector.solves == n and b.diagnostics.solver_tier == "cached" and b.frame == 5
    assert [p.albedo for p in b.palette_overrides] == [p.albedo for p in a.palette_overrides]
    _post(client, s, "tr-0-001")
    assert client.app.state.corrector.solves == n + 1


def test_observers_listing_and_unknown(client):
    r = client.get("/observers").json()
    assert r["train"] and r["heldout"] and all(o["split"] == "heldout" for o in r["heldout"])
    assert {o["id"] for o in r["train"]}.isdisjoint(o["id"] for o in r["heldout"])
    snap = next(iter(generate(SynthConfig(n_objects=3, n_frames=1), seed=0)))
    assert client.post("/correct", content=snap.model_copy(update={"observer_id": "nope"}).model_dump_json(by_alias=True),
                       headers={"content-type": "application/json"}).status_code == 404


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
