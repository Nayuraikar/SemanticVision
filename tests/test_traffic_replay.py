"""Recorded Three.js/Rapier traffic fixtures (one per environment) replay through /correct unchanged."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from core.server import create_app
from core.stabiliser.triggers import TriggerDetector
from protocol.io import read_ndjson
from protocol.models import Overrides, Snapshot

ENVS = ("day", "night", "fog", "rain")
FIX = Path("tests/fixtures")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    return TestClient(create_app(gt_dir=tmp_path_factory.mktemp("gt")))


@pytest.mark.parametrize("env", ENVS)
def test_fixture_replays_through_correct(client, env):
    raw = (FIX / f"traffic_{env}.ndjson").read_text().strip().splitlines()
    snaps = list(read_ndjson(FIX / f"traffic_{env}.ndjson", Snapshot))
    assert len(snaps) == len(raw) == 4
    assert sum(e.type == "collision" for s in snaps for e in s.events) == 1
    assert snaps[0].lighting.kind == "sh2"
    for line, s in zip(raw, snaps):
        r = client.post("/correct", content=line, headers={"content-type": "application/json"})   # bytes as recorded
        assert r.status_code == 200, r.text
        ov = Overrides.model_validate(r.json())
        assert ov.frame == s.frame
        assert {p.id for p in ov.palette_overrides} <= {o.id for o in s.objects}
        assert ov.diagnostics.session_id == s.session_id


def test_fixture_environments_differ_and_events_trigger():
    L = {e: next(iter(read_ndjson(FIX / f"traffic_{e}.ndjson", Snapshot))).lighting for e in ENVS}
    assert len({(tuple(map(tuple, l.coeffs)), l.exposure, l.white_point, l.fog_density) for l in L.values()}) == 4
    assert L["fog"].fog_density > L["rain"].fog_density > 0 and L["day"].fog_density is None
    det = TriggerDetector()
    fired = [det.detect(s) for s in read_ndjson(FIX / "traffic_day.ndjson", Snapshot)]
    assert any("collision" in f for f in fired)
