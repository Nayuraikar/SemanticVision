"""Held-out guards for the instrument-panel dashboard: fixtures replay through /correct unchanged, the replay leaves
every frozen core file untouched, and nothing that tunes the pipeline can see the dashboard."""
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from core.server import create_app
from protocol.io import read_ndjson
from protocol.models import Overrides, Snapshot
from tests.test_core_hash import assert_core_unchanged, current_hashes

ROOT = Path(__file__).resolve().parents[1]
LIGHTINGS = ("normal", "dim", "glare")
FIX = ROOT / "tests/fixtures"
TEXT = {".py", ".json", ".toml", ".yaml", ".yml", ".cfg", ".txt", ".md", ".sh", ".ipynb"}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("dash")
    return TestClient(create_app(gt_dir=d, profiles_db=d / "p.sqlite"))


@pytest.mark.parametrize("light", LIGHTINGS)
def test_fixture_replays_through_correct_with_frozen_core(client, light):
    assert_core_unchanged()
    before = current_hashes()
    path = FIX / f"dashboard_{light}.ndjson"
    raw = path.read_text().strip().splitlines()
    snaps = list(read_ndjson(path, Snapshot))
    assert len(snaps) == len(raw) == 4
    assert any(e.object_id == "master_warning" and e.data.get("to") == "red" for s in snaps for e in s.events)
    for line, s in zip(raw, snaps):
        r = client.post("/correct", content=line, headers={"content-type": "application/json"})   # bytes as recorded
        assert r.status_code == 200, r.text
        ov = Overrides.model_validate(r.json())
        ids = {o.id for o in s.objects}
        assert ov.frame == s.frame and ov.diagnostics.session_id == s.session_id
        assert ov.palette_overrides and {p.id for p in ov.palette_overrides} <= ids
        assert {c.id for c in ov.cues} <= ids and all(set(p.ids) <= ids for p in ov.confusion_pairs)
    assert current_hashes() == before
    assert_core_unchanged()


def test_fixture_lighting_conditions_differ():
    L = {k: next(iter(read_ndjson(FIX / f"dashboard_{k}.ndjson", Snapshot))).lighting for k in LIGHTINGS}
    assert len({(tuple(map(tuple, v.coeffs)), v.exposure, v.white_point) for v in L.values()}) == 3
    assert L["dim"].white_point[2] < L["glare"].white_point[2] < L["normal"].white_point[2]


def _sources(*roots):
    for r in roots:
        p = ROOT / r
        for f in ([p] if p.is_file() else p.rglob("*")):
            rel = f.relative_to(ROOT).parts
            if f.is_file() and f.suffix in TEXT and "__pycache__" not in rel and rel[:2] not in (("bench", "logs"), ("bench", "results")):
                yield f


# KNOWN CAVEAT, predates the held-out environment: the engine-free synthetic generator has a 4-colour palette NAMED
# "dashboard" (red/green/amber/blue spheres). It was in the learned solver's training mix and in the held-out set used
# to select the importance MLP. It is not this environment, but the held-out claim must be stated with it. Only these
# exact lines are tolerated; any other mention fails.
PRE_EXISTING = {
    ("learning/data.py", 'PALETTES = ("random", "confusable", "traffic", "dashboard")'),
    ("bench/experiments/importance_eval.py", 'HELD = dict(palette=["traffic", "dashboard"], lighting=["fog", "rain"], n=[30, 40])'),
}
# the 3D held-out environment itself (code, recorder, fixtures, logs): no exceptions
ENV_ARTIFACTS = r"dashboardSim|DashboardSim|recordDashboard|scenes/dashboard|logs/dashboard|dashboard_(normal|dim|glare)|dashboard\.ndjson|DASH_DIR|dash_log"


def _mention_lines(f):
    rel = f.relative_to(ROOT).as_posix()
    return [(rel, ln.strip()) for ln in f.read_text(errors="ignore").splitlines() if re.search("dashboard", ln, re.I)]


def test_nothing_that_tunes_the_pipeline_references_the_dashboard():
    files = list(_sources("learning", "calibration", "core", "bench/allocator_calibrate.py"))
    assert len(files) > 40
    hits = {h for f in files for h in _mention_lines(f)}
    assert hits <= PRE_EXISTING, hits - PRE_EXISTING
    assert [str(f) for f in files if re.search(ENV_ARTIFACTS, f.read_text(errors="ignore"))] == []


def test_only_the_transfer_experiment_reads_dashboard_data_under_bench():
    allowed = {"bench/experiments/transfer.py", "bench/configs/transfer.json", "bench/configs/transfer_b500.json"}
    others = [f for f in _sources("bench") if f.relative_to(ROOT).as_posix() not in allowed]
    hits = {h for f in others for h in _mention_lines(f)}
    assert hits <= PRE_EXISTING, hits - PRE_EXISTING
    assert [str(f) for f in others if re.search(ENV_ARTIFACTS, f.read_text(errors="ignore"))] == []
    assert all(_mention_lines(ROOT / a) for a in allowed)
    # and nobody else imports the transfer experiment (it cannot leak dashboard data into another bench)
    users = [f for f in _sources("bench", "core", "learning", "calibration")
             if f.name != "transfer.py" and re.search(r"experiments[. ].*\btransfer\b", f.read_text(errors="ignore"))]
    assert users == []
