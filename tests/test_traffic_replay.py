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


def _fixture(env):
    from bench.experiments import traffic_replay as T
    return T.window(list(read_ndjson(FIX / f"traffic_{env}.ndjson", Snapshot)), 1, 3)


@pytest.mark.parametrize("env", ENVS)
def test_fixture_scene_has_realistic_cvd_hazard_pairs_under_train_observers(env):
    """Scene-design check uses TRAIN observers only (held-out never looked at to pick colours)."""
    from bench.experiments import traffic_replay as T
    from bench.experiments.stability_ablation import Ctx
    from core.observers import train_observers
    snaps, k0 = _fixture(env)
    ids = {o.id for o in snaps[-1].objects}
    assert {"hedge_nw_n", "hedge_se_s", "cone_0", "worker0", "worker1", "works_lamp", "bystander0"} <= ids
    assert any(o.id.endswith("_ind_r") and o.state == "hazard" for o in snaps[-1].objects)
    for kind in ("protan", "deutan"):
        tr = [o for o in train_observers(60, 0) if o.kind == kind][:3]
        ctx = Ctx(snaps, tr[0], {o.id: o for o in tr}, tr)
        pairs = {p for p, _ in T.critical(ctx, k0, len(snaps) - 1, len(tr))}
        assert len(pairs) >= 2, (kind, pairs)


@pytest.fixture(scope="module")
def fixture_run():
    from bench.experiments import traffic_replay as T
    from core.observers import train_observers
    snaps, k0 = _fixture("night")
    tr = [o for o in train_observers(60, 0) if o.kind == "deutan"][:2]
    params = dict(n_heldout=2, solver={"step_min": 0.3, "retries": 0, "max_iter": 5}, workers=0,
                  arms=("uncorrected", "none", "event_driven"))
    load = lambda env, seed, p: (snaps, k0, 6 / 60, "deutan", tr)
    return T, params, load


def test_run_env_reports_frames_and_ms_and_censoring(fixture_run, monkeypatch):
    T, params, load = fixture_run
    monkeypatch.setattr(T, "_load", load)
    monkeypatch.setattr(T, "critical", lambda *a, **k: [(("hedge_nw_n", "veh0"), 0), (("tree_se_canopy", "veh1_ind"), 1)])
    out = T.run_env("night", 0, params)
    dt_ms = 6 / 60 * 1e3
    assert "collision_trigger_frames" not in out["uncorrected"]
    for a in ("none", "event_driven"):
        m = out[a]
        assert m["trigger_missed"] == 0.0
        assert m["collision_trigger_ms"] == pytest.approx(m["collision_trigger_frames"] * dt_ms)
        assert m["display_frames"] == m["collision_trigger_frames"] + 1 and m["display_ms"] == pytest.approx(m["display_frames"] * dt_ms)
        assert m["n_critical"] > 0 and m["no_critical"] == 0.0
        assert m["latency_lb_frames"] >= m.get("latency_frames", 0.0)
        assert m["n_censored_pairs"] == pytest.approx(m["unresolved"] * m["n_critical"])
        if m["n_censored_pairs"] == m["n_critical"]:
            assert "latency_frames" not in m            # all censored -> no latency number, never 0


def test_run_env_without_critical_pairs_is_censored_not_zero(fixture_run, monkeypatch):
    T, params, load = fixture_run
    monkeypatch.setattr(T, "_load", load)
    monkeypatch.setattr(T, "critical", lambda *a, **k: [])
    out = T.run_env("night", 0, params)
    for m in out.values():
        assert m["no_critical"] == 1.0 and m["n_critical"] == 0.0
        assert not {"latency_frames", "latency_ms", "latency_lb_frames", "unresolved"} & set(m)


def test_prefetch_block_and_censoring_report(tmp_path):
    import csv
    from bench.experiments import traffic_replay as T
    assert T.prefetch_block(0, 12) == list(range(12)) == T.prefetch_block(11, 12)
    assert T.prefetch_block(13, 12) == list(range(12, 24)) and T.prefetch_block(5, 1) == [5]
    rows = []
    for s, (d, n) in enumerate([(1, 0), (0, 0), (1, 1)]):
        for m, v in (("no_critical_day", d), ("no_critical_night", n), ("no_critical_envs", d + n)):
            rows.append(dict(seed=s, arm="none", metric=m, value=v))
    p = tmp_path / "raw.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    rep = T.censoring_report(p)
    assert "seeds=3" in rep and "day: no_critical_seeds=2" in rep and "night: no_critical_seeds=1" in rep
    assert "all_envs_no_critical_seeds=1" in rep


def test_log_path_tracks_scene_version():
    from bench.experiments import traffic_replay as T
    p = T.log_path("day", 3, 540, 3)
    assert p.name.endswith(f"_{T.scene_version()}.ndjson") and len(T.scene_version()) == 8


def test_fixture_environments_differ_and_events_trigger():
    L = {e: next(iter(read_ndjson(FIX / f"traffic_{e}.ndjson", Snapshot))).lighting for e in ENVS}
    assert len({(tuple(map(tuple, l.coeffs)), l.exposure, l.white_point, l.fog_density) for l in L.values()}) == 4
    assert L["fog"].fog_density > L["rain"].fog_density > 0 and L["day"].fog_density is None
    det = TriggerDetector()
    fired = [det.detect(s) for s in read_ndjson(FIX / "traffic_day.ndjson", Snapshot)]
    assert any("collision" in f for f in fired)
