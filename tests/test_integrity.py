"""Pipeline integrity: (1) what the solver does not change is emitted exactly, what it changes lands on the intended lit
colour; (2) budget_ms pays for the correction stage only. Toy, synthetic and traffic data only."""
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from adapters.synthetic import SynthConfig, generate
from core.lighting import LightField, object_lit
from core.perception import color as C
from core.server import Corrector, create_app
from core.solver.problem import Problem, Scene, SolverConfig
from core.solver.solver import changed_groups, display_colors
from protocol.io import read_ndjson
from protocol.models import Overrides, Snapshot

FIX = Path(__file__).parent / "fixtures"
CASES = ("toy_scene", "synthetic", "traffic_day", "traffic_night", "traffic_fog", "traffic_rain")
OBS = "tr-0-000"


def _snaps(name):
    if name == "synthetic":
        out = []
        for lighting_seed in (0, 1, 2):
            out.append(next(generate(SynthConfig(n_objects=40, n_frames=1, palette="confusable"), lighting_seed)))
    else:
        out = list(read_ndjson(FIX / f"{name}.ndjson", Snapshot))
    return [s.model_copy(update={"observer_id": OBS}) for s in out]


def _lit(snap, overrides):
    """Displayed lit colour per object under {id: PaletteOverride}."""
    fld = LightField.from_lighting(snap.lighting)
    out = []
    for o in snap.objects:
        p = overrides.get(o.id)
        if p is not None:
            o = o.model_copy(update={"albedo_rgb": p.albedo,
                                     "emissive_rgb": o.emissive_rgb if p.emissive is None else p.emissive})
        out.append(object_lit(fld, o)[0])
    return np.array(out)


def _de(a, b):
    return C.ciede2000(C.linear_to_lab(a), C.linear_to_lab(b))


@pytest.mark.parametrize("name", CASES)
def test_all_skip_correct_returns_original_colours(name, tmp_path):
    client = TestClient(create_app(gt_dir=tmp_path, profiles_db=tmp_path / "p.sqlite"))
    for s in _snaps(name):
        s = s.model_copy(update={"budget_ms": 1e-3})                    # no planning budget -> every group skipped
        r = client.post("/correct", content=s.model_dump_json(by_alias=True), headers={"content-type": "application/json"})
        assert r.status_code == 200, r.text
        ov = Overrides.model_validate(r.json())
        tc = ov.diagnostics.tier_counts
        if tc:
            assert sum(v for k, v in tc.items() if k != "skip") == 0, tc
        shown = {p.id: p for p in ov.palette_overrides}
        assert _de(_lit(s, shown), _lit(s, {})).max() < 0.01
        for o in s.objects:
            if o.id in shown:
                assert shown[o.id].albedo == tuple(o.albedo_rgb) and shown[o.id].emissive == tuple(o.emissive_rgb)


@pytest.mark.parametrize("name", CASES)
def test_changed_group_lands_on_intended_lit_colour_and_others_are_exact(name):
    from protocol.models import PaletteOverride
    obs = [Corrector().table[OBS]]
    rng = np.random.default_rng(5)
    for s in _snaps(name):
        sc = Scene.from_snapshot(s)
        pb = Problem(sc, obs, SolverConfig(delta=15.0))
        for trial in range(4):                                         # random feasible moves of every other group
            P = pb.lab0.copy()
            moved = np.arange(trial % 2, pb.E, 2)
            P[moved] += rng.uniform(-1, 1, (len(moved), 3)) * 15.0
            P = np.where(np.isin(np.arange(pb.E), moved)[:, None], pb.project(P[None])[0], pb.lab0)
            want = display_colors(pb, P)
            mem = changed_groups(pb, P)[sc.groups]
            assert np.array_equal(want[~mem], sc.colors[~mem])
            alb, em = sc.to_overrides(want)
            assert (alb >= 0).all() and (alb <= 1).all() and (em >= 0).all()
            po = {o.id: PaletteOverride(id=o.id, albedo=tuple(alb[i]), emissive=tuple(em[i])) for i, o in enumerate(s.objects)}
            assert _de(_lit(s, po), want).max() < 0.05
            for i in np.flatnonzero(~mem):
                assert tuple(alb[i]) == tuple(s.objects[i].albedo_rgb) and tuple(em[i]) == tuple(s.objects[i].emissive_rgb)


@pytest.mark.parametrize("extreme", (0.0, 1.0))
def test_round_trip_handles_clipped_targets(extreme):
    for name in ("toy_scene", "traffic_night"):
        s = _snaps(name)[0]
        sc = Scene.from_snapshot(s)
        want = np.full_like(sc.colors, extreme)
        alb, em = sc.to_overrides(want)
        got = np.clip(alb * sc.gain + em * sc.exposure, 0, 1)
        assert np.allclose(got, want, atol=1e-6) and (alb >= 0).all() and (alb <= 1).all() and (em >= 0).all()


def test_budget_40ms_pays_for_correction_only():
    snap = _snaps("synthetic")[0]
    snap = next(generate(SynthConfig(n_objects=100, n_frames=1, palette="confusable"), 3)).model_copy(
        update={"observer_id": OBS, "budget_ms": 40.0})
    cor = Corrector()
    if cor.allocator.tables is None:
        pytest.skip("run python -m bench.allocator_calibrate")
    ov = cor(snap)
    d = ov.diagnostics
    assert cor.allocator.tiers[2].calls > 0 and d.tier_counts["learned"] > 0
    assert d.tier_counts["skip"] < sum(d.tier_counts.values())
    lat = d.stage_latency_ms
    assert {"analysis", "planning", "solving", "correction", "verify", "total"} <= set(lat)
    assert lat["correction"] == pytest.approx(lat["planning"] + lat["solving"], abs=0.5)


def test_floor_goes_to_highest_importance_groups():
    from core.allocator import Allocator
    snap = next(generate(SynthConfig(n_objects=100, n_frames=1, palette="confusable"), 3))
    sc = Scene.from_snapshot(snap)
    pb = Problem(sc, [Corrector().table[OBS]], SolverConfig())
    al = Allocator()
    if al.tables is None:
        pytest.skip("run python -m bench.allocator_calibrate")
    imp = np.ones(pb.E)
    pl = al.plan(pb, 40.0, imp)
    active = np.flatnonzero(pl._active)
    imp[active[-3:]] = 9.0
    pl = al.apply_floor(al.plan(pb, 40.0, imp))
    assert all(pl.tiers[g] in (1, 2, 3, 4) for g in active[-3:])
    assert (al.apply_floor(al.plan(pb, 0.0, imp)).tiers == 0).all()        # no budget left -> no floor
