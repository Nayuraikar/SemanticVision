import itertools
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy import stats as st

from adapters.synthetic import SynthConfig, generate
from core.allocator import (Allocator, EntryPredictor, LinUCB, Tier, default_tiers, load_tables, mckp, group_stats,
                            TIER_NAMES)
from core.observers import train_observers
from core.server import create_app
from core.solver import Scene, SolverConfig
from core.solver.problem import Problem
from protocol.models import Overrides

TIER_MS = np.array([0, 0.25, 2.0, 30.0, 80.0])


def workload(n, seed):
    """TEST-ONLY parametric workload (the bench uses measured rows): concave benefit in cost, heavy-tailed value, noisy costs."""
    r = np.random.default_rng(seed)
    val = r.lognormal(0, 1.0, n)
    q = np.array([0, 0.3, 0.55, 0.85, 1.0])
    ben = val[:, None] * q[None] * r.uniform(0.7, 1.3, (n, 5))
    ben[:, 0] = 0
    cost = TIER_MS[None] * r.lognormal(0, 0.5, (n, 5))
    cost[:, 0] = 0
    return ben, cost


def value(b, c, ch):
    return mckp.total(b, c, ch)


@pytest.mark.parametrize("name", ["uniform", "greedy", "lagrangian", "dp"])
def test_solvers_feasible(name):
    for seed in range(5):
        b, c = workload(40, seed)
        B = 0.2 * c[:, 4].sum()
        ch = mckp.SOLVERS[name](b, c, B)
        assert ch.shape == (40,) and mckp.total(b, c, ch)[1] <= B + 1e-9


def test_bnb_matches_brute_force_and_bounds():
    for seed in range(3):
        b, c = workload(6, seed)
        B = 0.35 * c[:, 4].sum()
        best = max(value(b, c, np.array(t))[0] for t in itertools.product(range(5), repeat=6) if value(b, c, np.array(t))[1] <= B)
        ch, proven = mckp.bnb(b, c, B)
        assert proven and value(b, c, ch)[0] == pytest.approx(best)
        assert mckp.lp_bound(b, c, B) >= best - 1e-9
        assert value(b, c, mckp.dp(b, c, B, bins=2000))[0] <= best + 1e-9


def test_exit_mckp_beats_greedy_and_uniform_fixed_seeds():
    """Exit criterion (SPEC 5.8): MCKP (DP, Lagrangian) beats naive density-greedy and uniform at equal budget; B&B is optimal."""
    res = {m: [] for m in ("uniform", "greedy", "lagrangian", "dp", "bnb")}
    for seed in range(24):
        b, c = workload(60 if seed % 2 else 12, seed)
        B = (0.15 + 0.1 * (seed % 3)) * c[:, 4].sum()
        for m in ("uniform", "greedy", "lagrangian", "dp"):
            res[m].append(value(b, c, mckp.SOLVERS[m](b, c, B))[0])
        if seed % 2 == 0:                                  # small instances: branch and bound = offline optimum
            res["bnb"].append(value(b, c, mckp.bnb(b, c, B, time_limit=1.0)[0])[0])
    a = {m: np.array(v) for m, v in res.items()}
    for m in ("lagrangian", "dp"):
        for base in ("greedy", "uniform"):
            assert a[m].mean() > a[base].mean(), (m, base, a[m].mean(), a[base].mean())
            assert np.all(a[m] >= a[base] - 1e-9) or st.wilcoxon(a[m], a[base], alternative="greater").pvalue < 0.01
    ev = slice(0, None, 2)
    assert np.all(a["bnb"] >= np.array(res["lagrangian"])[ev] - 1e-9) and np.all(a["bnb"] >= np.array(res["dp"])[ev] - 1e-9)


def test_round_robin_cycles():
    b, c = workload(20, 0)
    rr = mckp.RoundRobin()
    B = 0.1 * c[:, 4].sum()
    served = [set(np.flatnonzero(rr(b, c, B) > 0)) for _ in range(6)]
    assert len(set().union(*served)) > len(served[0])


def test_linucb_learns_linear_reward():
    r = np.random.default_rng(0)
    th = np.array([0.0, 1.0, 2.0, 3.0])
    bd = LinUCB(4, n_tiers=3, alpha=0.2)
    for _ in range(400):
        x = np.r_[1.0, r.uniform(0, 1, 3)]
        t = int(r.integers(1, 3))
        bd.update(x, t, th @ x * t + r.normal(0, 0.05))
    x = np.r_[1.0, 0.5, 0.5, 0.5]
    est = bd.estimate(x[None], explore=False)[0]
    assert est[0] == 0 and est[2] == pytest.approx(2 * th @ x, rel=0.1) and est[1] == pytest.approx(th @ x, rel=0.1)


def test_predictor_flags_entering_only():
    from protocol import SnapshotBuilder
    pred = EntryPredictor(horizon=0.5)
    flagged = set()
    for f in range(30):
        t = f / 30
        b = SnapshotBuilder("o", 5.0, frame=f, t=t)
        b.ambient_dominant([0.3] * 3, [1, 1, 1], [0, 1, 0])
        b.add_object("in", "car", [0.5] * 3, bbox=(0.4, 0.4, 0.1, 0.1), screen_area=0.01, depth=5, velocity=(0.0, 0.0))
        b.add_object("coming", "car", [0.5] * 3, bbox=(-0.4 + 0.5 * t, 0.4, 0.1, 0.1), screen_area=0, depth=5, velocity=(0.5, 0.0))
        b.add_object("leaving", "car", [0.5] * 3, bbox=(1.1 + 0.5 * t, 0.4, 0.1, 0.1), screen_area=0, depth=5, velocity=(0.5, 0.0))
        flagged |= set(pred.update(b.build()))
    assert flagged == {"coming"}


class _Dummy(Tier):
    id, name = 2, "learned"

    def __init__(self):
        self.calls = 0

    def run(self, pb, P, groups, deadline=None, cache=None, keys=None):
        self.calls += 1
        return P


def _scene(seed=None, n=16):
    """A scene with confusable critical pairs under the first two train observers."""
    obs = train_observers(8, 0)[:2]
    for sd in ([seed] if seed is not None else range(40)):
        snap = next(generate(SynthConfig(n_objects=n, n_frames=1, palette="confusable"), sd))
        scene = Scene.from_snapshot(snap, adj_margin=0.3)
        if (group_stats(Problem(scene, obs, SolverConfig()))[1] > 0).sum() >= 3:
            return snap, scene
    raise AssertionError("no confusable scene")


TB = load_tables()
needs_tables = pytest.mark.skipif(TB is None, reason="run python -m bench.allocator_calibrate")


def test_tier_interface_swappable_and_ids():
    ts = default_tiers()
    assert [t.id for t in ts] == [0, 1, 2, 3, 4] and all(isinstance(t, Tier) for t in ts)
    pb = Problem(_scene()[1], train_observers(6, 0)[:2], SolverConfig())
    P = ts[0].run(pb, pb.lab0.copy(), np.arange(2))
    assert np.allclose(P, pb.lab0)
    P2 = ts[2].run(pb, pb.lab0.copy(), np.flatnonzero(pb.active)[:3])
    assert P2.shape == pb.lab0.shape


@needs_tables
def test_allocator_budget_extremes_and_swap():
    snap, scene = _scene()
    obs = train_observers(8, 0)[:2]
    tiny = Allocator(TB)
    sol0, i0 = tiny.solve(scene, obs, SolverConfig(budget_ms=1500.0), t_start=time.perf_counter() - 1.499)
    assert sum(i0["tier_counts"].values()) == scene.n_groups
    assert i0["tier_counts"]["fine"] == 0 and i0["tier_counts"]["coarse"] == 0
    big = Allocator(TB)
    dummy = _Dummy(); big.tiers[2] = dummy
    sol1, i1 = big.solve(scene, obs, SolverConfig(budget_ms=60000.0))
    assert i1["tier_counts"]["fine"] + i1["tier_counts"]["coarse"] > 0
    assert i1["alloc_ms"] < 0.05 * 60000.0
    assert sol1.colors.shape == scene.colors.shape
    assert not np.allclose(sol1.colors, scene.colors)


@needs_tables
def test_allocator_reuse_cached_tier():
    snap, scene = _scene()
    obs = train_observers(8, 0)[:2]
    al = Allocator(TB)
    al.solve(scene, obs, SolverConfig(budget_ms=60000.0))
    _, info = al.solve(scene, obs, SolverConfig(budget_ms=60000.0))
    assert info["tier_counts"]["reuse"] > 0


@needs_tables
def test_correct_endpoint_reports_tiers(tmp_path):
    snap = _scene()[0]
    out = []
    for budget in (5.0, 5000.0):
        c = TestClient(create_app(gt_dir=tmp_path / f"gt{budget}"))          # fresh signature cache per budget
        s = snap.model_copy(update={"observer_id": "tr-0-000", "budget_ms": budget})
        r = c.post("/correct", content=s.model_dump_json(by_alias=True), headers={"content-type": "application/json"})
        assert r.status_code == 200, r.text
        d = Overrides.model_validate(r.json()).diagnostics
        assert d.solver_tier == "allocated" and sum(d.tier_counts.values()) > 0 and d.allocator_ms >= 0
        out.append(d)
    assert out[0].tier_counts.get("fine", 0) <= out[1].tier_counts.get("fine", 0)
