import numpy as np
import pytest
from fastapi.testclient import TestClient

from adapters.synthetic import SynthConfig, generate
from bench.experiments import profile_error as pe
from calibration import Calibrator
from core.observers import heldout_observers
from core.profiles import samples_to_observers, posterior_mean_observer
from core.server import create_app
from core.solver import Scene, SolverConfig, Problem, correct
from protocol.models import Overrides

CFG = SynthConfig(n_objects=8, n_frames=1, palette="confusable")
FAST = dict(grid=3, sweeps=1, max_iter=10, retries=1, budget_ms=100000)


def scene(seed=0):
    return Scene.from_snapshot(next(iter(generate(CFG, seed))))


def post(level="wide_correct", seed=0, n=32):
    o = heldout_observers(2, seed)[0]
    return o, pe.make_posterior(o, level, np.random.default_rng(seed), n)


def test_risk_measures_order_and_degenerate_cases():
    s = scene()
    _, p = post()
    obs = samples_to_observers(p, 8)
    X = Problem(s, obs[:1], SolverConfig()).lab0[None]
    vals = {}
    for r in ("worst", "cvar", "softmin"):
        pb = Problem(s, obs, SolverConfig(risk=r, alpha=0.25))
        vals[r] = float(pb.objective(X)[0])
    assert vals["worst"] <= vals["cvar"] + 1e-9
    one = obs[:1]
    a = float(Problem(s, one, SolverConfig(risk="worst")).objective(X)[0])
    b = float(Problem(s, one, SolverConfig(risk="cvar")).objective(X)[0])
    c = float(Problem(s, one, SolverConfig(risk="softmin")).objective(X)[0])
    assert a == pytest.approx(b) and a == pytest.approx(c)
    pb = Problem(s, obs, SolverConfig(risk="cvar", alpha=1.0))      # alpha=1 -> mean over observers
    assert np.isfinite(pb.objective(X)[0])


def test_samples_to_observers_from_real_calibrator():
    cal = Calibrator(seed=1, n_particles=300)
    smp = cal.samples(64)
    obs = samples_to_observers(smp, 6)
    assert len(obs) == 6 and all(o.kind in ("protan", "deutan", "tritan") and 0 <= o.severity <= 1 and o.k > 0 for o in obs)
    mean = posterior_mean_observer(smp)
    assert mean.kind in ("protan", "deutan", "tritan")
    assert samples_to_observers(smp, 6) == obs                       # deterministic


def test_robust_solver_runs_and_respects_constraints():
    s = scene()
    _, p = post()
    obs = samples_to_observers(p, 6)
    for r in ("cvar", "worst"):
        sol = correct(s, obs, SolverConfig(risk=r, **FAST), seed=0)
        assert sol.colors.shape == s.colors.shape and sol.colors.min() >= 0 and sol.colors.max() <= 1
        assert sol.nat["max_de00"] <= 15.5


def test_profile_error_bench_fixed_seed():
    """Honest K2 check (fixed seed, small): on this synthetic setup the point estimate does NOT degrade under profile error
    (so robust has nothing to recover); robust is deterministic and pays a bounded cost when the profile is correct."""
    prm = dict(scene=dict(n_objects=12, n_frames=60, palette="confusable"), stride=20, n_scenes=1, n_true=4, m=6, n_samples=32,
               solver=dict(delta=6.0, **FAST), levels=["correct", "wrong_type"], methods=["point", "cvar"])
    r1, r2 = pe.run(prm, 0), pe.run(prm, 0)
    assert r1 == r2
    pt, cv = r1["correct|point"], r1["correct|cvar"]
    assert pt["success_rate"] >= 0.9
    assert pt["min_de00"] - cv["min_de00"] < 3.0                      # control: bounded cost of robustness
    assert r1["correct|point"]["min_de00"] - r1["wrong_type|point"]["min_de00"] < 2.0   # point barely hurt by profile error


def test_server_robust_flag_profile(tmp_path):
    import time
    o, p = post("wide_correct", 0, 64)
    snap = next(iter(generate(SynthConfig(n_objects=8, n_frames=1, palette="confusable"), 0)))
    res = {}
    for mode in ("point", "cvar", "worst"):
        c = TestClient(create_app(gt_dir=tmp_path / "gt", profiles_db=tmp_path / f"{mode}.db", robust=mode))
        store = c.app.state.profiles
        pid = store.create({}, {}, {})
        mean = posterior_mean_observer(p)
        summ = {"params": {k: {"mean": float(np.mean(p[k]))} for k in ("e_protan", "e_deutan", "e_tritan", "s_L", "s_C")}}
        summ["params"]["k"] = {"mean": mean.k / (0.5 * (mean.s_L + mean.s_C))}
        store.finish(pid, summ, p)
        s2 = snap.model_copy(update={"observer_id": f"profile:{pid}", "budget_ms": 2000})
        t = time.perf_counter()
        r = c.post("/correct", content=s2.model_dump_json(by_alias=True), headers={"content-type": "application/json"})
        dt = (time.perf_counter() - t) * 1e3
        assert r.status_code == 200, r.text
        ov = Overrides.model_validate(r.json())
        assert ov.diagnostics.solver_tier == "exact" and dt < 2000 * 1.5
        res[mode] = [tuple(x.albedo) for x in ov.palette_overrides]
    assert res["cvar"] != res["point"] or res["worst"] != res["point"]    # flag changes the solve
    with pytest.raises(ValueError):
        create_app(robust="bogus")
