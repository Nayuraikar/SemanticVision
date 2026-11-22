import numpy as np
from adapters.synthetic import SynthConfig, generate
from core.observers import train_observers, heldout_observers
from core.perception import color as C
from core.solver import (Scene, SolverConfig, Problem, correct, de_reference, global_daltonisation, local_daltonisation,
                         kuhn2008, none, assign_cues, memory_mask, naturalness, screen_check)
from bench import runner

CFG = SynthConfig(n_objects=10, n_frames=1, palette="confusable")


def scene(seed=0):
    return Scene.from_snapshot(next(iter(generate(CFG, seed))))


def tr(kind="deutan", n=3, seed=0):
    return [o for o in train_observers(60, seed) if o.kind == kind][:n]


def test_baselines_shape_and_gamut():
    s = scene()
    assert np.allclose(none(s.colors), s.colors)
    for f in (global_daltonisation, kuhn2008):
        out = f(s.colors, "deutan")
        assert out.shape == s.colors.shape and out.min() >= 0 and out.max() <= 1
    out = local_daltonisation(s.colors, "protan", scene=s)
    assert out.shape == s.colors.shape and out.min() >= 0
    assert not np.allclose(global_daltonisation(s.colors, "deutan"), s.colors)


def test_projection_constraints():
    s = scene()
    cfg = SolverConfig(delta=10.0, delta_mem=2.0)
    pb = Problem(s, tr(), cfg)
    rng = np.random.default_rng(0)
    P = pb.lab0[None] + rng.normal(0, 60, (50, pb.E, 3))
    X = pb.project(P)
    assert (pb.deviation(X) <= pb.delta + 0.3).all()
    assert (X[..., 0] >= cfg.L_range[0] - 1e-6).all() and (X[..., 0] <= cfg.L_range[1] + 1e-6).all()
    rgb = pb._rgb(X)
    assert rgb.min() > -2e-3 and rgb.max() < 1 + 2e-3


def test_memory_colour_protected_and_consistency():
    s = scene(1)
    s.protected[:] = False
    s.protected[s.groups == 0] = True
    sol = correct(s, tr(), SolverConfig(delta=15.0, delta_mem=3.0))
    d = C.ciede2000(C.linear_to_lab(s.colors), C.linear_to_lab(sol.colors))
    assert d[s.protected].max() <= 3.5
    for g in range(s.n_groups):          # one palette entry per semantic group
        assert np.allclose(sol.colors[s.groups == g], sol.colors[s.groups == g][0])
    assert (d <= 15.5).all()
    assert memory_mask(np.array([[0.2, 0.3, 0.8]]), classes=["sky/default"]).all()


def test_solver_uses_observer_thresholds():
    s = scene()
    o = tr(n=1)[0]
    from dataclasses import replace
    lax = replace(o, k=3.0)
    t1, t2 = Problem(s, [o]).tau, Problem(s, [lax]).tau
    assert np.allclose(t2, 3.0 * t1)


def test_solver_improves_objective_and_verifies():
    s = scene(2)
    obs = tr()
    sol = correct(s, obs, SolverConfig(budget_ms=3000))
    pb = Problem(s, obs, SolverConfig())
    assert sol.objective >= float(pb.objective(pb.lab0[None])[0]) - 1e-9
    gp, _ = screen_check(s, sol.colors, obs)
    assert gp == sol.infeasible and sol.attempts >= 1


def test_de_reference_not_worse_much():
    s = scene(3)
    obs = tr()
    a = correct(s, obs, SolverConfig())
    b = de_reference(s, obs, SolverConfig(), seed=0, maxiter=40, popsize=8)
    assert b.objective >= a.objective - 1.0


def test_cue_csp():
    pairs = [(0, 1), (1, 2), (0, 2), (2, 3)]
    cues, bad = assign_cues(pairs)
    assert not bad
    for a, b in pairs:
        assert cues.get(a) != cues.get(b) and (a in cues or b in cues)
    cues, bad = assign_cues([(0, 1), (1, 2), (0, 2)], [("hatch", "x"), ("hatch", "y")])   # triangle needs 3 > 2 cues... cover is 2 groups
    assert not bad
    cues, bad = assign_cues([(i, j) for i in range(4) for j in range(i + 1, 4)], [("a", 1), ("a", 2)])
    assert len(bad) > 0   # K4 cover = 3 vertices, only 2 cues -> clash reported


def test_naturalness_zero_and_positive():
    s = scene()
    assert naturalness(s.colors, s.colors)["mean_de00"] == 0
    assert naturalness(s.colors, global_daltonisation(s.colors, "deutan"))["mean_de00"] > 0


def test_solver_beats_global_daltonisation_heldout():
    """Exit criterion: higher min dE00 of critical pairs than global daltonisation, at bounded deviation,
    scored on held-out observers (solver sees train observers only). Small paired run."""
    cfg = runner.load_config("solver_compare")
    cfg["params"]["methods"] = ["global_dalt", "cd"]
    cfg["params"]["n_scenes"] = 2
    r = runner.run(cfg, seed=0, n_seeds=8, out_dir=None or __import__("tempfile").mkdtemp())
    p = {x["metric"]: x for x in r["paired"] if x["arm"] == "cd"}
    s = {(x["arm"], x["metric"]): x["mean"] for x in r["summary"]}
    assert p["min_de00"]["mean_diff"] > 0
    assert s[("cd", "success_rate")] > s[("global_dalt", "success_rate")]
    assert s[("cd", "deviation")] <= cfg["params"]["solver"]["delta"] + 0.5
    assert s[("cd", "deviation")] < s[("global_dalt", "deviation")]
