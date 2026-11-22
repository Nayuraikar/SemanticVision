import time
import numpy as np
from adapters.synthetic import WorldConfig, generate_world
from core.observers import train_observers
from core.perception.graph import build_graph
from core.solver import Scene, SolverConfig, correct
from core.server import solve, observer_table, signature


def test_graph_sensitivity_per_observer_not_min_over_nodes():
    cols = np.array([[0.6, 0.2, 0.2], [0.55, 0.25, 0.2], [0.1, 0.1, 0.8]])
    kw = dict(size_deg=[2.0] * 3, adjacency=np.ones((3, 3), bool), k=1.0)
    scalar = build_graph(cols, s_L=2.0, s_C=2.0, **kw)
    arr = build_graph(cols, s_L=[2.0, 2.0, 0.1], s_C=[2.0, 2.0, 0.1], **kw)
    # pair (0,1): both nodes are sensitive -> same tau as the scalar observer, unaffected by the insensitive node 2
    assert np.isclose(arr.tau[0, 1], scalar.tau[0, 1])
    assert arr.tau[0, 2] > scalar.tau[0, 2] > 0 and np.isclose(arr.tau[0, 2], arr.tau[2, 0])
    lo = build_graph(cols, s_L=0.5, s_C=0.5, **kw)
    assert lo.tau[0, 1] > scalar.tau[0, 1]


def test_solver_budget_enforced_in_loops():
    tr = [o for o in train_observers(30, 0) if o.kind == "deutan"][:3]
    snap, _, _ = generate_world(WorldConfig(n_objects=30, palette="mixed", lighting="night"), 2)
    sc = Scene.from_snapshot(snap)
    t = {}
    for b in (200, 5000):
        t0 = time.perf_counter()
        sol = correct(sc, tr, SolverConfig(budget_ms=b), 0)
        t[b] = (time.perf_counter() - t0) * 1e3
        assert abs(sol.time_ms - t[b]) < 5
    assert t[200] < 200 + 40                     # fixed overhead only (problem build), loops stop at the deadline
    assert t[200] < 0.5 * t[5000]                # unbudgeted run is much longer, i.e. the cut-off is real
    sol0 = correct(sc, tr, SolverConfig(budget_ms=1.0), 0)     # already late on entry: valid identity-ish palette
    assert sol0.colors.shape == sc.colors.shape and np.isfinite(sol0.colors).all()


def test_scene_uses_lit_colours_and_cat16():
    snap, truth, _ = generate_world(WorldConfig(n_objects=10, lighting="night", palette="mixed"), 1)
    lit, alb = Scene.from_snapshot(snap), Scene.from_snapshot(snap, lighting="albedo")
    assert lit.illum_xyz is not None and alb.illum_xyz is None
    refl = np.array([max(o.emissive_rgb) == 0 for o in snap.objects])
    assert lit.colors[refl].mean() < 0.5 * alb.colors[refl].mean()   # night: reflective objects far darker than albedo
    assert np.abs(lit.colors - np.array([truth["lit"][i] for i in lit.ids])).mean() < 0.05
    assert set(lit.shadow) <= {"lit", "penumbra", "shadow"}
    # round trip: uncorrected colours map back to the authored albedo / emission
    a, e = lit.to_overrides(lit.colors)
    for i, o in enumerate(snap.objects):
        if max(o.emissive_rgb) == 0 and lit.colors[i].max() < 0.99:
            assert np.allclose(a[i], o.albedo_rgb, atol=2e-3) and np.allclose(e[i], 0)
        elif max(o.emissive_rgb) > 0 and lit.emis[i].sum() > lit.refl[i].sum():
            assert np.allclose(a[i], o.albedo_rgb, atol=2e-3)          # emissive-dominated keep their albedo


def test_correct_endpoint_emissive_changes_emission_and_cache_key():
    snap, _, _ = generate_world(WorldConfig(n_objects=12, lighting="tinted", palette="signals"), 0)
    table = observer_table()
    snap = snap.model_copy(update={"observer_id": "tr-0-000", "budget_ms": 2000.0})
    ov = solve(snap, table["tr-0-000"], table)
    by = {o.id: o for o in snap.objects}
    changed = 0
    for p in ov.palette_overrides:
        o = by[p.id]
        if o.cls == "light":
            assert np.allclose(p.albedo, o.albedo_rgb, atol=2e-3)       # emissive-dominated: albedo untouched
            changed += not np.allclose(p.emissive, o.emissive_rgb, atol=1e-3)
    assert changed >= 1
    s2 = snap.model_copy(update={"objects": [o.model_copy(update={"shadow_vis": 0.0}) for o in snap.objects]})
    assert signature(snap) != signature(s2)
