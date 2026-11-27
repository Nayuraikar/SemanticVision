"""Learned solver (SPEC 5.6): differentiable perception vs core, projection feasibility, shared palette, ONNX parity,
verified fallback, embedding, determinism, core stays torch-free."""
import ast
from pathlib import Path
import numpy as np
import pytest
import torch
from core.observers import train_observers, heldout_observers, Observer
from core.perception import color as C, cvd as V, distance as Dst
from core.learned import embed_observer, embed_profile, embed_samples, scene_inputs, F_DIM, Z_DIM, cat_rgb_matrix
from core.solver import SolverConfig
from core.solver.problem import Problem
from learning import torch_percept as TP
from learning.model import LearnedSolver, NetConfig, project, L_RANGE
from learning.export import export_onnx, verify_onnx, feeds, INPUTS
from tests.test_perception import SHARMA

ROOT = Path(__file__).resolve().parents[1]
T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)


def rand_lab(n, seed):
    r = np.random.default_rng(seed)
    rgb = r.uniform(0.02, 0.98, (n, 3))
    return C.linear_to_lab(rgb)


def test_ciede2000_matches_core():
    a, b = rand_lab(4000, 0), rand_lab(4000, 1)
    b[:500] = a[:500] + np.random.default_rng(2).normal(0, 1.0, (500, 3))      # near pairs
    b[500:600, 1:] = 0; a[600:700, 1:] = 0                                       # achromatic edge cases
    ref = C.ciede2000(a, b)
    got = TP.ciede2000(T(a), T(b)).numpy()
    assert np.abs(got - ref).max() < 2e-3
    for x, y, d in SHARMA:
        assert abs(float(TP.ciede2000(T(x), T(y))) - d) < 2e-3


def test_ciede2000_gradients_finite():
    a = T(rand_lab(64, 3)).requires_grad_(True)
    b = a.detach().clone(); b[:8] = a.detach()[:8]                               # identical pairs (sqrt(0) case)
    b[8:16, 1:] = 0
    TP.ciede2000(a, b).sum().backward()
    assert torch.isfinite(a.grad).all()


@pytest.mark.parametrize("model,kind,sev", [("machado", "protan", 0.7), ("machado", "tritan", 0.3), ("vienot", "deutan", 0.4),
                                            ("brettel", "protan", 1.0), ("brettel", "tritan", 0.25)])
def test_observer_matches_core(model, kind, sev):
    rgb = np.random.default_rng(4).uniform(0, 1, (2000, 3))
    ref = V.simulate(rgb, model, kind, sev)
    got = TP.simulate(T(rgb), model, kind, sev).numpy()
    assert np.abs(got - ref).max() < 1e-4


def test_observer_lab_with_cat_matches_core():
    lab = rand_lab(500, 5)
    illum = np.array([0.98, 1.0, 0.75])
    rgb = np.clip(C.xyz_to_linear(C.lab_to_xyz(lab)), 0, 1)
    ref = C.linear_to_lab(Dst.perceived(rgb, model="machado", kind="deutan", severity=0.8, illum_xyz=illum))
    got = TP.observer_lab(T(lab), T(cat_rgb_matrix(illum)), T(V.machado_matrix("deutan", 0.8))).numpy()
    assert np.abs(got - ref).max() < 2e-2
    assert np.abs(TP.lab_to_linear(TP.linear_to_lab(T(rgb))).numpy() - rgb).max() < 1e-4


@pytest.mark.parametrize("seed", range(4))
def test_projection_satisfies_constraints(seed):
    r = np.random.default_rng(seed)
    n = 3000
    lab0 = rand_lab(n, 100 + seed)
    lab0 = lab0[(lab0[:, 0] > L_RANGE[0]) & (lab0[:, 0] < L_RANGE[1])]
    n = len(lab0)
    delta = np.where(r.uniform(size=n) < 0.2, 3.0, 15.0)                         # memory colours: tight bound
    X = lab0 + r.normal(0, 1, (n, 3)) * r.choice([2.0, 20.0, 80.0], (n, 1))
    out = project(T(X), T(lab0), T(delta)).numpy().astype(float)
    dev = C.ciede2000(out, lab0)
    rgb = C.xyz_to_linear(C.lab_to_xyz(out))
    assert (dev <= delta + 1e-3).all()
    assert (rgb >= -2e-3).all() and (rgb <= 1 + 2e-3).all()
    assert (out[:, 0] >= L_RANGE[0] - 1e-3).all() and (out[:, 0] <= L_RANGE[1] + 1e-3).all()
    feas = (C.ciede2000(X, lab0) <= delta - 1e-3) & (np.abs(C.xyz_to_linear(C.lab_to_xyz(X)) - 0.5) <= 0.5 - 1e-3).all(-1) \
        & (X[:, 0] > L_RANGE[0]) & (X[:, 0] < L_RANGE[1])
    assert np.abs(out[feas] - X[feas]).max() < 1e-3                              # feasible inputs are untouched


def _scene(seed=0, n=14):
    from adapters.synthetic import SynthConfig, generate
    from core.solver import Scene
    obs = train_observers(30, 0)
    for s in range(seed, seed + 50):
        sc = Scene.from_snapshot(next(generate(SynthConfig(n_objects=n, n_frames=1, palette="confusable"), s)))
        for o in obs:
            g = sc.graph(o)
            if any(sc.groups[a] != sc.groups[b] for a, b in g.edges) and sc.n_groups < len(sc.colors):
                return sc, o
    raise AssertionError


def _inputs(net, inp):
    f = feeds(inp)
    return [torch.as_tensor(f[k]) for k in INPUTS]


def test_features_and_shared_palette():
    sc, o = _scene()
    inp = scene_inputs(sc, embed_observer(o))
    assert inp["x"].shape == (len(sc.colors), F_DIM) and inp["z"].shape == (Z_DIM,) and np.isfinite(inp["x"]).all()
    torch.manual_seed(0)
    net = LearnedSolver(NetConfig(arch="deepsets", hidden=32, layers=2))
    torch.nn.init.normal_(net.head[-1].weight, std=0.5)
    with torch.no_grad():
        out = net(*_inputs(net, inp))[0][0].numpy()
    for g in np.unique(sc.groups):
        m = sc.groups == g
        assert np.abs(out[m] - out[m][0]).max() < 1e-4                            # same class/state cluster -> one colour
    assert np.abs(out - inp["labg"]).max() > 0.5


@pytest.mark.parametrize("arch", ["deepsets", "gnn", "mlp"])
def test_onnx_matches_torch(tmp_path, arch):
    torch.manual_seed(1)
    net = LearnedSolver(NetConfig(arch=arch, hidden=32, layers=2))
    torch.nn.init.normal_(net.head[-1].weight, std=0.3)
    p = export_onnx(net, tmp_path / f"{arch}.onnx")
    samples = [scene_inputs(*_scene(s, n)[:1], embed_observer(train_observers(4, s)[0])) for s, n in ((0, 6), (3, 14), (9, 31))]
    r = verify_onnx(net, p, samples)
    assert r["ok"] and r["max_abs"] < 1e-2 and len(set(r["sizes"])) == 3, r


def test_fallback_triggers_on_bad_prediction(tmp_path):
    from core.allocator.onnx_tier import OnnxTier
    torch.manual_seed(0)
    net = LearnedSolver(NetConfig(hidden=16, layers=1))           # zero-init head => predicts "no correction" (bad)
    p = export_onnx(net, tmp_path / "bad.onnx")
    sc, o = _scene()
    pb = Problem(sc, [o], SolverConfig())
    groups = np.flatnonzero(pb.active)
    tier = OnnxTier(p, repair_iters=0)                            # no local repair: verification failure -> exact
    P = tier.run(pb, pb.lab0.copy(), groups)
    assert tier.calls == 1 and tier.fallbacks == 1 and tier.last["fallback"] and tier.last["reason"] in ("screen", "objective")
    assert pb.objective(P[None])[0] > pb.objective(pb.lab0[None])[0]               # exact fallback improved it
    tier.predict = lambda pb_: np.full((pb_.E, 3), np.nan)                          # garbage output
    P2 = tier.run(pb, pb.lab0.copy(), groups)
    assert tier.fallbacks == 2 and tier.last["reason"] == "nonfinite" and np.isfinite(P2).all()
    assert tier.fallback_rate == 1.0
    good = OnnxTier(p)
    good.predict = lambda pb_: P                                                    # a verified-good prediction passes
    good.check = lambda *a: (True, "ok", groups[:0])
    assert np.allclose(good.run(pb, pb.lab0.copy(), groups)[groups], P[groups]) and good.fallbacks == 0


def test_embedding_profile_and_observers():
    o = Observer("x", "train", "machado", "deutan", 0.6, k=1.2, s_L=0.9, s_C=1.1)
    z = embed_observer(o)
    assert np.allclose(z[:3], [0, 1, 0]) and np.isclose(z[3], 0.6) and np.isclose(z[4], np.log(1.2))
    h = heldout_observers(3, 0)[0]
    assert np.isclose(embed_observer(h)[3], h.severity)
    summ = {"params": {"e_protan": {"mean": 0.1}, "e_deutan": {"mean": 0.5}, "e_tritan": {"mean": 0.0},
                       "k": {"mean": 1.0}, "s_L": {"mean": 0.8}, "s_C": {"mean": 1.2}, "bg_gain": {"mean": 0.4}}}
    zp = embed_profile(summ)
    assert np.allclose(zp[:3], [1 / 6, 5 / 6, 0]) and np.isclose(zp[3], 0.5) and np.isclose(zp[7], 0.4)
    samples = {"e_protan": [0.1, 0.1], "e_deutan": [0.4, 0.6], "e_tritan": [0, 0], "k": [1, 1], "s_L": [0.8, 0.8],
               "s_C": [1.2, 1.2]}
    assert np.allclose(embed_samples(samples)[:7], zp[:7])


def test_fixed_seed_determinism():
    from learning import data as D
    from learning.train import train, TrainConfig
    d = D.build("test", splits=("train", "val"), procs=1, cache=False, verbose=False)
    d2 = D.build("test", splits=("train", "val"), procs=1, cache=False, verbose=False)
    assert all(np.array_equal(a["inp"]["x"], b["inp"]["x"]) for a, b in zip(d["train"], d2["train"]))
    tc = TrainConfig(epochs=2, batch=2, net=NetConfig(hidden=16, layers=1))
    n1, h1 = train(d, tc, "cpu", verbose=False)
    n2, h2 = train(d, tc, "cpu", verbose=False)
    for (k, a), (_, b) in zip(n1.state_dict().items(), n2.state_dict().items()):
        assert torch.equal(a, b), k
    assert h1["history"][-1]["train_total"] == h2["history"][-1]["train_total"]


def test_learned_runtime_core_is_torch_free():
    for f in list((ROOT / "core" / "learned").rglob("*.py")) + list((ROOT / "core" / "allocator").rglob("*.py")):
        tree = ast.parse(f.read_text())
        mods = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        mods |= {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        assert not mods & {"torch", "learning", "three", "unity"}, f
