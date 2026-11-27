"""Dataset generator (SPEC 5.6): EXACT-solver outputs (core/solver coordinate descent; DE reference on a subset) on seeded
synthetic scenes with lit colours (core/lighting via Scene.from_snapshot), one observer per sample.
Splits:  train/val     Machado observers (+mild threshold jitter), train lightings, scene seeds 1e5/2e5
         test_obs      held-out observers (Brettel/Vienot, severity [.1,.5), threshold noise; core/observers), train-dist scenes
         test_scene    new Machado observers on held-out scenes (held-out lighting, larger N, disjoint seeds)
         test_both     held-out observers on held-out scenes
Only `val` may be used for any tuning. Exact targets for held-out observers are solved WITH that observer (oracle reference).
python -m learning.data [--scale small|kaggle] [--procs 8]   (cached in learning/cache/<scale>.pkl)"""
from __future__ import annotations
import argparse, hashlib, json, os, pickle, time
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path
import numpy as np
from adapters.synthetic import SynthConfig, generate
from core.observers import train_observers, heldout_observers
from core.perception import cvd as V
from core.solver import Scene, SolverConfig, correct, de_reference
from core.learned import embed_observer, scene_inputs, cat_rgb_matrix

CACHE = Path(__file__).resolve().parent / "cache"
TRAIN_LIGHTS = ("day", "rain", "night")
HELD_LIGHTS = ("fog",)
PALETTES = ("random", "confusable", "traffic", "dashboard")
JITTER = 0.15                    # train-family threshold jitter: k,s_L,s_C ~ exp(U(-J,J)) (held-out: lognormal sigma .25)
SPLITS = {
    "train": dict(obs="train", lights=TRAIN_LIGHTS, n=(6, 32), base=100_000, obs_seed=11),
    "val": dict(obs="train", lights=TRAIN_LIGHTS, n=(6, 32), base=200_000, obs_seed=12),
    "test_obs": dict(obs="heldout", lights=TRAIN_LIGHTS, n=(6, 32), base=300_000, obs_seed=13),
    "test_scene": dict(obs="train", lights=HELD_LIGHTS, n=(8, 40), base=400_000, obs_seed=14),
    "test_both": dict(obs="heldout", lights=HELD_LIGHTS, n=(8, 40), base=500_000, obs_seed=15),
}
SCALES = {   # samples per split, DE-reference samples per test split, exact-solver config
    "test": dict(counts=dict(train=4, val=2, test_obs=0, test_scene=0, test_both=0), n_de=0, budget_ms=150.0, retries=0,
                 de_maxiter=5, de_pop=4),
    "tiny": dict(counts=dict(train=24, val=8, test_obs=8, test_scene=8, test_both=8), n_de=2, budget_ms=600.0, retries=0,
                 de_maxiter=10, de_pop=5),
    "small": dict(counts=dict(train=800, val=80, test_obs=80, test_scene=80, test_both=80), n_de=12, budget_ms=1000.0,
                  retries=1, de_maxiter=30, de_pop=8),
    "kaggle": dict(counts=dict(train=8000, val=400, test_obs=400, test_scene=400, test_both=400), n_de=60,
                   budget_ms=2000.0, retries=2, de_maxiter=80, de_pop=10),   # ~1.5-2 h on 4 Kaggle CPU cores
}
SOLVER = dict(delta=15.0, delta_mem=3.0)


def observer_pool(split, n=64):
    sp = SPLITS[split]
    if sp["obs"] == "heldout":
        return heldout_observers(n, sp["obs_seed"])
    r = np.random.default_rng([sp["obs_seed"], 0x717])
    out = []
    for o in train_observers(n, sp["obs_seed"]):
        k, sl, sc = np.exp(r.uniform(-JITTER, JITTER, 3))
        out.append(replace(o, id=f"L{o.id}", k=float(k), s_L=float(sl), s_C=float(sc)))
    return out


def loss_pairs(scene, obs):
    """Object pairs for the training loss under the TRUE observer: adjacent, different groups, weight > 0.
    -> pairs (P,2), tau (P,), w (P,) (core Problem convention: w / max, floored at .2)."""
    g = scene.graph(obs)
    diff = scene.groups[:, None] != scene.groups[None, :]
    iu = np.triu((g.weight > 1e-9) & diff, 1)
    pr = np.argwhere(iu)
    if not len(pr):
        return np.zeros((0, 2), int), np.zeros(0), np.zeros(0)
    w = g.weight[iu]
    return pr, g.tau[iu], np.maximum(w / w.max(), 0.2)


def make_scene(split, i, n_try=30):
    sp = SPLITS[split]
    pool = observer_pool(split)
    for t in range(n_try):
        seed = sp["base"] + i * n_try + t
        r = np.random.default_rng([seed, 0x5CE])
        obs = pool[int(r.integers(len(pool)))]
        cfg = SynthConfig(n_objects=int(r.integers(sp["n"][0], sp["n"][1] + 1)), n_frames=1,
                          palette=PALETTES[int(r.integers(len(PALETTES)))],
                          lighting=sp["lights"][int(r.integers(len(sp["lights"])))])
        scene = Scene.from_snapshot(next(generate(cfg, seed)))
        if scene.n_groups < 2:
            continue
        g = scene.graph(obs)
        if any(scene.groups[a] != scene.groups[b] for a, b in g.edges):     # >=1 confusable cross-group pair
            return scene, obs, seed, cfg
    raise RuntimeError(f"no confusable scene for {split}/{i}")


def sim_matrix(obs):
    return V.machado_matrix(obs.kind, obs.severity) if obs.model == "machado" else None


def gen_sample(args):
    split, i, scale = args
    sc = SCALES[scale]
    scene, obs, seed, cfg = make_scene(split, i)
    scfg = SolverConfig(budget_ms=sc["budget_ms"], retries=sc["retries"], **SOLVER)
    z = embed_observer(obs)
    inp = scene_inputs(scene, z, **SOLVER)
    sol = correct(scene, [obs], scfg, seed=seed)
    target = sol.palette_lab[scene.groups].astype(np.float32)
    de, de_ms = None, None
    if split.startswith("test") and i < sc["n_de"]:
        t0 = time.perf_counter()
        d = de_reference(scene, [obs], scfg, seed=seed, maxiter=sc["de_maxiter"], popsize=sc["de_pop"])
        de, de_ms = d.palette_lab[scene.groups].astype(np.float32), (time.perf_counter() - t0) * 1e3
    pr, tau, w = loss_pairs(scene, obs)
    return dict(split=split, i=i, seed=seed, n=len(scene.colors), palette=cfg.palette, lighting=cfg.lighting,
                scene=scene, obs=obs, inp=inp, target=target, exact_ms=sol.time_ms, de=de, de_ms=de_ms,
                pairs=pr, ptau=tau.astype(np.float32), pw=w.astype(np.float32),
                M_cat=cat_rgb_matrix(scene.illum_xyz).astype(np.float32),
                M_cvd=None if sim_matrix(obs) is None else sim_matrix(obs).astype(np.float32))


def cache_key(scale, splits):
    sc = SCALES[scale]
    blob = json.dumps(dict(sc=sc, splits={k: SPLITS[k] for k in splits}, jit=JITTER, pal=PALETTES, solver=SOLVER, v=3),
                      sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def build(scale="small", splits=tuple(SPLITS), procs=None, cache=True, verbose=True) -> dict:
    """-> {split: [sample dict]}; deterministic given scale (seeded scenes/observers/solver)."""
    path = CACHE / f"{scale}_{cache_key(scale, splits)}.pkl"
    if cache and path.exists():
        return pickle.loads(path.read_bytes())
    jobs = [(s, i, scale) for s in splits for i in range(SCALES[scale]["counts"][s])]
    jobs.sort(key=lambda j: (not j[0].startswith("test"), j[1]))       # DE-heavy jobs first
    t0 = time.time()
    procs = procs or min(8, os.cpu_count() or 1)
    if procs > 1:
        with get_context("fork").Pool(procs) as p:
            rows = p.map(gen_sample, jobs, chunksize=1)
    else:
        rows = [gen_sample(j) for j in jobs]
    out = {s: sorted([r for r in rows if r["split"] == s], key=lambda r: r["i"]) for s in splits}
    if verbose:
        print(f"dataset {scale}: {len(rows)} samples in {time.time() - t0:.0f}s", flush=True)
    if cache:
        CACHE.mkdir(exist_ok=True)
        path.write_bytes(pickle.dumps(out))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m learning.data")
    ap.add_argument("--scale", default="small", choices=list(SCALES))
    ap.add_argument("--procs", type=int, default=None)
    a = ap.parse_args(argv)
    d = build(a.scale, procs=a.procs)
    for s, rs in d.items():
        print(s, len(rs), "exact ms mean %.0f" % np.mean([r["exact_ms"] for r in rs]),
              "DE", sum(r["de"] is not None for r in rs))


if __name__ == "__main__":
    main()
