"""Correction methods on synthetic scenes. Solvers see only TRAIN observers of the held-out observer's
kind; every number is scored under HELD-OUT observers (own CVD model, k, s_L, s_C) on the critical pairs
(confusable + adjacent in the original scene under that observer)."""
from __future__ import annotations
import time
import numpy as np
from bench.registry import experiment
from adapters.synthetic import SynthConfig, generate
from core.observers import train_observers, heldout_observers, KINDS
from core.solver import (Scene, SolverConfig, correct, de_reference, global_daltonisation, local_daltonisation,
                         kuhn2008, none, naturalness, pair_scores)

ALL = ("none", "global_dalt", "local_dalt", "kuhn2008", "cd", "de")


def _run_method(name, scene, kind, tr, scfg, seed, de_kw):
    t = time.perf_counter()
    extra = {}
    if name == "none":
        c = none(scene.colors)
    elif name == "global_dalt":
        c = global_daltonisation(scene.colors, kind)
    elif name == "local_dalt":
        c = local_daltonisation(scene.colors, kind, scene=scene)
    elif name == "kuhn2008":
        c = kuhn2008(scene.colors, kind)
    else:
        sol = correct(scene, tr, scfg, seed) if name == "cd" else de_reference(scene, tr, scfg, seed, **de_kw)
        c = sol.colors
        extra = dict(infeasible_train=sol.n_infeasible, cue_unresolved=len(sol.cue_unresolved), attempts=sol.attempts)
    return c, (time.perf_counter() - t) * 1e3, extra


@experiment("solver_compare")
def run(params, seed):
    cfg = SynthConfig.from_dict(params.get("scene", {}))
    snaps = list(generate(cfg, seed))[:: max(1, int(params.get("stride", 20)))][: int(params.get("n_scenes", 3))]
    scfg = SolverConfig(**params.get("solver", {}))
    methods = params.get("methods", list(ALL))
    de_kw = params.get("de", {})
    n_tr = int(params.get("n_train_per_kind", 3))
    pool = train_observers(60, seed)
    train = {k: [o for o in pool if o.kind == k][:n_tr] for k in KINDS}
    held = heldout_observers(int(params.get("n_heldout", 12)), seed)
    sc_kw = dict(adj_margin=float(params.get("adj_margin", 0.3)), cluster_tol=float(params.get("cluster_tol", 8.0)))
    acc = {m: {} for m in methods}
    add = lambda m, k, v: acc[m].setdefault(k, []).append(float(v))
    for snap in snaps:
        scene = Scene.from_snapshot(snap, **sc_kw)
        out = {m: {k: _run_method(m, scene, k, train[k], scfg, seed, de_kw) for k in KINDS} for m in methods}
        for o in held:
            g = scene.graph(o)
            ed = g.edges[scene.groups[g.edges[:, 0]] != scene.groups[g.edges[:, 1]]] if len(g.edges) else g.edges
            if not len(ed):   # same class+state+colour pairs are one semantic entry: not critical
                continue
            for m in methods:
                c, ms, extra = out[m][o.kind]
                s = pair_scores(c, o, ed, g.tau)
                nat = naturalness(scene.colors, c, scene.protected)
                for k, v in dict(min_de00=s["min"], mean_de00=s["mean"], success_rate=s["success"], infeasible=s["failing"],
                                 deviation=nat["max_de00"], naturalness=nat["mean_de00"], mem_drift=nat["mem_drift"],
                                 solver_ms=ms, **extra).items():
                    add(m, k, v)
    return {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in acc.items() if d}
