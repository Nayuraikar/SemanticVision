"""Real-solver quality at EQUAL budget. Per scene and CVD kind: the allocator plans tiers for the palette groups with each method
(estimates from the measured tables, importance from the rule/MLP scorer), core/solver tiers run only those groups under the
budget; the result is SCORED under HELD-OUT observers (own CVD model + threshold noise) on their critical pairs, weighted by
the INDEPENDENT task observer's per-object value (never the scorer used to allocate). Solver sees train observers only.
Budget = frac x (table-estimated cost of fine-tier for every active group). Arms: <method>|f<frac>, plus none and full."""
from __future__ import annotations
import time
import numpy as np
from bench.registry import experiment
from adapters.synthetic import SynthConfig, generate
from core.allocator import Allocator, load_tables
from core.importance import Tracker, load_scorer, geometry, task_labels
from core.observers import train_observers, heldout_observers
from core.perception import distance as Dst
from core.solver import Scene, SolverConfig, naturalness
from core.solver.problem import Problem

METHODS = ("uniform", "round_robin", "greedy", "dp", "lagrangian")


def _prep(scene, o, wlab):
    g = scene.graph(o)
    ed = g.edges[scene.groups[g.edges[:, 0]] != scene.groups[g.edges[:, 1]]] if len(g.edges) else g.edges
    if not len(ed):
        return None
    w = 0.5 * (wlab[ed[:, 0]] + wlab[ed[:, 1]]) + 1e-3
    return o, ed, g.tau[ed[:, 0], ed[:, 1]], w


def _score(scene, colors, prep):
    o, ed, tau, w = prep
    kw = dict(o.sim_kwargs(), **({"illum_xyz": scene.illum_xyz} if scene.illum_xyz is not None else {}))
    d = Dst.pair_matrix(np.asarray(colors, float), **kw)[ed[:, 0], ed[:, 1]]
    return float((w * (d >= tau)).sum() / w.sum()), float(d.min())


@experiment("allocator_quality")
def run(params, seed):
    tb = load_tables()
    scorer = load_scorer()
    cfg = SolverConfig()
    fracs = params.get("fracs", [0.15, 0.3, 0.6])
    kinds = params.get("kinds", ["protan", "deutan"])
    pool = train_observers(60, seed)
    held = heldout_observers(int(params.get("n_heldout", 12)), seed)
    acc: dict = {}

    def add(arm, **kv):
        for k, v in kv.items():
            acc.setdefault(arm, {}).setdefault(k, []).append(float(v))

    for si in range(int(params.get("n_scenes", 2))):
        snap = next(generate(SynthConfig(n_objects=int(params.get("n_objects", 50)), n_frames=1, palette="confusable",
                                         motion=0.25), seed * 100 + si))
        scene = Scene.from_snapshot(snap, adj_margin=0.3)
        X, geo = Tracker().update(snap)
        scene.importance = 0.5 + 1.5 * np.asarray(scorer(X), float)            # allocator's view (rules or MLP)
        wlab = task_labels(snap, geometry(snap))                              # scoring truth (independent observer)
        for kind in kinds:
            tr = [o for o in pool if o.kind == kind][:3]
            ho = [p for p in (_prep(scene, o, wlab) for o in held if o.kind == kind) if p]
            if not ho:
                continue
            pb = Problem(scene, tr, cfg)
            al = Allocator(tb)
            probe = al.plan(pb, 1e9, method="uniform")
            full = float(probe.cost[np.arange(pb.E), 4].sum())
            if full <= 0:
                continue
            runs = {"none": (pb.lab0.copy(), 0.0, 0.0, 0.0)}
            t0 = time.perf_counter()
            Pf = al.tiers[4].run(pb, pb.lab0.copy(), np.flatnonzero(pb.active))
            runs["full"] = (Pf, (time.perf_counter() - t0) * 1e3, 0.0, 0.0)
            for f in fracs:
                B = f * full
                for m in METHODS:
                    a = Allocator(tb, method=m)
                    a.rr.ptr = int(np.random.default_rng([seed, si]).integers(max(pb.E, 1)))
                    t0 = time.perf_counter()
                    pl = a.plan(pb, B)
                    P = a.execute(pb, pl, t0 + B / 1e3)
                    runs[f"{m}|f{f}"] = (P, (time.perf_counter() - t0) * 1e3, pl.alloc_ms, B)
            base = {}
            for arm, (P, ms, ov, B) in runs.items():
                colors = pb.to_rgb(pb.project(P[None])[0])[scene.groups]
                sc = [_score(scene, colors, p) for p in ho]
                ws, mn = float(np.mean([s[0] for s in sc])), float(np.mean([s[1] for s in sc]))
                if arm == "none":
                    base[kind] = ws
                add(arm, wsuccess=ws, min_de00=mn, ms=ms, overhead_ms=ov,
                    naturalness=naturalness(scene.colors, colors, scene.protected)["mean_de00"])
                add(arm, gain=ws - base[kind], gain_per_ms=(ws - base[kind]) / max(ms, 1e-9) if arm != "none" else 0.0,
                    budget_use=ms / B if B else 0.0)
    return {a: {k: float(np.mean(v)) for k, v in d.items()} for a, d in acc.items()}
