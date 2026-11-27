"""python -m bench.allocator_calibrate [n_seeds]: MEASURE per-tier cost (ms/group) and quality (min dE00 gain on a group's critical
pairs) by running the tiers of core/allocator on seeded synthetic scenes (train observers only), in three batch regimes
(single group, random 4-group subset, all active groups -- deployment runs a whole tier's groups in one call, which amortises the
descent), then fit core/allocator/tables.json (cost ~ [1,E,P,deg,1/k], gain ~ GAIN_FEATS, reuse decay) and write
bench/results/allocator_{tier,heldout}_rows.csv. Machine-specific; calibration seeds 7000+, held-out (bench-only) rows 9000+."""
from __future__ import annotations
import sys, time, json, csv
from pathlib import Path
import numpy as np
from scipy.optimize import nnls
from adapters.synthetic import SynthConfig, generate
from core.observers import train_observers, KINDS
from core.perception import color as C
from core.solver import Scene, SolverConfig
from core.solver.problem import Problem
from core.allocator import default_tiers, GroupCache, group_stats, group_features, realised_gain, Tables
from core.allocator.tiers import group_keys

OUT = Path(__file__).resolve().parent / "results"
SIZES = (12, 20, 32, 48)


def scene_for(n, seed, adj=0.3):
    snap = next(generate(SynthConfig(n_objects=n, n_frames=1, palette="confusable"), seed))
    return Scene.from_snapshot(snap, adj_margin=adj)


def observers_for(seed):
    kind = KINDS[seed % 2]
    return [o for o in train_observers(60, seed) if o.kind == kind][:3]


def measure(n_seeds=8, verbose=True, base=7000):
    tiers = default_tiers()
    rows = []
    for s in range(n_seeds):
        for n in SIZES:
            seed = base + s * 10 + SIZES.index(n)
            scene = scene_for(n, seed)
            pb = Problem(scene, observers_for(seed), SolverConfig())
            D0, deficit, deg, crit = group_stats(pb)
            F = group_features(pb, deficit, deg)
            cand = np.flatnonzero((deficit > 0) & (deg > 0))
            if not len(cand):
                continue
            rng = np.random.default_rng([seed, 0xCA1])
            singles = rng.permutation(cand)[:6]
            runs = [("single", [g]) for g in singles] + [("sub", list(rng.permutation(cand)[:4])), ("all", list(cand))]
            base_row = lambda g, t: dict(seed=seed, tier=t, E=pb.E, P=len(pb.pairs), deg=deg[g], **{f"f{i}": F[g, i] for i in range(6)})
            for regime, gs in runs:
                if regime == "sub" and len(cand) <= 4:
                    continue
                for t in (2, 3, 4):
                    t0 = time.perf_counter()
                    P = tiers[t].run(pb, pb.lab0.copy(), np.array(gs), None, None, None)
                    ms = (time.perf_counter() - t0) * 1e3
                    gn = realised_gain(pb, D0, crit, P)
                    for g in gs:
                        rows.append(dict(base_row(g, t), ms=ms / len(gs), k=len(gs), regime=regime, gain=float(gn[g]), delta_de=0.0))
            cache_keys = group_keys(scene)
            for g in singles:                                              # reuse: cached tier-4 correction under scene change
                P4 = tiers[4].run(pb, pb.lab0.copy(), np.array([g]))
                g4 = float(realised_gain(pb, D0, crit, P4)[g])
                cache = GroupCache(); cache.put(cache_keys[g], pb.lab0[g], P4[g], 4)
                for eps in (0.0, 0.05, 0.15, 0.3):                         # lit colours scale by (1+eps)
                    sc2 = Scene.build(np.clip(scene.colors * (1 + eps), 0, 1), scene.keys, scene.size_deg, scene.area,
                                      scene.adjacency, scene.importance, scene.bg_lum, scene.ids, scene.protected)
                    pb2 = Problem(sc2, pb.obs, pb.cfg)
                    if pb2.E != pb.E:
                        continue
                    D02, _, _, crit2 = group_stats(pb2)
                    t0 = time.perf_counter()
                    P1 = tiers[1].run(pb2, pb2.lab0.copy(), np.array([g]), None, cache, cache_keys)
                    ms = (time.perf_counter() - t0) * 1e3
                    rows.append(dict(base_row(g, 1), ms=ms, k=1, regime="single", ref_gain4=g4, delta_de=float(C.ciede2000(pb.lab0[g], pb2.lab0[g])),
                                     gain=float(realised_gain(pb2, D02, crit2, P1)[g])))
        if verbose:
            print(f"seed {s}: {len(rows)} rows", flush=True)
    return rows


def fit(rows) -> Tables:
    cost, gain, meta = {}, {}, {}
    for t in (1, 2, 3, 4):
        R = [r for r in rows if r["tier"] == t]
        A = np.array([[1, r["E"], r["P"], r["deg"], 1.0 / r["k"]] for r in R], float)
        y = np.array([r["ms"] for r in R])
        w, _ = nnls(A, y)
        cost[str(t)] = w.tolist()
        meta[f"cost_r2_{t}"] = float(1 - ((A @ w - y) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))
        meta[f"ms_per_group_mean_{t}"] = float(y.mean())
    for t in (2, 3, 4):
        R = [r for r in rows if r["tier"] == t]
        A = np.array([[r[f"f{i}"] for i in range(6)] for r in R], float)
        y = np.maximum([r["gain"] for r in R], 0.0)
        w = np.linalg.solve(A.T @ A + 1e-3 * np.eye(6), A.T @ y)
        gain[str(t)] = w.tolist()
        meta[f"gain_r2_{t}"] = float(1 - ((A @ w - y) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))
        meta[f"gain_mean_{t}"] = float(y.mean())
    R1 = [r for r in rows if r["tier"] == 1 and r.get("ref_gain4", 0) > 0.5 and r["gain"] > 0]
    x = np.array([r["delta_de"] for r in R1]); lr = np.log(np.array([r["gain"] / r["ref_gain4"] for r in R1]))
    slope = float((x * lr).sum() / max((x * x).sum(), 1e-12)) if len(x) else -1 / 6.0   # through origin: r = exp(slope * de)
    meta["reuse_rows"] = len(R1)
    return Tables(cost, gain, {"decay_de": float(min(max(-1 / slope if slope < -1e-9 else 60.0, 1.0), 60.0))}, meta)


def main(n_seeds=6):
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    for name, base, n in (("allocator_tier_rows", 7000, n_seeds), ("allocator_heldout_rows", 9000, max(4, n_seeds // 2))):
        rs = measure(n, base=base)                                   # held-out rows: disjoint scene seeds, bench only
        keys = sorted({k for r in rs for k in r})
        with open(OUT / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, restval=""); w.writeheader(); w.writerows(rs)
        if base == 7000:
            rows = rs
    tb = fit(rows)
    tb.meta.update(n_rows=len(rows), n_seeds=n_seeds, seconds=time.time() - t0)
    tb.save()
    print(json.dumps(dict(cost=tb.cost, reuse=tb.reuse, meta=tb.meta), indent=1))
    return tb


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)        # 6 seeds keeps the run < 5 min with the ONNX tier 2
