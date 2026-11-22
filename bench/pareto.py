"""Optional NSGA-II Pareto front: separation (soft-min dE00, maximise) vs naturalness (mean dE00, minimise).
python -m bench.pareto [--seed S] [--gen G] [--pop N] [--out bench/plots/pareto.json]"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from adapters.synthetic import SynthConfig, generate
from core.observers import train_observers
from core.solver import Scene, Problem, SolverConfig


def _ranks(F):
    """F (n,2) both minimised -> front index per row; crowding distance."""
    n = len(F)
    dom = (F[:, None] <= F[None]).all(-1) & (F[:, None] < F[None]).any(-1)
    cnt, rank, cur = dom.sum(0), np.full(n, -1), 0
    while (rank < 0).any():
        fr = np.flatnonzero((cnt == 0) & (rank < 0))
        rank[fr] = cur
        cnt = cnt - dom[fr].sum(0); cnt[fr] = -1
        cur += 1
    crowd = np.zeros(n)
    for r in range(cur):
        idx = np.flatnonzero(rank == r)
        for m in range(2):
            o = idx[np.argsort(F[idx, m])]
            crowd[o[[0, -1]]] = np.inf
            if len(o) > 2:
                crowd[o[1:-1]] += (F[o[2:], m] - F[o[:-2], m]) / max(np.ptp(F[idx, m]), 1e-12)
    return rank, crowd


def nsga2(pb: Problem, pop=40, gens=40, seed=0):
    rng = np.random.default_rng(seed)
    act = np.flatnonzero(pb.active)
    D = 3 * len(act)

    def decode(x):
        P = np.repeat(pb.lab0[None], len(x), 0)
        P[:, act] = pb.lab0[act] + x.reshape(len(x), -1, 3) * pb.delta[act][None, :, None]
        return pb.project(P)

    def evalf(x):
        X = decode(x)
        sep = pb.softmin(pb.separation_terms(X)) if len(pb.pairs) else np.zeros(len(x))
        return np.c_[-sep, pb.deviation(X).mean(-1)], X

    x = rng.uniform(-1, 1, (pop, D)); x[0] = 0
    F, _ = evalf(x)
    for _ in range(gens):
        rank, crowd = _ranks(F)
        pick = lambda: (lambda a, b: a if (rank[a], -crowd[a]) < (rank[b], -crowd[b]) else b)(*rng.integers(pop, size=2))
        kids = []
        for _ in range(pop):
            a, b = x[pick()], x[pick()]
            w = rng.uniform(-.25, 1.25, D)                           # blend crossover
            kids.append(np.clip(a + w * (b - a) + rng.normal(0, 0.05, D) * (rng.random(D) < 1.5 / D), -1, 1))
        kids = np.array(kids)
        Fk, _ = evalf(kids)
        xa, Fa = np.r_[x, kids], np.r_[F, Fk]
        rank, crowd = _ranks(Fa)
        keep = np.lexsort((-crowd, rank))[:pop]
        x, F = xa[keep], Fa[keep]
    rank, _ = _ranks(F)
    front = F[rank == 0]
    return front[np.argsort(front[:, 1])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--gen", type=int, default=40)
    ap.add_argument("--pop", type=int, default=40); ap.add_argument("--kind", default="deutan")
    ap.add_argument("--out", default=str(Path(__file__).parent / "plots" / "pareto.json"))
    a = ap.parse_args()
    s = Scene.from_snapshot(next(iter(generate(SynthConfig(n_objects=10, n_frames=1), a.seed))))
    obs = [o for o in train_observers(60, a.seed) if o.kind == a.kind][:3]
    front = nsga2(Problem(s, obs, SolverConfig()), a.pop, a.gen, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(seed=a.seed, kind=a.kind, softmin_de00=(-front[:, 0]).tolist(),
                                           mean_dev_de00=front[:, 1].tolist())))
    print(f"{len(front)} pareto points -> {a.out}")


if __name__ == "__main__":
    main()
