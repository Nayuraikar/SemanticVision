"""Correction tiers behind one interface (SPEC 5.8): 0 skip, 1 reuse cached, 2 learned (stand-in until Phase 9 ONNX),
3 coarse exact, 4 fine exact. A tier maps (Problem, palette Lab P, group indices) -> P with only those groups changed.
Swap the learned solver in with `Allocator.tiers[2] = MyOnnxTier()` (same run() signature)."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import replace
import numpy as np
from core.perception import color as C
from core.solver.solver import coordinate_descent

REUSE_TOL = 6.0     # dE00 between cached and current original colour for a cache hit


class GroupCache:
    """key (class/state) -> [(orig Lab, corrected Lab, tier)]; bounded per key."""

    def __init__(self, per_key=16):
        self.d: dict = {}
        self.per_key = per_key

    def put(self, key, lab0, lab, tier):
        L = self.d.setdefault(key, [])
        L[:] = [e for e in L if float(C.ciede2000(e[0], lab0)) > 1.0]
        L.append((np.array(lab0, float), np.array(lab, float), int(tier)))
        del L[:-self.per_key]

    def lookup(self, key, lab0):
        best = None
        for e in self.d.get(key, ()):
            de = float(C.ciede2000(e[0], lab0))
            if de < REUSE_TOL and (best is None or de < best[0]):
                best = (de, e)
        return None if best is None else (best[0], best[1][1], best[1][0], best[1][2])


def group_keys(scene):
    first = {}
    for i, g in enumerate(scene.groups):
        first.setdefault(int(g), scene.keys[i])
    return [first[g] for g in range(scene.n_groups)]


class Tier(ABC):
    id: int = -1
    name: str = ""

    @abstractmethod
    def run(self, pb, P, groups, deadline=None, cache: GroupCache | None = None, keys=None):
        """-> new palette Lab (E,3); only `groups` may change."""


class Skip(Tier):
    id, name = 0, "skip"

    def run(self, pb, P, groups, deadline=None, cache=None, keys=None):
        return P


class Reuse(Tier):
    id, name = 1, "reuse"

    def run(self, pb, P, groups, deadline=None, cache=None, keys=None):
        P = P.copy()
        for g in groups:
            hit = cache.lookup(keys[g], pb.lab0[g]) if cache is not None else None
            if hit is not None:
                _, lab, lab0c, _ = hit
                P[g] = pb.lab0[g] + (lab - lab0c)          # replay the cached correction offset
        return pb.project(P[None])[0]


class LearnedStandIn(Tier):
    """Phase-9 placeholder: cheap coordinate descent, grid 1 (no grid stage, start at the original colour), 1 pass of
    6 axis moves per entry at step0*delta. Same interface as the future ONNX solver."""
    id, name = 2, "learned"

    def run(self, pb, P, groups, deadline=None, cache=None, keys=None):
        import time
        P = P.copy()
        cur = float(pb.objective(P[None])[0])
        for e in groups:
            if deadline is not None and time.perf_counter() > deadline:
                break
            cand = np.repeat(P[None], 6, 0)
            for k in range(6):
                cand[k, e, k // 2] += pb.cfg.step0 * pb.delta[e] * (1 if k % 2 == 0 else -1)
            f = pb.objective(cand); i = int(np.argmax(f))
            if f[i] > cur + 1e-7:
                P, cur = pb.project(cand[i]), float(f[i])
        return P


class _Exact(Tier):
    over: dict = {}

    def run(self, pb, P, groups, deadline=None, cache=None, keys=None):
        cfg = replace(pb.cfg, **self.over)
        P, _ = coordinate_descent(pb, P0=P, deadline=deadline, act=np.asarray(groups, int), cfg=cfg, init_grid=True)
        return P


class CoarseExact(_Exact):
    id, name = 3, "coarse"
    over = dict(grid=3, sweeps=1, max_iter=12, step_min=0.05)


class FineExact(_Exact):
    id, name = 4, "fine"
    over = dict(grid=5, sweeps=2, max_iter=60, step_min=0.02)


def default_tiers(learned: str = "auto"):
    """learned: 'auto' (ONNX tier if core/allocator/learned.onnx + onnxruntime exist and SV_LEARNED != standin) | 'standin'."""
    t2 = None
    if learned == "auto":
        from .onnx_tier import load_learned_tier
        t2 = load_learned_tier()
    return [Skip(), Reuse(), t2 or LearnedStandIn(), CoarseExact(), FineExact()]
