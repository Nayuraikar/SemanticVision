"""Budgeted allocator (SPEC 5.8): per palette group choose a tier by MCKP under budget_ms, run only those tiers via core/solver.
Costs/benefits come from measured tables (tables.py) or an online LinUCB bandit; importance weights the benefit."""
from __future__ import annotations
import time
from dataclasses import dataclass, field, replace
import numpy as np
from core.solver.problem import Problem, Scene, SolverConfig
from core.solver.solver import Solution, _package, screen_check
from . import mckp
from .tables import Tables, load_tables, N_TIERS
from .tiers import GroupCache, default_tiers, group_keys

TIER_NAMES = ("skip", "reuse", "learned", "coarse", "fine")


@dataclass
class Plan:
    tiers: np.ndarray                 # (E,) chosen tier per palette group
    benefit: np.ndarray               # (E,T) estimated (importance-weighted)
    cost: np.ndarray                  # (E,T) estimated ms
    X: np.ndarray                     # (E,F) group features
    budget_ms: float                  # budget handed to the MCKP
    est_cost: float
    est_benefit: float
    alloc_ms: float
    method: str
    hits: dict = field(default_factory=dict)


def group_stats(pb: Problem):
    """-> D0 (1,M,P), deficit (E,), deg (E,), critical-pair mask (P,). Deficit = weighted mean over incident pairs of
    mean_m max(0, 1 - D/tau) on the ORIGINAL palette (train observers)."""
    D0 = pb.distance_tensor(pb.lab0[None])
    E, P = pb.E, len(pb.pairs)
    if not P:
        return D0, np.zeros(E), np.zeros(E), np.zeros(0, bool)
    pdef = np.maximum(0.0, 1.0 - D0[0] / np.maximum(pb.tau, 1e-9)).mean(0)
    w = pb.w
    num, den, deg = np.zeros(E), np.zeros(E), np.zeros(E)
    for ends in (pb.ia, pb.ib):
        np.add.at(num, ends, w * pdef); np.add.at(den, ends, w); np.add.at(deg, ends, 1.0)
    return D0, num / np.maximum(den, 1e-12), deg, pdef > 0


def group_features(pb: Problem, deficit, deg):
    sz = pb.scene.palette()[1]
    return np.stack([np.ones(pb.E), deficit, deficit ** 2, np.log1p(deg), np.log(np.maximum(sz, 1e-3)),
                     pb.prot.astype(float)], 1)


def realised_gain(pb: Problem, D0, crit, P):
    """Per group: min over incident critical pairs of the observer-mean dE00 gain of palette P vs the original."""
    g = np.zeros(pb.E)
    if not crit.any():
        return g
    D1 = pb.distance_tensor(pb.project(np.asarray(P)[None]))
    d = (D1 - D0)[0].mean(0)
    best = np.full(pb.E, np.inf)
    for ends in (pb.ia, pb.ib):
        np.minimum.at(best, ends[crit], d[crit])
    return np.where(np.isfinite(best), best, 0.0)


class Allocator:
    def __init__(self, tables: Tables | None = "auto", method: str = "lagrangian", bandit=None, tiers=None,
                 cache: GroupCache | None = None):
        self.tables = load_tables() if isinstance(tables, str) else tables
        self.method, self.bandit = method, bandit
        self.tiers = tiers or default_tiers()
        self.cache = cache or GroupCache()
        self.rr = mckp.RoundRobin()
        self.overhead_ms = 1.0            # EMA of allocator's own latency (reserved from the budget)

    # ---- estimates ---------------------------------------------------------------------------------------
    def estimates(self, pb, scene_keys, deficit, deg, F, imp):
        n, T = pb.E, N_TIERS
        n_act = max(int(((deficit > 0) & (deg > 0)).sum()), 1)
        cost = self.tables.cost_matrix(pb.E, len(pb.pairs), deg, 2.0 / n_act)   # ~half the active groups share a tier call
        gain = self.tables.gain_matrix(F)
        if self.bandit is not None:
            gain = self.bandit.estimate(F)
        hits = {}
        lam = float(self.tables.reuse.get("decay_de", 6.0))
        for g in range(n):
            h = self.cache.lookup(scene_keys[g], pb.lab0[g]) if deficit[g] > 0 else None
            if h is not None:
                de, _, _, ct = h
                hits[g] = (de, ct)
                gain[g, 1] = gain[g, max(ct, 2)] * np.exp(-de / lam)
        act = (deficit > 0) & (deg > 0)
        ben = gain * imp[:, None] * act[:, None]
        cost = cost * act[:, None]
        ben[:, 0] = cost[:, 0] = 0.0
        for g in range(n):
            if g not in hits:
                ben[g, 1] = 0.0
        return ben, cost, hits

    def choose(self, ben, cost, B, method=None):
        m = method or self.method
        if m == "round_robin":
            return self.rr(ben, cost, B)
        if m == "bnb":
            return mckp.bnb(ben, cost, B)[0]
        return mckp.SOLVERS[m](ben, cost, B)

    def plan(self, pb: Problem, budget_ms: float, imp=None, method=None) -> Plan:
        t0 = time.perf_counter()
        D0, deficit, deg, crit = group_stats(pb)
        F = group_features(pb, deficit, deg)
        im = pb.scene.palette()[3] if imp is None else np.asarray(imp, float)
        keys = group_keys(pb.scene)
        ben, cost, hits = self.estimates(pb, keys, deficit, deg, F, im)
        B = max(budget_ms - self.overhead_ms, 0.0)
        ch = self.choose(ben, cost, B, method)
        i = np.arange(pb.E)
        el = (time.perf_counter() - t0) * 1e3
        self.overhead_ms = 0.8 * self.overhead_ms + 0.2 * el
        pl = Plan(ch, ben, cost, F, B, float(cost[i, ch].sum()), float(ben[i, ch].sum()), el, method or self.method, hits)
        pl._stats = (D0, crit, keys)
        return pl

    # ---- execution ---------------------------------------------------------------------------------------
    def execute(self, pb, plan: Plan, deadline=None):
        P = pb.lab0.copy()
        keys = plan._stats[2]
        for t in range(1, N_TIERS):                       # cheap tiers first; most valuable groups first within a tier
            gs = np.flatnonzero(plan.tiers == t)
            if not len(gs):
                continue
            gs = gs[np.argsort(-plan.benefit[gs, t])]
            P = self.tiers[t].run(pb, P, gs, deadline, self.cache, keys)
        return P

    def remember(self, pb, plan, P):
        keys = plan._stats[2]
        for g in np.flatnonzero(plan.tiers >= 2):
            self.cache.put(keys[g], pb.lab0[g], P[g], int(plan.tiers[g]))

    def feedback(self, pb, plan, P):
        """Online learning: realised gain per (features, chosen tier) -> bandit."""
        if self.bandit is None:
            return
        D0, crit, _ = plan._stats
        r = realised_gain(pb, D0, crit, P)
        sel = np.flatnonzero(plan.tiers >= 1)
        self.bandit.update_batch(plan.X[sel], plan.tiers[sel], np.maximum(r[sel], 0.0))

    def solve(self, scene: Scene, observers, cfg: SolverConfig = SolverConfig(), seed=0, t_start=None, importance=None,
              method=None):
        """-> (Solution, info). Whole call (incl. verification and packaging) stays inside cfg.budget_ms of t_start."""
        t0 = t_start if t_start is not None else time.perf_counter()
        hard = t0 + cfg.budget_ms / 1e3
        pb = Problem(scene, observers, cfg)
        tc = time.perf_counter()
        screen_check(scene, scene.colors, observers)
        reserve = 1.5 * (time.perf_counter() - tc)           # verification + packaging must still fit
        if self.tables is None:                              # uncalibrated: exact fine tier for everything
            pl = self._fallback_plan(pb)
        else:
            pl = self.plan(pb, (hard - time.perf_counter() - reserve) * 1e3, importance, method)
        P = self.execute(pb, pl, hard - reserve)
        self.remember(pb, pl, P)
        self.feedback(pb, pl, P)
        sol = _package(scene, pb, P, float(pb.objective(P[None])[0]), t0, 1, observers)
        cnt = {TIER_NAMES[t]: int((pl.tiers == t).sum()) for t in range(N_TIERS)}
        return sol, dict(tier_counts=cnt, alloc_ms=pl.alloc_ms, method=pl.method, est_cost=pl.est_cost,
                         est_benefit=pl.est_benefit, plan=pl)

    def _fallback_plan(self, pb):
        E = pb.E
        ch = np.where(pb.active, 4, 0)
        z = np.zeros((E, N_TIERS))
        _, deficit, deg, crit = group_stats(pb)
        pl = Plan(ch, z, z, group_features(pb, deficit, deg), 0.0, 0.0, 0.0, 0.0, "uncalibrated")
        pl._stats = (pb.distance_tensor(pb.lab0[None]), crit, group_keys(pb.scene))
        return pl

    # ---- prediction --------------------------------------------------------------------------------------
    @staticmethod
    def with_entering(scene: Scene, snap, boxes: dict, margin=0.3) -> Scene:
        """Adjacency recomputed with the predicted bbox of objects about to enter view, so their groups are planned
        (and cached for tier-1 reuse) before they appear."""
        if not boxes:
            return scene
        from core.perception.graph import bbox_adjacency
        bb = np.array([(o.bbox.x, o.bbox.y, o.bbox.w, o.bbox.h) for o in snap.objects], float)
        for k, o in enumerate(snap.objects):
            if o.id in boxes:
                bb[k] = boxes[o.id]
        xyxy = np.c_[bb[:, 0], bb[:, 1], bb[:, 0] + bb[:, 2], bb[:, 1] + bb[:, 3]]
        return replace(scene, adjacency=scene.adjacency | bbox_adjacency(xyxy, margin))
