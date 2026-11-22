"""Lab-space coordinate descent (grid init + constraint projection), DE reference, verification loop."""
from __future__ import annotations
import time
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import differential_evolution
from .problem import Problem, Scene, SolverConfig
from .cues import assign_cues, VOCAB
from .metrics import naturalness


@dataclass
class Solution:
    colors: np.ndarray            # (N,3) corrected linear RGB per object
    palette_rgb: np.ndarray       # (E,3)
    palette_lab: np.ndarray
    objective: float
    infeasible: list              # group pairs still confusable in screen space after retries
    infeasible_ids: list          # same, as object-id pairs
    cues: dict                    # group -> (type, variant)
    cue_unresolved: list
    attempts: int
    time_ms: float
    nat: dict = field(default_factory=dict)

    @property
    def n_infeasible(self):
        return len(self.infeasible)


def _late(deadline):
    return deadline is not None and time.perf_counter() > deadline


def coordinate_descent(pb: Problem, P0=None, deadline=None):
    """Grid init + coordinate descent. `deadline` (perf_counter time) is checked before every candidate batch,
    so the loop returns the best feasible palette found so far as soon as the budget is spent."""
    cfg = pb.cfg
    P = pb.lab0.copy() if P0 is None else pb.project(np.array(P0, float))
    cur = float(pb.objective(P[None])[0])
    act = np.flatnonzero(pb.active)
    if P0 is None and len(act):
        g = np.linspace(-1, 1, cfg.grid)
        off = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
        for _ in range(cfg.sweeps):
            for e in act:
                if _late(deadline):
                    return P, cur
                cand = np.repeat(P[None], len(off), 0)
                cand[:, e] = pb.lab0[e] + off * pb.delta[e]
                f = pb.objective(cand); i = int(np.argmax(f))
                if f[i] > cur + 1e-9:
                    P, cur = pb.project(cand[i]), float(f[i])
    step = cfg.step0
    for _ in range(cfg.max_iter * 4):
        if not len(act) or _late(deadline):
            break
        improved = False
        for e in act:
            if _late(deadline):
                return P, cur
            cand = np.repeat(P[None], 6, 0)
            for k in range(6):
                cand[k, e, k // 2] += step * pb.delta[e] * (1 if k % 2 == 0 else -1)
            f = pb.objective(cand); i = int(np.argmax(f))
            if f[i] > cur + 1e-7:
                P, cur, improved = pb.project(cand[i]), float(f[i]), True
        if not improved:
            step /= 2
            if step < cfg.step_min:
                break
    return P, cur


def screen_check(scene: Scene, obj_rgb, observers):
    """Screen-space re-evaluation: per-object confusion graph of the corrected colours (true object sizes,
    adjacency) under each observer; returns failing (group, group) pairs and object-id pairs."""
    gp, ip = set(), set()
    for o in observers:
        g = scene.graph(o, obj_rgb)
        for i, j in g.edges:
            a, b = int(scene.groups[i]), int(scene.groups[j])
            if a != b:
                gp.add((min(a, b), max(a, b))); ip.add((scene.ids[i], scene.ids[j]))
    return sorted(gp), sorted(ip)


def _package(scene, pb, P, obj, t0, attempts, observers, fail=None):
    X = pb.project(np.asarray(P)[None])[0]
    rgb = pb.to_rgb(X)
    colors = rgb[scene.groups]
    gp, ip = screen_check(scene, colors, observers) if fail is None else fail
    cues, bad = assign_cues(gp, VOCAB[: pb.cfg.n_cues])
    return Solution(colors, rgb, X, float(obj), gp, ip, cues, bad, attempts, (time.perf_counter() - t0) * 1e3,
                    naturalness(scene.colors, colors, scene.protected))


def correct(scene: Scene, observers, cfg: SolverConfig = SolverConfig(), seed=0) -> Solution:
    """Primary solver + verification loop: re-check in screen space; failing pairs are boosted and the
    descent re-run (perturbed restart) within budget/retries; what is still confusable is escalated to cues."""
    t0 = time.perf_counter()
    hard = t0 + cfg.budget_ms / 1e3                      # whole call, incl. verification and packaging
    rng = np.random.default_rng([seed, 0xC0D])
    pb = Problem(scene, observers, cfg)
    tc = time.perf_counter()
    screen_check(scene, scene.colors, observers)         # also measures the verification cost to reserve for it
    reserve = 1.5 * (time.perf_counter() - tc)
    deadline = hard - reserve                            # descent stops here so verification still fits the budget
    P, obj = coordinate_descent(pb, deadline=deadline)
    best, att = None, 0
    for att in range(cfg.retries + 1):
        X = pb.project(P[None])[0]
        fail = screen_check(scene, pb.to_rgb(X)[scene.groups], observers)
        if best is None or len(fail[0]) < len(best[2][0]):
            best = (P.copy(), obj, fail)
        if not fail[0] or time.perf_counter() > deadline or att == cfg.retries:
            break
        idx = {tuple(p): k for k, p in enumerate(pb.pairs)}
        for a, b in fail[0]:
            k = idx.get((a, b))
            if k is not None:
                pb.boost[k] *= 2.0; pb.tau[:, k] *= 1.3
        pb.beta *= 1.5
        P0 = best[0] + rng.normal(0, 0.1, best[0].shape) * pb.delta[:, None] * pb.active[:, None]
        P, obj = coordinate_descent(pb, P0=P0, deadline=deadline)
    P, obj, fail = best
    return _package(scene, pb, P, obj, t0, att + 1, observers, fail)


def de_reference(scene: Scene, observers, cfg: SolverConfig = SolverConfig(), seed=0, maxiter=80, popsize=10) -> Solution:
    """Offline optimality reference: differential evolution over per-entry Lab offsets (same objective)."""
    t0 = time.perf_counter()
    pb = Problem(scene, observers, cfg)
    act = np.flatnonzero(pb.active)
    if not len(act):
        return _package(scene, pb, pb.lab0, pb.objective(pb.lab0[None])[0], t0, 0, observers)

    def decode(x):                                   # x (D,S) -> (S,E,3)
        P = np.repeat(pb.lab0[None], x.shape[1], 0)
        P[:, act] = pb.lab0[act] + x.T.reshape(-1, len(act), 3) * pb.delta[act][None, :, None]
        return P

    r = differential_evolution(lambda x: -pb.objective(decode(x)), [(-1, 1)] * (3 * len(act)), vectorized=True,
                               updating="deferred", seed=seed, maxiter=maxiter, popsize=popsize, tol=1e-6,
                               polish=False, init="sobol")
    P = decode(r.x[:, None])[0]
    return _package(scene, pb, P, -r.fun, t0, 0, observers)
