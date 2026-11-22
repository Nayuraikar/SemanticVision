"""K4 temporal-stability ablation (SPEC 5.9): none -> EMA -> hysteresis -> slew -> event-driven, each ALONE and
COMBINED, on replayed scripted streams (collision, state change, enter view, lighting change + noise).
The solver and the stabiliser's pair decisions see only TRAIN observers of the kind; every number is scored under
HELD-OUT observers (own model, severity, k, s_L, s_C) on held-out streams. Flicker is scored on displayed lit colours;
stability must not be bought by freezing: critical-pair min dE00 / success are reported next to it.
Arms: uncorrected (reference), none (raw per-frame solve), ema, hysteresis, slew, event_driven, combined (all four),
no_ema (combined minus EMA). Latency = frames (x dt) from the scripted event to a sustained resolved pair under the
held-out observer, counted only for (event, observer) pairs that are confusable without correction."""
from __future__ import annotations
import numpy as np
from adapters.synthetic import StreamConfig, generate_stream
from bench.registry import experiment
from core.observers import heldout_observers, train_observers
from core.perception import distance as Dst
from core.server import pair_ratios, solve, solver_observers
from core.solver.problem import Scene
from core.stabiliser import StabConfig, Stabiliser, de00

ARMS = ("uncorrected", "none", "ema", "hysteresis", "slew", "event_driven", "combined", "no_ema")
SCENE_KW = dict(adj_margin=0.0)     # adjacency = bbox overlap
WARMUP = 3


def arm_cfg(name, **kw):
    if name == "combined":
        return StabConfig.all_on(**kw)
    if name == "no_ema":                       # combined without the EMA lag
        return StabConfig.only("hysteresis", "slew", "event_driven", **kw)
    return StabConfig() if name == "none" else StabConfig.only(name, **kw)


_POOL = None


def _pool(workers):
    """One spawn pool for the whole bench run (workers pin BLAS to 1 thread)."""
    global _POOL
    if _POOL is None:
        import multiprocessing as mp
        import os
        from concurrent.futures import ProcessPoolExecutor
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ.setdefault(k, "1")
        _POOL = ProcessPoolExecutor(workers, mp_context=mp.get_context("fork"))
    return _POOL


def _solve_one(args):
    snap, display, table, cfg_extra = args
    info = {}
    ov = solve(snap, display, table, cfg_extra=cfg_extra, scene_kw=SCENE_KW, info=info)
    return ov, info["analysis"]


class Ctx:
    """Per-seed shared state: scenes, cached solves (identical across arms: the solver is stateless)."""

    def __init__(self, snaps, display, table, held, cfg_extra=None):
        self.snaps, self.display, self.table, self.held = snaps, display, table, held
        self.cfg_extra = cfg_extra or {}
        self.scenes = [Scene.from_snapshot(s, **SCENE_KW) for s in snaps]
        self.sobs = solver_observers(display, table)
        self._solve, self._an = {}, {}
        self.taus = [[sc.graph(o).tau for o in held] for sc in self.scenes]

    def precompute(self, workers):
        """Solve every frame once (arms share them: the solver is stateless); parallel over frames."""
        args = [(s, self.display, self.table, self.cfg_extra) for s in self.snaps]
        res = list(_pool(workers).map(_solve_one, args, chunksize=4)) if workers else [_solve_one(a) for a in args]
        self._solve = {s.frame: r for s, r in zip(self.snaps, res)}

    def solve_fn(self, snap):
        if snap.frame not in self._solve:
            self._solve[snap.frame] = _solve_one((snap, self.display, self.table, self.cfg_extra))
        return self._solve[snap.frame]

    def analyse_fn(self, snap):
        if snap.frame not in self._an:
            self._an[snap.frame] = pair_ratios(self.scenes[snap.frame], self.sobs)
        return self._an[snap.frame]

    def lit(self, f, shown):
        """Displayed lit colours {id: rgb} of frame f given the exposed overrides (missing -> original)."""
        sc, snap = self.scenes[f], self.snaps[f]
        out = {}
        for k, o in enumerate(snap.objects):
            a, e = shown.get(o.id, (np.array(o.albedo_rgb), np.array(o.emissive_rgb)))
            out[o.id] = np.clip(np.asarray(a) * sc.gain[k] + np.asarray(e) * sc.exposure, 0, 1)
        return out

    def pair_D(self, f, lit, pair, oi):
        sc = self.scenes[f]
        i, j = sc.ids.index(pair[0]), sc.ids.index(pair[1])
        D = Dst.pair_matrix(np.array([lit[pair[0]], lit[pair[1]]]), **sc.perceive_kw(self.held[oi]))[0, 1]
        return float(D), float(self.taus[f][oi][i, j])


def run_arm(ctx, name, **kw):
    snaps = ctx.snaps
    if name == "uncorrected":
        lits = [ctx.lit(f, {}) for f in range(len(snaps))]
        return dict(lits=lits, solves=0, clamped=0, cue_changes=0)
    st = Stabiliser(arm_cfg(name, **kw))
    lits, cues, clamped, prev = [], [], 0, {}
    cue_changes = 0
    for f, s in enumerate(snaps):
        r = st.step(s, ctx.solve_fn, ctx.analyse_fn)
        lits.append(ctx.lit(f, r.displayed))
        clamped += r.slew_clamped
        cur = {c.id: (c.type, c.parameters.get("variant")) for c in r.overrides.cues}
        cue_changes += sum(cur.get(i) != prev.get(i) for i in set(cur) | set(prev))
        prev = cur
    return dict(lits=lits, solves=st.n_solves, clamped=clamped, cue_changes=cue_changes)


def flicker_of(lits, quiet=None):
    """Mean over objects of the sum over frames of dE00 between consecutive displayed colours."""
    tot = {}
    for f in range(WARMUP, len(lits)):         # skip the start-up transient (first palette appearing)
        if quiet is not None and not quiet[f]:
            continue
        for i, c in lits[f].items():
            if i in lits[f - 1]:
                tot[i] = tot.get(i, 0.0) + float(de00(c, lits[f - 1][i]))
    return float(np.mean(list(tot.values()))) if tot else 0.0


@experiment("stability_ablation")
def run(params, seed):
    cfg = StreamConfig.from_dict(params.get("stream", {}))
    split = params.get("split", "heldout")
    rng = np.random.default_rng([seed, 0x4B1D])
    cfg.kind = ("protan", "deutan")[int(rng.integers(2))]
    snaps, events = generate_stream(cfg, seed, split)
    tr = [o for o in train_observers(60, seed) if o.kind == cfg.kind][:int(params.get("n_train", 3))]
    held = [o for o in heldout_observers(40, seed) if o.kind == cfg.kind][:int(params.get("n_heldout", 4))]
    if not tr or not held:
        return {}
    ctx = Ctx(snaps, held[0], {o.id: o for o in tr} | {held[0].id: held[0]}, held, params.get("solver"))
    ctx.precompute(int(params.get("workers", 0)))
    kw = dict(compute_frames=int(params.get("compute_frames", 1)), default_dt=cfg.dt, **params.get("stab", {}))
    win = int(params.get("window", 20))
    n = len(snaps)
    ev_frames = [s.frame for s in snaps if s.events]
    quiet = np.array([not any(f - 10 <= g <= f for g in ev_frames) for f in range(n)])
    runs = {a: run_arm(ctx, a, **(kw if a != "uncorrected" else {})) for a in params.get("arms", ARMS)}

    # valid critical (event, observer): confusable without correction at the event frame and at the window end
    valid = []
    base = runs["uncorrected"]["lits"]
    for e in events:
        f1 = min(e.frame + win, n) - 1
        if f1 <= e.frame:
            continue
        for oi in range(len(held)):
            try:
                d0, t0 = ctx.pair_D(e.frame, base[e.frame], e.pair, oi)
                d1, t1 = ctx.pair_D(f1, base[f1], e.pair, oi)
            except (ValueError, KeyError):
                continue
            if d0 < t0 and d1 < t1:
                valid.append((e, oi))
    if not valid:
        return {}
    out = {}
    fl0 = flicker_of(base)
    flq0 = flicker_of(base, quiet)
    for a, r in runs.items():
        lits = r["lits"]
        lat, unres, dmin, succ, succ_fin = [], [], [], [], []
        frame_min = {}
        for e, oi in valid:
            f1 = min(e.frame + win, n)
            ser = [ctx.pair_D(f, lits[f], e.pair, oi) for f in range(e.frame, f1)]
            ok = [d >= t for d, t in ser]
            first = next((k for k in range(len(ok)) if all(ok[k:k + 3])), None)
            lat.append(len(ok) if first is None else first)
            unres.append(first is None)
            succ.append(np.mean(ok))
            succ_fin.append(np.mean(ok[-8:]))
            for k, (d, t) in enumerate(ser):
                frame_min.setdefault((e.frame + k, oi), []).append(d)
        dmin = [min(v) for v in frame_min.values()]
        fl, flq = flicker_of(lits), flicker_of(lits, quiet)
        out[a] = dict(flicker=fl, flicker_excess=fl - fl0, flicker_quiet=flq, flicker_quiet_excess=flq - flq0,
                      latency_frames=float(np.mean(lat)), latency_ms=float(np.mean(lat)) * cfg.dt * 1e3,
                      unresolved=float(np.mean(unres)), crit_min_de00=float(np.mean(dmin)),
                      crit_success=float(np.mean(succ)), crit_success_final=float(np.mean(succ_fin)),
                      solves_per_frame=r["solves"] / n, clamped_per_frame=r["clamped"] / n,
                      cue_changes=float(r["cue_changes"]), n_critical=float(len(valid)))
    return out
