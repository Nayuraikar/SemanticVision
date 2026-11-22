"""Temporal stabiliser (SPEC 5.9): wraps a per-frame solver. Each mechanism is an independent switch so the ablation
can run them alone and combined.

Pipeline per frame: triggers -> (re)solve or hold -> hysteresis gate -> EMA -> slew -> candidate palette ->
(optional compute delay) -> verify -> double-buffer swap -> renderer reads the validated front palette."""
from __future__ import annotations
import math
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Callable, Optional

import numpy as np

from protocol.models import Diagnostics, Overrides, PaletteOverride, Snapshot
from .buffer import DoubleBuffer, Palette
from .hysteresis import CueHysteresis, PairHysteresis
from .metrics import de00
from .slew import slew_limit
from .triggers import TriggerConfig, TriggerDetector

MECHANISMS = ("ema", "hysteresis", "slew", "event_driven")
Pair = tuple


@dataclass
class StabConfig:
    ema: bool = False
    hysteresis: bool = False
    slew: bool = False
    event_driven: bool = False
    ema_tau: float = 0.25          # s
    enter: float = 1.8             # pair becomes confusing when dE00/tau < enter ... (margin over 1: the solver acts up
    exit: float = 2.2              # ... and stops when dE00/tau > exit        to cap=2*tau; held-out observers differ)
    dwell: int = 8                 # min frames a pair stays active
    hold: int = 8                  # frames a vanished pair (adjacency flap) keeps its state
    cue_hold: int = 15
    r_max: float = 60.0            # dE00 / s
    compute_frames: int = 0        # simulated solver latency (candidate published this many frames later)
    max_age: int = 0               # event-driven safety net: force a re-solve every n frames (0 = off)
    default_dt: float = 1 / 30
    importance_jump: float = 0.5
    lighting_eps: float = 0.05

    @classmethod
    def all_on(cls, **kw):
        return cls(**{**{m: True for m in MECHANISMS}, **kw})

    @classmethod
    def only(cls, *names, **kw):
        bad = set(names) - set(MECHANISMS)
        if bad:
            raise ValueError(f"unknown mechanism {bad}")
        return cls(**{**{m: True for m in names}, **kw})


@dataclass
class Analysis:
    """Cheap per-frame view of the scene: pair ratio dE00/tau under the solver's observers (original colours),
    palette group per object id, id pairs that need cues."""
    ratios: dict = field(default_factory=dict)
    groups: dict = field(default_factory=dict)
    cue_pairs: set = field(default_factory=set)


@dataclass
class StepResult:
    overrides: Overrides
    triggers: list
    solved: bool
    slew_clamped: int
    rejected: bool
    palette: Optional[Palette]
    displayed: dict                 # id -> (albedo, emissive) actually exposed to the renderer this frame


def _arr(d, ids, k):
    return np.array([d[i][k] for i in ids], float).reshape(-1, 3)


class Stabiliser:
    def __init__(self, cfg: StabConfig | None = None, verify: Callable | None = None):
        self.cfg = cfg or StabConfig()
        self.extra_verify = verify
        self.buffer = DoubleBuffer()
        self.reset()

    def reset(self):
        c = self.cfg
        self.det = TriggerDetector(TriggerConfig(c.importance_jump, c.lighting_eps))
        self.pairs = (PairHysteresis(c.enter, c.exit, c.dwell, c.hold) if c.hysteresis
                      else PairHysteresis(c.enter, c.enter))
        self.cues = CueHysteresis(c.cue_hold)
        self.buffer = DoubleBuffer()
        self.t = None
        self.version = 0
        self.age = 0
        self.sol: dict = {}            # id -> (albedo, emissive) of the last solve
        self.sol_cues: dict = {}
        self.sol_ov: Optional[Overrides] = None
        self.an = Analysis()
        self.ema: dict = {}            # id -> smoothed correction offset
        self.off: dict = {}            # id -> last candidate's correction offset (slew reference = original + off)
        self.ref: dict = {}            # id -> slew reference used for the current candidate
        self.corr: dict = {}           # id -> last real correction (held while a pair is inside its hysteresis band)
        self.pending: deque = deque()
        self.n_solves = 0

    # ------------------------------------------------------------------ verification
    def _verify(self, back: Palette, front: Optional[Palette]) -> bool:
        for d in (back.albedo, back.emissive):
            for v in d.values():
                if not all(math.isfinite(x) for x in v):
                    return False
        if any(not all(0.0 <= x <= 1.0 for x in v) for v in back.albedo.values()):
            return False
        if any(any(x < 0.0 for x in v) for v in back.emissive.values()):
            return False
        if self.cfg.slew and back.ref:          # the candidate must sit within r_max*dt of its slew reference
            for k, d_b in enumerate((back.albedo, back.emissive)):
                ids = [i for i in back.ref if i in d_b]
                if ids:
                    d = de00(np.array([d_b[i] for i in ids]), np.array([back.ref[i][k] for i in ids]))
                    if np.any(d > self.cfg.r_max * back.dt + 1e-6):
                        return False
        return True if self.extra_verify is None else bool(self.extra_verify(back, front))

    # ------------------------------------------------------------------ step
    def step(self, snap: Snapshot, solve_fn: Callable, analyse_fn: Optional[Callable] = None) -> StepResult:
        """solve_fn(snap) -> (Overrides, Analysis); analyse_fn(snap) -> Analysis (cheap monitor, optional)."""
        c, f = self.cfg, snap.frame
        dt = snap.t - self.t if self.t is not None and snap.t > self.t else c.default_dt
        self.t = snap.t
        ids = [o.id for o in snap.objects]
        orig = {o.id: (np.array(o.albedo_rgb, float), np.array(o.emissive_rgb, float)) for o in snap.objects}
        first = self.sol_ov is None

        # 1. decide whether to re-solve
        if c.event_driven:
            if analyse_fn is not None:
                self.an = analyse_fn(snap)
            _, new_pairs = self.pairs.update(self.an.ratios, f)
            triggers = self.det.detect(snap, new_pairs)
            solve = bool(triggers) or first or (c.max_age and self.age >= c.max_age)
        else:
            solve, new_pairs = True, ()
        if solve:
            ov, self.an = solve_fn(snap)
            self.n_solves += 1
            self.age = 0
            self.sol_ov = ov
            self.sol = {p.id: (np.array(p.albedo, float),
                               np.array(p.emissive if p.emissive is not None else orig.get(p.id, (0, np.zeros(3)))[1], float))
                        for p in ov.palette_overrides}
            self.sol_cues = {k.id: k for k in ov.cues}
        else:
            self.age += 1
        if not c.event_driven:
            _, new_pairs = self.pairs.update(self.an.ratios, f)
            triggers = self.det.detect(snap, new_pairs)
        active_pairs = {k for k, st in self.pairs.state.items() if st[0]}

        # 2. hysteresis gate: only entries in an active confusing pair carry the correction
        if c.hysteresis:
            act_ids = {i for p in active_pairs for i in p}
            grp = self.an.groups
            act_groups = {grp[i] for i in act_ids if i in grp}
            gate = lambda i: i in act_ids or (i in grp and grp[i] in act_groups)
        else:
            gate = lambda i: True
        target = {}
        for i in ids:
            s = self.sol.get(i)
            if not gate(i):
                self.corr.pop(i, None)
                target[i] = orig[i]
            elif s is None:
                target[i] = orig[i]
            elif not c.hysteresis or max(float(de00(s[0], orig[i][0])), float(de00(s[1], orig[i][1]))) > 0.5:
                self.corr[i] = s
                target[i] = s
            else:                                   # solver does not act now (pair vanished / inside the band): hold
                target[i] = self.corr.get(i, s)

        # 3./4. EMA and slew act on the CORRECTION OFFSET (displayed - original), so true scene changes (a light
        # turning green) are never delayed; only the stabiliser's own changes are smoothed / rate-limited.
        clamped = 0
        a = 1.0 - math.exp(-dt / c.ema_tau)
        want, ref = {}, {}
        for i in ids:
            off = (target[i][0] - orig[i][0], target[i][1] - orig[i][1])
            if c.ema and i in self.ema:
                off = tuple(p + a * (q - p) for p, q in zip(self.ema[i], off))
            self.ema[i] = off
            want[i] = (orig[i][0] + off[0], orig[i][1] + off[1])
            if i in self.off:
                ref[i] = (np.clip(orig[i][0] + self.off[i][0], 0.0, 1.0), np.maximum(orig[i][1] + self.off[i][1], 0.0))
        cand = dict(want)
        if c.slew:
            old = [i for i in ids if i in ref]
            if old:
                lim = c.r_max * dt
                oa, ca = slew_limit(_arr(ref, old, 0), _arr(want, old, 0), lim)
                oe, ce = slew_limit(_arr(ref, old, 1), _arr(want, old, 1), lim)
                clamped = int(ca.sum() + ce.sum())
                for k, i in enumerate(old):
                    cand[i] = (oa[k], oe[k])
        cand = {i: (np.clip(a_, 0.0, 1.0), np.maximum(e_, 0.0)) for i, (a_, e_) in cand.items()}
        self.off = {i: (cand[i][0] - orig[i][0], cand[i][1] - orig[i][1]) for i in ids}
        self.ema = {i: v for i, v in self.ema.items() if i in orig}
        self.ref = {i: ref[i] for i in ids if i in ref}

        # 5. cues
        raw = {i: k for i, k in self.sol_cues.items() if i in orig}
        cues = (self.cues.update(raw, self.an.cue_pairs, f, set(ids)) if c.hysteresis else raw)

        # 6. candidate -> (delay) -> verify -> swap
        self.version += 1
        pal = Palette.build(self.version, f, dt, {i: v[0] for i, v in cand.items()}, {i: v[1] for i, v in cand.items()},
                            {i: k for i, k in cues.items() if i in cand},
                            ref={i: (tuple(map(float, a)), tuple(map(float, e))) for i, (a, e) in self.ref.items()})
        self.pending.append((f + c.compute_frames, pal))
        rejected = False
        while self.pending and self.pending[0][0] <= f:
            _, p = self.pending.popleft()
            self.buffer.stage(p)
            if not self.buffer.commit(self._verify):
                rejected = True
                self.pending.clear()                       # later candidates were chained from a rejected one
                self.off, self.ema = {}, {}                # restart the correction from the validated front palette
                break

        # 7. renderer view
        front = self.buffer.read()
        shown = {} if front is None else {i: (np.array(front.albedo[i]), np.array(front.emissive[i]))
                                          for i in ids if i in front.albedo}
        po = [PaletteOverride(id=i, albedo=tuple(float(x) for x in a), emissive=tuple(float(x) for x in e))
              for i, (a, e) in shown.items()]
        base = self.sol_ov
        diag = Diagnostics(stage_latency_ms=dict(base.diagnostics.stage_latency_ms) if solve else {},
                           solver_tier=base.diagnostics.solver_tier if solve else "cached",
                           fallback_flags=list(base.diagnostics.fallback_flags),
                           triggers=list(triggers), slew_clamped=clamped,
                           tier_counts=dict(base.diagnostics.tier_counts) if solve else {},
                           allocator_ms=base.diagnostics.allocator_ms if solve else 0.0,
                           entering=list(base.diagnostics.entering) if solve else [])
        keep = set(ids)
        out = Overrides(frame=f, palette_overrides=po,
                        cues=[k for i, k in (front.cues.items() if front else []) if i in keep],
                        confusion_pairs=[p for p in base.confusion_pairs if p.ids[0] in keep and p.ids[1] in keep],
                        diagnostics=diag)
        return StepResult(out, list(triggers), bool(solve), clamped, rejected, front, shown)


def replay(snaps, cfg: StabConfig, solve_fn, analyse_fn=None, verify=None):
    """Deterministic replay of a snapshot stream through a fresh stabiliser."""
    st = Stabiliser(cfg, verify)
    return [st.step(s, solve_fn, analyse_fn) for s in snaps]
