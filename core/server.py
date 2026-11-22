"""FastAPI service. /correct: Snapshot -> Overrides (real solver, signature-cached); /observers. /gt: dev-only ground-truth sink
(training/scoring only, never read at runtime)."""
from __future__ import annotations
import base64
import hashlib
from collections import OrderedDict
import json
import os
import threading
import time
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from calibration.session import CalibConfig, Session
from core.observers import Observer, heldout_observers, train_observers
from core.profiles import PROFILE_PREFIX, ProfileStore, profile_observer, samples_to_observers
from core.solver.problem import Scene, SolverConfig
from core.solver.solver import correct as solver_correct
from protocol.models import ConfusionPair, Cue, Diagnostics, Overrides, PaletteOverride, Snapshot

DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
GT_DIR = Path("bench/logs/gt")
MAX_PIXELS = 4096 * 4096
OBS_N, OBS_SEED = 8, 0
BUFFERS = {"instance": 4, "class": 4, "state": 4, "color": 4}  # name -> bytes/pixel (RGBA8)


class GTFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frame: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    legend: dict = Field(default_factory=dict)  # {"instance": {id: idx}, "class": {...}, "state": {...}}
    buffers: dict[str, str]  # name -> base64 RGBA8 (row 0 = bottom, as read from WebGL)


LIVE_CFG = dict(retries=1, grid=3, sweeps=1, max_iter=20)
CUE_PARAMS = {"outline": lambda v: {"style": v, "width": 0.06, "color": [1, 1, 1]},
              "hatch": lambda v: {"pattern": v, "color": [1, 1, 1]},
              "icon": lambda v: {"glyph": {"triangle": "▲", "circle": "●"}.get(v, v)},
              "label": lambda v: {"text": v}}


def observer_table() -> dict[str, Observer]:
    return {o.id: o for o in train_observers(OBS_N, OBS_SEED) + heldout_observers(OBS_N, OBS_SEED)}


def solver_observers(obs: Observer, table: dict[str, Observer]) -> list[Observer]:
    """Never optimise for a held-out observer: use the train observers of the same (diagnosed) kind."""
    if obs.split == "train":
        return [obs]
    pool = [o for o in table.values() if o.split == "train" and o.kind == obs.kind]
    return pool or [Observer("fallback", "train", "machado", obs.kind, 1.0)]


def signature(snap: Snapshot, extra: str = "") -> str:
    h = hashlib.sha1((snap.observer_id + extra).encode())
    h.update(snap.lighting.model_dump_json().encode())
    for o in snap.objects:
        h.update(repr((o.id, o.cls, o.state, [round(x, 3) for x in o.albedo_rgb], [round(x, 3) for x in o.emissive_rgb],
                       round(o.bbox.x, 2), round(o.bbox.y, 2), round(o.bbox.w, 2), round(o.bbox.h, 2),
                       tuple(o.task_tags), o.normal and [round(x, 2) for x in o.normal],
                       o.shadow_vis and round(o.shadow_vis, 2))).encode())
    return h.hexdigest()


ROBUST_MODES = ("point", "cvar", "worst")
POSTERIOR_M = 6      # posterior samples in the live solve (latency scales ~M)


def solve(snap: Snapshot, obs: Observer, table: dict[str, Observer], cfg_extra: dict | None = None,
          posterior: list[Observer] | None = None, risk: str = "point") -> Overrides:
    """posterior + risk in (cvar, worst): optimise the risk measure over the posterior observers; the reported
    confusion pairs stay under `obs` (the posterior mean)."""
    t0 = time.perf_counter()
    lat: dict[str, float] = {}
    if not snap.objects:
        return Overrides(frame=snap.frame, diagnostics=Diagnostics(stage_latency_ms={"total": 0.0}, solver_tier="none"))
    scene = Scene.from_snapshot(snap)
    t1 = time.perf_counter(); lat["scene"] = (t1 - t0) * 1e3
    robust = bool(posterior) and risk in ("cvar", "worst")
    cfg = SolverConfig(budget_ms=float(snap.budget_ms), **{**LIVE_CFG, **(cfg_extra or {})},
                       **(dict(risk=risk) if robust else {}))
    sol = solver_correct(scene, posterior if robust else solver_observers(obs, table), cfg, seed=0)
    t2 = time.perf_counter(); lat["solve"] = (t2 - t1) * 1e3
    g0, g1 = scene.graph(obs), scene.graph(obs, sol.colors)
    D1 = g1.D
    pairs = [ConfusionPair(ids=(scene.ids[i], scene.ids[j]), de00_before=float(g0.D[i, j]), de00_after=float(D1[i, j]),
                           feasible=bool(D1[i, j] >= g1.tau[i, j])) for i, j in g0.edges
             if scene.ids[i] != scene.ids[j]]
    cues = []
    for gidx, (typ, var) in sol.cues.items():
        for k in np.flatnonzero(scene.groups == gidx):
            cues.append(Cue(id=scene.ids[k], type=typ, parameters={"variant": var, **CUE_PARAMS[typ](var)}))
    t3 = time.perf_counter(); lat["pairs_cues"] = (t3 - t2) * 1e3
    alb, em = scene.to_overrides(sol.colors)       # lit colours back to albedo / emission (never scaled by irradiance)
    po = [PaletteOverride(id=o.id, albedo=tuple(float(x) for x in alb[i]),
                          emissive=o.emissive_rgb if em is None else tuple(float(x) for x in em[i]))
          for i, o in enumerate(snap.objects)]
    flags = (["cue_unresolved"] if sol.cue_unresolved else []) + (["infeasible_pairs"] if sol.infeasible_ids else [])
    lat["total"] = (time.perf_counter() - t0) * 1e3
    return Overrides(frame=snap.frame, palette_overrides=po, cues=cues, confusion_pairs=pairs,
                     diagnostics=Diagnostics(stage_latency_ms=lat, solver_tier="exact", fallback_flags=flags))


class Corrector:
    """Signature-cached live corrector: re-solves only when the scene/observer signature changes
    (or every `resolve_every` frames; 0 = never)."""

    def __init__(self, resolve_every: int = 0, max_cache: int = 64, robust: str = "point"):
        if robust not in ROBUST_MODES:
            raise ValueError(f"robust must be one of {ROBUST_MODES}")
        self.robust = robust                  # profile observers: point (posterior mean) | cvar | worst
        self.table = observer_table()
        self.resolve_every, self.max_cache = resolve_every, max_cache
        self.cache: OrderedDict[str, tuple[int, Overrides]] = OrderedDict()
        self.solves = 0
        self.cfg_extra: dict = {}

    def __call__(self, snap: Snapshot, observer: Observer | None = None, posterior: list[Observer] | None = None) -> Overrides:
        """observer: explicit (e.g. profile posterior mean); else looked up by snap.observer_id in the fixed pool.
        posterior: M posterior-sample observers, used when self.robust is cvar|worst."""
        obs = observer or self.table.get(snap.observer_id)
        if obs is None:
            raise KeyError(snap.observer_id)
        risk = self.robust if posterior else "point"
        t0 = time.perf_counter()
        sig = signature(snap, (repr(obs.to_dict()) + risk + repr([o.to_dict() for o in posterior or []])
                               if risk != "point" else repr(obs.to_dict())) if observer else "")
        hit = self.cache.get(sig)
        if hit and not (self.resolve_every and snap.frame - hit[0] >= self.resolve_every):
            self.cache.move_to_end(sig)
            d = hit[1].diagnostics
            lat = {"total": (time.perf_counter() - t0) * 1e3}
            return hit[1].model_copy(update={"frame": snap.frame, "diagnostics": Diagnostics(
                stage_latency_ms=lat, solver_tier="cached", fallback_flags=d.fallback_flags)})
        ov = solve(snap, obs, self.table, self.cfg_extra, posterior, risk)
        self.solves += 1
        self.cache[sig] = (snap.frame, ov)
        while len(self.cache) > self.max_cache:
            self.cache.popitem(last=False)
        return ov


class SelfReportIn(BaseModel):
    """Optional; only shapes the prior."""
    model_config = ConfigDict(extra="forbid")
    type_guess: Literal["protan", "deutan", "tritan", "unsure"] | None = None
    struggle_colors: list[str] = Field(default_factory=list, max_length=12)


class CreateProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    self_report: SelfReportIn | None = None
    config: dict = Field(default_factory=dict)   # CalibConfig fields: seed, mode, target_width, min_trials, max_trials


class AnswerIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trial_id: int
    response: str | float   # "same"|"different" or slider value in [0,1]


NOTE = "Accessibility parameters estimated from your answers; not a medical diagnosis."


class Sessions:
    """Live calibration sessions; rebuilt from the stored trial log (exact replay) after a restart."""

    def __init__(self, store: ProfileStore):
        self.store, self.live, self.lock = store, {}, threading.Lock()

    def get(self, pid: str):
        with self.lock:
            ent = self.live.get(pid)
            if ent is None:
                prof = self.store.get(pid)
                if prof is None:
                    return None, None
                sess = Session.replay(CalibConfig.from_dict(prof["config"]), prof["self_report"], self.store.trials(pid))
                ent = self.live[pid] = (sess, threading.Lock())
            return ent


def create_app(gt_dir: Path | str | None = None, resolve_every: int = 0, profiles_db: Path | str | None = None,
               robust: str | None = None) -> FastAPI:
    app = FastAPI(title="SemanticVision core")
    app.add_middleware(CORSMiddleware, allow_origins=DEV_ORIGINS, allow_methods=["*"], allow_headers=["*"])
    out_dir = Path(gt_dir or os.environ.get("SV_GT_DIR") or GT_DIR)

    @app.get("/health")
    def health():
        return {"ok": True}

    corrector = Corrector(resolve_every, robust=robust or os.environ.get("SV_ROBUST") or "point")
    app.state.corrector = corrector

    @app.get("/observers")
    def observers():
        return {"train": [o.to_dict() for o in corrector.table.values() if o.split == "train"],
                "heldout": [o.to_dict() for o in corrector.table.values() if o.split == "heldout"]}

    store = ProfileStore(profiles_db or os.environ.get("SV_PROFILE_DB") or None)
    sessions = Sessions(store)
    app.state.profiles, app.state.sessions = store, sessions

    @app.post("/correct", response_model=Overrides)
    def post_correct(snap: Snapshot) -> Overrides:
        try:
            if snap.observer_id.startswith(PROFILE_PREFIX):
                pid = snap.observer_id[len(PROFILE_PREFIX):]
                prof = store.get(pid)
                if prof is None:
                    raise KeyError(pid)
                post = (samples_to_observers(prof["samples"], POSTERIOR_M, pid)
                        if corrector.robust != "point" and prof.get("samples") else None)
                return corrector(snap, profile_observer(pid, prof["summary"]), post)
            return corrector(snap)
        except KeyError:
            raise HTTPException(404, f"unknown observer_id {snap.observer_id}")

    # ---- calibration profiles ----
    def _view(prof: dict, sess: Session | None, samples: bool = False) -> dict:
        out = dict(id=prof["id"], status=prof["status"], n_trials=prof["n_trials"], config=prof["config"],
                   self_report=prof["self_report"], summary=prof["summary"], note=NOTE,
                   n_samples=len(next(iter(prof["samples"].values()))) if prof["samples"] else 0)
        if sess is not None:
            out["progress"] = sess.progress()
        if samples:
            out["samples"] = prof["samples"]
        return out

    @app.post("/profiles", status_code=201)
    def post_profile(req: CreateProfile):
        sr = req.self_report.model_dump() if req.self_report else {}
        try:
            cfg = CalibConfig.from_dict(req.config)
            sess = Session(cfg, sr)
        except (ValueError, TypeError) as e:
            raise HTTPException(422, str(e))
        pid = store.create(cfg.to_dict(), sr, sess.summary())
        sessions.live[pid] = (sess, threading.Lock())
        return _view(store.get(pid), sess)

    @app.get("/profiles")
    def list_profiles():
        out = []
        for p in store.list_complete():
            try:
                o = profile_observer(p["id"], p["summary"])
            except (KeyError, TypeError):
                continue
            out.append(dict(id=p["id"], observer_id=o.id, status=p["status"], n_trials=p["n_trials"], created=p["created"],
                            updated=p["updated"], width=p["summary"].get("width"), model=o.model, kind=o.kind,
                            severity=o.severity))
        return out

    @app.get("/profiles/{pid}")
    def get_profile(pid: str, samples: bool = False):
        prof = store.get(pid)
        if prof is None:
            raise HTTPException(404, "unknown profile")
        sess, _ = sessions.get(pid)
        return _view(prof, sess, samples)

    @app.get("/profiles/{pid}/next")
    def profile_next(pid: str):
        sess, lock = sessions.get(pid)
        if sess is None:
            raise HTTPException(404, "unknown profile")
        if store.get(pid)["status"] == "complete":
            raise HTTPException(409, "profile already finished")
        with lock:
            stim = sess.next()
            return {"done": stim is None, "progress": sess.progress(), "stimulus": stim}

    @app.post("/profiles/{pid}/answer")
    def profile_answer(pid: str, a: AnswerIn):
        sess, lock = sessions.get(pid)
        if sess is None:
            raise HTTPException(404, "unknown profile")
        if store.get(pid)["status"] == "complete":
            raise HTTPException(409, "profile already finished")
        with lock:
            try:
                stim, mode, resp = sess.answer(a.trial_id, a.response)
            except KeyError:
                raise HTTPException(409, "no such pending trial (call /next first)")
            except ValueError as e:
                raise HTTPException(422, str(e))
            store.add_trial(pid, sess.n - 1, stim, mode, resp, sess.summary())
            return {"progress": sess.progress(), "summary": sess.summary()}

    @app.post("/profiles/{pid}/finish")
    def profile_finish(pid: str):
        sess, lock = sessions.get(pid)
        if sess is None:
            raise HTTPException(404, "unknown profile")
        with lock:
            store.finish(pid, sess.summary(full=True), sess.cal.samples(500))
        return _view(store.get(pid), sess)

    @app.post("/gt")
    def post_gt(g: GTFrame):
        if g.width * g.height > MAX_PIXELS:
            raise HTTPException(413, "frame too large")
        arrs = {}
        for name, b64 in g.buffers.items():
            if name not in BUFFERS:
                raise HTTPException(422, f"unknown buffer {name}")
            try:
                raw = base64.b64decode(b64, validate=True)
            except Exception:
                raise HTTPException(422, f"bad base64 in {name}")
            if len(raw) != g.width * g.height * BUFFERS[name]:
                raise HTTPException(422, f"buffer {name}: wrong size")
            arrs[name] = np.frombuffer(raw, np.uint8).reshape(g.height, g.width, 4)[::-1]  # top row first
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"gt_{g.frame:06d}.npz"
        np.savez_compressed(p, **arrs)
        (out_dir / f"gt_{g.frame:06d}.json").write_text(
            json.dumps({"frame": g.frame, "width": g.width, "height": g.height, "legend": g.legend}))
        return {"ok": True, "path": str(p)}

    return app


app = create_app()
