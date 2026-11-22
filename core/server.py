"""FastAPI service. /correct: Snapshot -> Overrides (real solver, signature-cached); /observers. /gt: dev-only ground-truth sink
(training/scoring only, never read at runtime)."""
from __future__ import annotations
import base64
import hashlib
from collections import OrderedDict
import json
import os
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from core.observers import Observer, heldout_observers, train_observers
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


def signature(snap: Snapshot) -> str:
    h = hashlib.sha1(snap.observer_id.encode())
    for o in snap.objects:
        h.update(repr((o.id, o.cls, o.state, [round(x, 3) for x in o.albedo_rgb], [round(x, 3) for x in o.emissive_rgb],
                       round(o.bbox.x, 2), round(o.bbox.y, 2), round(o.bbox.w, 2), round(o.bbox.h, 2),
                       tuple(o.task_tags))).encode())
    return h.hexdigest()


def solve(snap: Snapshot, obs: Observer, table: dict[str, Observer], cfg_extra: dict | None = None) -> Overrides:
    t0 = time.perf_counter()
    lat: dict[str, float] = {}
    if not snap.objects:
        return Overrides(frame=snap.frame, diagnostics=Diagnostics(stage_latency_ms={"total": 0.0}, solver_tier="none"))
    scene = Scene.from_snapshot(snap)
    t1 = time.perf_counter(); lat["scene"] = (t1 - t0) * 1e3
    cfg = SolverConfig(budget_ms=float(snap.budget_ms), **{**LIVE_CFG, **(cfg_extra or {})})
    sol = solver_correct(scene, solver_observers(obs, table), cfg, seed=0)
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
    po = [PaletteOverride(id=o.id, albedo=tuple(float(np.clip(x, 0, 1)) for x in sol.colors[i]), emissive=o.emissive_rgb)
          for i, o in enumerate(snap.objects)]
    flags = (["cue_unresolved"] if sol.cue_unresolved else []) + (["infeasible_pairs"] if sol.infeasible_ids else [])
    lat["total"] = (time.perf_counter() - t0) * 1e3
    return Overrides(frame=snap.frame, palette_overrides=po, cues=cues, confusion_pairs=pairs,
                     diagnostics=Diagnostics(stage_latency_ms=lat, solver_tier="exact", fallback_flags=flags))


class Corrector:
    """Signature-cached live corrector: re-solves only when the scene/observer signature changes
    (or every `resolve_every` frames; 0 = never)."""

    def __init__(self, resolve_every: int = 0, max_cache: int = 64):
        self.table = observer_table()
        self.resolve_every, self.max_cache = resolve_every, max_cache
        self.cache: OrderedDict[str, tuple[int, Overrides]] = OrderedDict()
        self.solves = 0
        self.cfg_extra: dict = {}

    def __call__(self, snap: Snapshot) -> Overrides:
        obs = self.table.get(snap.observer_id)
        if obs is None:
            raise KeyError(snap.observer_id)
        t0 = time.perf_counter()
        sig = signature(snap)
        hit = self.cache.get(sig)
        if hit and not (self.resolve_every and snap.frame - hit[0] >= self.resolve_every):
            self.cache.move_to_end(sig)
            d = hit[1].diagnostics
            lat = {"total": (time.perf_counter() - t0) * 1e3}
            return hit[1].model_copy(update={"frame": snap.frame, "diagnostics": Diagnostics(
                stage_latency_ms=lat, solver_tier="cached", fallback_flags=d.fallback_flags)})
        ov = solve(snap, obs, self.table, self.cfg_extra)
        self.solves += 1
        self.cache[sig] = (snap.frame, ov)
        while len(self.cache) > self.max_cache:
            self.cache.popitem(last=False)
        return ov


def create_app(gt_dir: Path | str | None = None, resolve_every: int = 0) -> FastAPI:
    app = FastAPI(title="SemanticVision core")
    app.add_middleware(CORSMiddleware, allow_origins=DEV_ORIGINS, allow_methods=["*"], allow_headers=["*"])
    out_dir = Path(gt_dir or os.environ.get("SV_GT_DIR") or GT_DIR)

    @app.get("/health")
    def health():
        return {"ok": True}

    corrector = Corrector(resolve_every)
    app.state.corrector = corrector

    @app.get("/observers")
    def observers():
        return {"train": [o.to_dict() for o in corrector.table.values() if o.split == "train"],
                "heldout": [o.to_dict() for o in corrector.table.values() if o.split == "heldout"]}

    @app.post("/correct", response_model=Overrides)
    def post_correct(snap: Snapshot) -> Overrides:
        try:
            return corrector(snap)
        except KeyError:
            raise HTTPException(404, f"unknown observer_id {snap.observer_id}")

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
