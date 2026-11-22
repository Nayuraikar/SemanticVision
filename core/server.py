"""FastAPI service. /correct: Snapshot -> Overrides (identity stub). /gt: dev-only ground-truth sink
(training/scoring only, never read at runtime)."""
from __future__ import annotations
import base64
import json
import os
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from protocol.models import Diagnostics, Overrides, PaletteOverride, Snapshot

DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
GT_DIR = Path("bench/logs/gt")
MAX_PIXELS = 4096 * 4096
BUFFERS = {"instance": 4, "class": 4, "state": 4, "color": 4}  # name -> bytes/pixel (RGBA8)


class GTFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frame: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    legend: dict = Field(default_factory=dict)  # {"instance": {id: idx}, "class": {...}, "state": {...}}
    buffers: dict[str, str]  # name -> base64 RGBA8 (row 0 = bottom, as read from WebGL)


def correct(snap: Snapshot) -> Overrides:
    """Identity stub: every object keeps its colours."""
    t0 = time.perf_counter()
    po = [PaletteOverride(id=o.id, albedo=o.albedo_rgb, emissive=o.emissive_rgb) for o in snap.objects]
    dt = (time.perf_counter() - t0) * 1e3
    return Overrides(frame=snap.frame, palette_overrides=po,
                     diagnostics=Diagnostics(stage_latency_ms={"total": dt}, solver_tier="none",
                                             fallback_flags=["identity_stub"]))


def create_app(gt_dir: Path | str | None = None) -> FastAPI:
    app = FastAPI(title="SemanticVision core")
    app.add_middleware(CORSMiddleware, allow_origins=DEV_ORIGINS, allow_methods=["*"], allow_headers=["*"])
    out_dir = Path(gt_dir or os.environ.get("SV_GT_DIR") or GT_DIR)

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.post("/correct", response_model=Overrides)
    def post_correct(snap: Snapshot) -> Overrides:
        return correct(snap)

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
