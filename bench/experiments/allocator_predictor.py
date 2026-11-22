"""Kalman entry predictor on synthetic streams (ground truth = enter_view events in the snapshot). An enter event is 'precomputed'
if the object was flagged >= 1 frame before it appeared. Precision: flagged (object, frame) pairs that really enter within
horizon + 0.25 s. Baseline (no predictor) precomputes nothing by construction; lead time = seconds between first flag and entry."""
from __future__ import annotations
import numpy as np
from bench.registry import experiment
from adapters.synthetic import SynthConfig, generate
from core.allocator import EntryPredictor


@experiment("allocator_predictor")
def run(params, seed):
    horizon = float(params.get("horizon", 0.5))
    cfg = SynthConfig(n_objects=int(params.get("n_objects", 20)), n_frames=int(params.get("n_frames", 300)),
                      motion=float(params.get("motion", 0.3)), margin=0.15)
    pred = EntryPredictor(horizon=horizon)
    first_flag, enters, flags = {}, [], []
    for snap in generate(cfg, seed):
        for oid in pred.update(snap):
            flags.append((oid, snap.t)); first_flag.setdefault(oid, snap.t)
        for e in snap.events:
            if e.type == "enter_view":
                enters.append((e.object_id, snap.t, first_flag.get(e.object_id)))
                first_flag.pop(e.object_id, None)
    hit = [(o, t, f) for o, t, f in enters if f is not None and t - f > 0]
    entry_t = {}
    for o, t, _ in enters:
        entry_t.setdefault(o, []).append(t)
    tp = sum(any(0 <= et - t <= horizon + 0.25 for et in entry_t.get(o, [])) for o, t in flags)
    return {"kalman": dict(recall=len(hit) / max(len(enters), 1), precision=tp / max(len(flags), 1),
                           lead_s=float(np.mean([t - f for _, t, f in hit])) if hit else 0.0, n_enter=len(enters)),
            "none": dict(recall=0.0, precision=0.0, lead_s=0.0, n_enter=len(enters))}
