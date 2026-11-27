"""Per-object importance features + scene geometry from Snapshots (engine-free, no perception/solver imports).
Tracker is stateful (state-change rate, time since change); Geometry feeds the independent task observer."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

CLASSES = ("car", "light", "sign", "pedestrian", "indicator", "wall")
STATE_CRIT = {"collided": 1.0, "damaged": 1.0, "fault": 1.0, "red": 0.9, "warn": 0.8, "amber": 0.7, "green": 0.3, "ok": 0.2,
              "off": 0.1, "default": 0.2}
FEATURE_NAMES = [f"cls_{c}" for c in CLASSES] + ["state_crit", "change_rate", "speed", "collision", "log_area",
                                                 "depth", "hazard", "signal", "info", "ecc"]
IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}
FOV_DEG = 60.0          # horizontal field of view mapping viewport units -> degrees (same as Scene.from_snapshot)
RATE_TAU = 2.0          # s, decay of the state-change-rate trace


@dataclass
class Geometry:
    ecc: np.ndarray         # deg from fixation (viewport centre)
    size: np.ndarray        # deg, sqrt(bbox area)
    crowd: np.ndarray       # count of other objects within Bouma spacing (0.5*ecc, floor 0.5 deg)
    ttc: np.ndarray         # s to bbox contact with the nearest closing object (inf if none)
    change_age: np.ndarray  # s since last state change (inf if never)
    n: int


def _boxes(snap):
    b = np.array([(o.bbox.x, o.bbox.y, o.bbox.w, o.bbox.h) for o in snap.objects], float).reshape(-1, 4)
    return b, b[:, :2] + b[:, 2:] / 2


def geometry(snap, change_age=None, fov=FOV_DEG) -> Geometry:
    n = len(snap.objects)
    b, c = _boxes(snap)
    ecc = np.hypot(c[:, 0] - 0.5, c[:, 1] - 0.5) * fov
    size = np.sqrt(np.maximum(b[:, 2] * b[:, 3], 1e-12)) * fov
    d = np.hypot(c[:, None, 0] - c[None, :, 0], c[:, None, 1] - c[None, :, 1]) * fov
    np.fill_diagonal(d, np.inf)
    crowd = (d < np.maximum(0.5 * ecc, 0.5)[:, None]).sum(1)
    ttc = np.full(n, np.inf)
    if n > 1:
        v = np.array([o.velocity for o in snap.objects], float)
        x1, y1 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
        gx = np.maximum(b[:, None, 0] - x1[None, :], b[None, :, 0] - x1[:, None])
        gy = np.maximum(b[:, None, 1] - y1[None, :], b[None, :, 1] - y1[:, None])
        gap = np.maximum(np.maximum(gx, gy), 0.0)
        dc = c[:, None] - c[None]
        nrm = np.maximum(np.hypot(dc[..., 0], dc[..., 1]), 1e-9)
        rv = v[:, None] - v[None]
        closing = -(rv * dc).sum(-1) / nrm
        t = np.where(closing > 1e-6, gap / np.maximum(closing, 1e-6), np.inf)
        np.fill_diagonal(t, np.inf)
        ttc = t.min(1)
    ca = np.full(n, np.inf) if change_age is None else np.asarray(change_age, float)
    return Geometry(ecc, size, crowd, ttc, ca, n)


class Tracker:
    """Stateful feature extractor. update(snap) -> (X (N,F), Geometry). Frames may be sparse (state change is detected
    between consecutive calls; rate decays with the elapsed snapshot time)."""

    def __init__(self):
        self.state: dict = {}
        self.rate: dict = {}
        self.last_change: dict = {}
        self.t = None
        self.frame = None

    def update(self, snap):
        t = float(snap.t)
        dt = 0.0 if self.t is None else max(t - self.t, 0.0)
        ids = [o.id for o in snap.objects]
        ev = {e.object_id for e in snap.events if e.type == "state_change" and e.object_id}
        age = np.full(len(ids), np.inf)
        for k, o in enumerate(snap.objects):
            r = self.rate.get(o.id, 0.0) * np.exp(-dt / RATE_TAU)
            prev = self.state.get(o.id)
            if (prev is not None and prev != o.state) or o.id in ev:
                r += 1.0 / RATE_TAU
                self.last_change[o.id] = t
            self.rate[o.id], self.state[o.id] = r, o.state
            if o.id in self.last_change:
                age[k] = t - self.last_change[o.id]
        self.t, self.frame = t, snap.frame
        g = geometry(snap, age)
        return self._matrix(snap, g), g

    def _matrix(self, snap, g):
        n = g.n
        X = np.zeros((n, len(FEATURE_NAMES)))
        for k, o in enumerate(snap.objects):
            if o.cls in CLASSES:
                X[k, CLASSES.index(o.cls)] = 1.0
            X[k, IDX["state_crit"]] = STATE_CRIT.get(o.state, STATE_CRIT["default"])
            X[k, IDX["change_rate"]] = self.rate[o.id]
            X[k, IDX["speed"]] = np.hypot(*o.velocity)
            X[k, IDX["log_area"]] = np.log(max(o.bbox.w * o.bbox.h, 1e-6))
            X[k, IDX["depth"]] = o.depth
            tags = set(o.task_tags)
            X[k, IDX["hazard"]], X[k, IDX["signal"]], X[k, IDX["info"]] = ("hazard" in tags, "signal" in tags, "info" in tags)
        X[:, IDX["collision"]] = np.where(np.isfinite(g.ttc), np.exp(-g.ttc / 1.0), 0.0)
        X[:, IDX["ecc"]] = g.ecc
        return X
