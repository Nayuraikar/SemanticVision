"""Constant-velocity Kalman predictor on bbox centres (state x,y,vx,vy; measures position and the exported velocity).
Flags objects that are off-screen now but whose predicted bbox intersects the viewport within `horizon` seconds,
so their correction can be precomputed before they enter view."""
from __future__ import annotations
import numpy as np


class _KF:
    def __init__(self, z, v, q, r_pos, r_vel):
        self.x = np.array([z[0], z[1], v[0], v[1]], float)
        self.P = np.diag([r_pos, r_pos, r_vel, r_vel]) ** 2
        self.q, self.R = q, np.diag([r_pos, r_pos, r_vel, r_vel]) ** 2

    def step(self, z, v, dt):
        F = np.eye(4); F[0, 2] = F[1, 3] = dt
        G = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.q ** 2 * (G @ G.T)
        y = np.array([z[0], z[1], v[0], v[1]]) - self.x
        S = self.P + self.R
        K = self.P @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K) @ self.P


class EntryPredictor:
    def __init__(self, horizon=0.5, q=0.5, r_pos=0.01, r_vel=0.05, margin=0.0):
        self.horizon, self.q, self.r_pos, self.r_vel, self.margin = horizon, q, r_pos, r_vel, margin
        self.kf: dict = {}
        self.t = None

    def update(self, snap):
        """-> {id: predicted bbox (x,y,w,h) at t+horizon} for objects off-screen now that will intersect the viewport."""
        t = float(snap.t)
        dt = 1 / 30 if self.t is None else max(t - self.t, 1e-3)
        self.t = t
        out = {}
        for o in snap.objects:
            b = o.bbox
            c = np.array([b.x + b.w / 2, b.y + b.h / 2])
            v = np.asarray(o.velocity, float)
            f = self.kf.get(o.id)
            if f is None:
                self.kf[o.id] = f = _KF(c, v, self.q, self.r_pos, self.r_vel)
            else:
                f.step(c, v, dt)
            if self._inside(b.x, b.y, b.w, b.h):
                continue
            px, py = f.x[:2] + f.x[2:] * self.horizon
            if self._inside(px - b.w / 2, py - b.h / 2, b.w, b.h):
                out[o.id] = (float(px - b.w / 2), float(py - b.h / 2), b.w, b.h)
        return out

    def _inside(self, x, y, w, h):
        m = self.margin
        return x + w > -m and x < 1 + m and y + h > -m and y < 1 + m
