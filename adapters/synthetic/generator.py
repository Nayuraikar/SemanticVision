"""Headless synthetic scene generator -> protocol Snapshots. Seeded, fixed timestep, engine-free."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Iterator
import numpy as np
from protocol import Snapshot, SnapshotBuilder

PALETTES = {
    "random": None,
    # near red/green & blue/purple: stresses CVD confusion
    "confusable": [(0.80, 0.10, 0.05), (0.15, 0.55, 0.05), (0.55, 0.40, 0.05), (0.10, 0.30, 0.75),
                   (0.45, 0.15, 0.65), (0.75, 0.45, 0.10), (0.40, 0.45, 0.10)],
    "traffic": [(0.85, 0.05, 0.05), (0.95, 0.65, 0.02), (0.05, 0.60, 0.15), (0.5, 0.5, 0.5), (0.1, 0.1, 0.1)],
    "dashboard": [(0.9, 0.05, 0.05), (0.05, 0.8, 0.1), (0.95, 0.7, 0.05), (0.1, 0.4, 0.9)],
}
LIGHTING = {  # ambient, dominant, direction, exposure
    "day": ((0.35, 0.40, 0.50), (1.0, 0.95, 0.85), (0.3, 1.0, 0.2), 1.0),
    "night": ((0.02, 0.03, 0.06), (0.10, 0.12, 0.20), (0.0, 1.0, 0.1), 2.5),
    "fog": ((0.55, 0.57, 0.60), (0.35, 0.35, 0.36), (0.0, 1.0, 0.0), 1.0),
    "rain": ((0.20, 0.22, 0.27), (0.45, 0.47, 0.52), (0.1, 1.0, 0.1), 1.2),
}
CLASSES = ["car", "light", "sign", "pedestrian", "indicator", "wall"]
STATES = {"light": ["red", "amber", "green"], "indicator": ["off", "ok", "warn", "fault"]}
TAGS = {"car": ["hazard"], "pedestrian": ["hazard"], "light": ["signal"], "indicator": ["signal"],
        "sign": ["info"], "wall": []}
EMISSIVE = {"light", "indicator"}
LIGHT_RGB = {"red": (0.9, 0.03, 0.03), "amber": (0.95, 0.6, 0.02), "green": (0.03, 0.8, 0.12)}


@dataclass
class SynthConfig:
    n_objects: int = 12
    n_frames: int = 120
    dt: float = 1 / 30
    palette: str = "confusable"
    lighting: str = "day"          # key of LIGHTING, or "cycle" (day->night drift w/ lighting_change events)
    motion: float = 0.15           # max speed (viewport units / s); 0 = static
    state_rate: float = 0.02       # per-object per-second-ish state_change probability scale
    margin: float = 0.15           # objects may drift this far outside viewport (enter/exit events)
    observer_id: str = "obs-0"
    budget_ms: float = 8.0

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self):
        return asdict(self)


def _r6(x):
    return [round(float(v), 6) for v in x]


def _clip01(x):
    return np.clip(x, 0.0, 1.0)


class SyntheticScene:
    def __init__(self, cfg: SynthConfig, seed: int):
        self.cfg, self.seed = cfg, int(seed)
        r = self.rng = np.random.default_rng([self.seed, 0x5EED])
        n = cfg.n_objects
        pal = PALETTES[cfg.palette]
        self.cls = [CLASSES[int(i)] for i in r.integers(len(CLASSES), size=n)]
        if pal is None:
            self.albedo = r.uniform(0.02, 0.95, size=(n, 3))
        else:
            base = np.array([pal[int(i)] for i in r.integers(len(pal), size=n)])
            self.albedo = _clip01(base + r.normal(0, 0.03, size=(n, 3)))
        self.size = r.uniform(0.04, 0.2, size=(n, 2))
        self.pos = r.uniform(0, 1, size=(n, 2))
        ang = r.uniform(0, 2 * np.pi, size=n)
        spd = r.uniform(0, 1, size=n) * cfg.motion
        self.vel = np.stack([np.cos(ang), np.sin(ang)], 1) * spd[:, None]
        self.depth = r.uniform(2.0, 40.0, size=n)
        self.state = [STATES.get(c, ["default"])[0] for c in self.cls]
        self.emis_base = np.array([self.albedo[i] * r.uniform(2, 6) if self.cls[i] in EMISSIVE else np.zeros(3)
                                   for i in range(n)])
        self.visible = [self._vis(i) for i in range(n)]
        self.cool = {}  # collision cooldown per pair
        self.base_light = cfg.lighting if cfg.lighting != "cycle" else "day"
        self.lkey_prev = None

    def _bbox(self, i):
        return self.pos[i, 0], self.pos[i, 1], self.size[i, 0], self.size[i, 1]

    def _vis(self, i):
        x, y, w, h = self._bbox(i)
        return x + w > 0 and x < 1 and y + h > 0 and y < 1

    def _lighting(self, f):
        c = self.cfg
        if c.lighting != "cycle":
            return LIGHTING[c.lighting], c.lighting
        ph = (f / max(c.n_frames - 1, 1))
        a, b = LIGHTING["day"], LIGHTING["night"]
        mix = lambda u, v: tuple(float(p * (1 - ph) + q * ph) for p, q in zip(u, v))
        ex = a[3] * (1 - ph) + b[3] * ph
        return (mix(a[0], b[0]), mix(a[1], b[1]), mix(a[2], b[2]), ex), f"cycle{int(ph * 4)}"

    def step(self, f: int) -> Snapshot:
        c, r = self.cfg, self.rng
        n = c.n_objects
        events = []
        if f > 0:
            self.pos += self.vel * c.dt
            lo, hi = -c.margin - self.size, 1 + c.margin
            for ax in (0, 1):
                lo_a, hi_a = lo[:, ax], hi
                m = self.pos[:, ax] < lo_a
                self.pos[m, ax] = lo_a[m]; self.vel[m, ax] *= -1
                m = self.pos[:, ax] > hi_a
                self.pos[m, ax] = hi_a; self.vel[m, ax] *= -1
        # visibility
        for i in range(n):
            v = self._vis(i)
            if v != self.visible[i]:
                events.append(("enter_view" if v else "exit_view", f"o{i}", {}))
                self.visible[i] = v
        # collisions (bbox overlap, both visible & moving, cooldown)
        for k in list(self.cool):
            self.cool[k] -= 1
            if self.cool[k] <= 0:
                del self.cool[k]
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) in self.cool or not (self.visible[i] and self.visible[j]):
                    continue
                if abs(self.depth[i] - self.depth[j]) > 8 or not (np.any(self.vel[i]) or np.any(self.vel[j])):
                    continue
                (xi, yi, wi, hi_), (xj, yj, wj, hj) = self._bbox(i), self._bbox(j)
                if xi < xj + wj and xj < xi + wi and yi < yj + hj and yj < yi + hi_:
                    self.cool[(i, j)] = 30
                    self.vel[[i, j]] = self.vel[[j, i]]
                    events.append(("collision", f"o{i}", {"other": f"o{j}"}))
                    for q in (i, j):
                        if self.cls[q] in ("car", "pedestrian"):
                            self.state[q] = "collided"
                            events.append(("state_change", f"o{q}", {"to": "collided"}))
        # state changes
        p = c.state_rate
        for i in range(n):
            st = STATES.get(self.cls[i])
            if st and r.random() < p:
                new = st[int(r.integers(len(st)))]
                if new != self.state[i]:
                    self.state[i] = new
                    events.append(("state_change", f"o{i}", {"to": new}))
        # lighting
        (amb, dom, dr, ex), lkey = self._lighting(f)
        if self.lkey_prev is not None and lkey != self.lkey_prev:
            events.append(("lighting_change", None, {"preset": lkey}))
        self.lkey_prev = lkey
        b = SnapshotBuilder(c.observer_id, c.budget_ms, frame=f, t=round(f * c.dt, 9))
        b.ambient_dominant(_r6(amb), _r6(dom), _r6(dr), exposure=round(float(ex), 6))
        for i in range(n):
            cl, st = self.cls[i], self.state[i]
            alb = self.albedo[i]
            emi = self.emis_base[i] * (0.0 if st == "off" else 1.0)
            if cl == "light":
                emi = np.array(LIGHT_RGB[st]) * 4.0
            x, y, w, h = self._bbox(i)
            sc = 1.0 / (1.0 + self.depth[i] / 10.0)
            w2, h2 = w * sc * 1.5, h * sc * 1.5
            cx0, cy0 = max(x, 0), max(y, 0)
            area = max(0.0, min(x + w2, 1) - cx0) * max(0.0, min(y + h2, 1) - cy0)
            b.add_object(f"o{i}", cl, _r6(alb), bbox=(round(float(x), 6), round(float(y), 6), round(float(w2), 6), round(float(h2), 6)),
                         screen_area=round(float(min(area, 1.0)), 6), depth=round(float(self.depth[i]), 6), state=st,
                         emissive_rgb=_r6(emi), velocity=_r6(self.vel[i]), task_tags=TAGS[cl])
        for t, oid, data in events:
            b.add_event(t, oid, **data)
        return b.build()


def generate(cfg: SynthConfig | None, seed: int) -> Iterator[Snapshot]:
    cfg = cfg or SynthConfig()
    s = SyntheticScene(cfg, seed)
    for f in range(cfg.n_frames):
        yield s.step(f)
