"""Lit synthetic worlds + deterministic software renderer (ground truth for K3, SPEC 5.2).
Pinhole camera at the origin looking down -z; y up; ground plane; sphere objects (emissive lamps included).
Truth renderer: per-pixel direct lighting with shadow rays -- sampled area/point dominant light + cosine-weighted
sky/ground environment visibility rays (spheres occlude). Headless, seeded, numpy only; Three.js GT is an optional cross-check."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
from core.lighting import sh as S
from protocol import Snapshot, SnapshotBuilder
from protocol.models import DominantLight, SHLighting

GROUND_Y = -1.8
GROUND_ALBEDO = np.array([0.35, 0.33, 0.30])


@dataclass(frozen=True)
class LightSpec:
    sky: tuple             # env radiance towards +y
    ground: tuple          # env radiance towards -y
    dom_rgb: tuple         # normalised irradiance on a facing surface (point lamp: intensity at ref distance)
    dom_dir: tuple         # towards the light (directional)
    dom_pos: tuple | None  # point lamp position
    exposure: float
    soft: float            # angular radius (rad) of directional light / radius (m) of lamp


LIGHT_PRESETS = {
    "day": LightSpec((0.40, 0.48, 0.65), (0.15, 0.14, 0.12), (0.75, 0.70, 0.62), (0.3, 1.0, 0.2), None, 1.0, 0.05),
    "night": LightSpec((0.015, 0.02, 0.045), (0.005, 0.006, 0.01), (0.06, 0.075, 0.14), (-0.3, 1.0, 0.4), None, 3.0, 0.1),
    "fog": LightSpec((0.55, 0.57, 0.60), (0.45, 0.46, 0.48), (0.12, 0.12, 0.125), (0.0, 1.0, 0.0), None, 1.0, 0.3),
    "tinted": LightSpec((0.20, 0.13, 0.05), (0.10, 0.06, 0.02), (0.85, 0.48, 0.16), (-0.4, 1.0, 0.3), None, 1.0, 0.05),
    "street": LightSpec((0.02, 0.025, 0.05), (0.01, 0.01, 0.02), (0.9, 0.7, 0.35), (0.0, 1.0, 0.0), (0.0, 6.0, -10.0), 1.5, 0.4),
    "dusk": LightSpec((0.30, 0.22, 0.30), (0.10, 0.07, 0.08), (0.80, 0.45, 0.25), (0.9, 0.35, 0.2), None, 1.2, 0.04),
}
TRAIN_LIGHTS = ("day", "night", "fog")
HELDOUT_LIGHTS = ("tinted", "street", "dusk")
SIGNAL_RGB = {"red": (0.9, 0.03, 0.03), "amber": (0.95, 0.6, 0.02), "green": (0.03, 0.8, 0.12), "white": (0.9, 0.9, 0.85),
              "blue": (0.05, 0.2, 0.9)}
ANNUNCIATOR = {"ok": SIGNAL_RGB["green"], "warn": SIGNAL_RGB["amber"], "fault": SIGNAL_RGB["red"], "off": (0.05, 0.05, 0.05)}
CONFUSABLE = [(0.80, 0.10, 0.05), (0.15, 0.55, 0.05), (0.55, 0.40, 0.05), (0.10, 0.30, 0.75), (0.45, 0.15, 0.65),
              (0.75, 0.45, 0.10), (0.40, 0.45, 0.10)]
WORLD_PALETTES = ("surfaces", "signals", "annunciator", "mixed")   # emissive small-field cases: signals, annunciator, mixed


@dataclass
class WorldConfig:
    n_objects: int = 12
    palette: str = "mixed"
    lighting: str = "day"
    width: int = 112
    height: int = 112
    n_env: int = 16
    n_light: int = 8
    n_big: int = 2
    shadow_frac: float = 0.5   # share of reflective objects placed in a big occluder's shadow
    fov: float = 60.0
    min_pixels: int = 2
    observer_id: str = "obs-0"
    budget_ms: float = 8.0

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self):
        return asdict(self)


def _u(v):
    v = np.asarray(v, float)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _r6(x):
    return [round(float(v), 6) for v in np.ravel(x)]


def sphere_hits(o, d, cen, rad):
    """Nearest positive hit of rays (o (M,3), d (M,3)) on spheres -> (t (M,), idx (M,)), t=inf if none."""
    oc = o[:, None, :] - cen[None]
    b = (oc * d[:, None, :]).sum(-1)
    c = (oc * oc).sum(-1) - rad[None] ** 2
    disc = b * b - c
    t = -b - np.sqrt(np.maximum(disc, 0))
    t = np.where((disc > 0) & (t > 1e-5), t, np.inf)
    i = t.argmin(1)
    return t[np.arange(len(o)), i], i


def occluded(o, d, cen, rad, tmax=None, skip=None):
    """Any sphere hit along the ray within tmax; `skip` (M,) sphere index to ignore (-1 none)."""
    oc = o[:, None, :] - cen[None]
    b = (oc * d[:, None, :]).sum(-1)
    c = (oc * oc).sum(-1) - rad[None] ** 2
    disc = b * b - c
    t = -b - np.sqrt(np.maximum(disc, 0))
    hit = (disc > 0) & (t > 1e-5)
    if tmax is not None:
        hit &= t < np.asarray(tmax)[:, None]
    if skip is not None:
        m = np.asarray(skip) >= 0
        hit[np.flatnonzero(m), np.asarray(skip)[m]] = False
    return hit.any(1)


@dataclass
class Render:
    inst: np.ndarray       # (H,W) object index, -1 sky, -2 ground
    rgb: np.ndarray        # (H,W,3) HDR linear display radiance
    E: np.ndarray          # (H,W,3) normalised irradiance (0 on sky)
    vis: np.ndarray        # (H,W) dominant-light visibility (nan on sky)
    white_rgb: np.ndarray  # mean irradiance over surface pixels (what the observer adapts to)
    bg_lum: float          # mean luminance of background (non-object) pixels, clipped display


class LitWorld:
    def __init__(self, cfg: WorldConfig, seed: int):
        self.cfg, self.seed = cfg, int(seed)
        self.spec = LIGHT_PRESETS[cfg.lighting]
        self.tanf = np.tan(np.radians(cfg.fov) / 2)
        self.rng = np.random.default_rng([self.seed, 0x1117])
        self._build()

    # ---- scene -----------------------------------------------------------------------------
    @property
    def centroid(self):
        return np.array([0.0, 0.0, -12.0])

    def to_light(self, p):
        """Unit direction towards the dominant light and its effective intensity multiplier at points p (...,3)."""
        sp = self.spec
        if sp.dom_pos is None:
            d = np.broadcast_to(_u(sp.dom_dir), np.shape(p)).copy()
            return d, np.ones(np.shape(p)[:-1]), np.full(np.shape(p)[:-1], np.inf)
        v = np.asarray(sp.dom_pos, float) - p
        dist = np.linalg.norm(v, axis=-1)
        ref = np.linalg.norm(np.asarray(sp.dom_pos) - self.centroid)
        return v / dist[..., None], (ref / dist) ** 2, dist

    def _place(self, r, rad, zr=(-18, -6)):
        z = r.uniform(*zr)
        h = self.tanf * -z
        y = max(r.uniform(-0.8, 0.8) * h, GROUND_Y + rad + 0.01)
        return np.array([r.uniform(-0.8, 0.8) * h, y, z])

    def _build(self):
        c, r = self.cfg, self.rng
        objs = []   # dict(cls, state, center, rad, albedo, emis)

        def add(cls, state, center, rad, albedo, emis=(0, 0, 0)):
            objs.append(dict(cls=cls, state=state, center=np.asarray(center, float), rad=float(rad),
                             albedo=np.clip(np.asarray(albedo, float), 0, 1), emis=np.asarray(emis, float)))

        big = []
        for _ in range(c.n_big):
            rad = r.uniform(1.2, 2.2)
            ctr = self._place(r, rad, (-18, -9))
            add("wall", "default", ctr, rad, r.uniform(0.25, 0.6, 3))
            big.append(objs[-1])
        n_left = max(c.n_objects - c.n_big, 0)
        pal = c.palette
        n_sig = {"surfaces": 0, "signals": n_left, "annunciator": 0, "mixed": n_left // 2}[pal]
        n_ann = {"surfaces": 0, "signals": 0, "annunciator": min(n_left, 6), "mixed": 0}[pal]
        if pal == "mixed":
            n_ann = 0
        n_surf = n_left - n_sig - n_ann
        if pal == "annunciator":
            n_surf = n_left - n_ann
        # reflective objects (confusable colours), some in a big occluder's shadow
        for _ in range(n_surf):
            rad = r.uniform(0.35, 0.9)
            alb = np.array(CONFUSABLE[int(r.integers(len(CONFUSABLE)))]) + r.normal(0, 0.03, 3)
            ctr = self._place(r, rad)
            if big and r.random() < c.shadow_frac:
                b = big[int(r.integers(len(big)))]
                ld = self.to_light(b["center"])[0]
                perp = _u(np.cross(ld, r.normal(size=3)))
                cand = b["center"] - ld * (b["rad"] + rad + r.uniform(0.0, 0.8) * b["rad"]) + perp * r.uniform(-0.7, 0.7) * b["rad"]
                cand[1] = max(cand[1], GROUND_Y + rad + 0.01)
                if -cand[2] > 4 and abs(cand[0]) < 0.9 * self.tanf * -cand[2] and abs(cand[1]) < 0.9 * self.tanf * -cand[2]:
                    ctr = cand
            add(["car", "sign", "pedestrian"][int(r.integers(3))], "default", ctr, rad, alb)
        # signal lamps: small emissive fields (housing albedo dark, emission HDR)
        for _ in range(n_sig):
            st = list(SIGNAL_RGB)[int(r.integers(len(SIGNAL_RGB)))]
            rad = r.uniform(0.08, 0.25)
            ctr = self._place(r, rad, (-20, -8))
            add("light", st, ctr, rad, np.array(SIGNAL_RGB[st]) * 0.15, np.array(SIGNAL_RGB[st]) * r.uniform(2.5, 5.0))
        # annunciator panel: adjacent lamps of ok/warn/fault/off states
        if n_ann:
            rad = r.uniform(0.1, 0.18)
            o0 = self._place(r, rad, (-14, -7))
            for k in range(n_ann):
                st = list(ANNUNCIATOR)[int(r.integers(len(ANNUNCIATOR)))]
                col = np.array(ANNUNCIATOR[st])
                ctr = o0 + np.array([(k % 3) * 3.0 * rad, (k // 3) * 3.0 * rad, 0.0])
                ctr[1] = max(ctr[1], GROUND_Y + rad + 0.01)
                add("indicator", st, ctr, rad, col * 0.15 if st != "off" else col,
                    col * r.uniform(2.5, 5.0) if st != "off" else (0, 0, 0))
        self.objs = objs
        self.cen = np.array([o["center"] for o in objs])
        self.rad = np.array([o["rad"] for o in objs])
        self.albedo = np.array([o["albedo"] for o in objs])
        self.emis = np.array([o["emis"] for o in objs])

    # ---- lighting description (what the engine would export) ---------------------------------
    def env(self, d):
        t = (0.5 + 0.5 * d[..., 1])[..., None]
        return np.asarray(self.spec.ground) * (1 - t) + np.asarray(self.spec.sky) * t

    def lighting(self) -> SHLighting:
        sp = self.spec
        ref = float(np.linalg.norm(np.asarray(sp.dom_pos) - self.centroid)) if sp.dom_pos is not None else 1.0
        nd = _u(sp.dom_dir) if sp.dom_pos is None else _u(np.asarray(sp.dom_pos) - self.centroid)
        coeffs = S.project(self.env) + S.delta_coeffs(nd, sp.dom_rgb)
        dom = DominantLight(rgb=tuple(sp.dom_rgb), direction=tuple(float(x) for x in nd),
                            position=None if sp.dom_pos is None else tuple(sp.dom_pos), ref_dist=ref)
        return SHLighting(coeffs=[tuple(float(x) for x in row) for row in coeffs], dominant=dom, exposure=sp.exposure)

    # ---- renderer ---------------------------------------------------------------------------
    def render(self) -> Render:
        c, sp = self.cfg, self.spec
        rng = np.random.default_rng([self.seed, 0x7E57])
        H, W = c.height, c.width
        u = (np.arange(W) + 0.5) / W * 2 - 1
        v = 1 - (np.arange(H) + 0.5) / H * 2
        uu, vv = np.meshgrid(u, v)
        d = _u(np.stack([uu * self.tanf, vv * self.tanf, -np.ones_like(uu)], -1)).reshape(-1, 3)
        P = len(d)
        o0 = np.zeros((P, 3))
        ts, si = sphere_hits(o0, d, self.cen, self.rad)
        tp = np.where(d[:, 1] < -1e-6, GROUND_Y / np.minimum(d[:, 1], -1e-6), np.inf)
        inst = np.where(ts < tp, si, np.where(np.isfinite(tp), -2, -1))
        inst = np.where(np.isfinite(np.minimum(ts, tp)), inst, -1)
        t = np.minimum(ts, tp)
        hit = np.isfinite(t)
        pt = d * np.where(hit, t, 0)[:, None]
        n = np.zeros((P, 3)); n[:, 1] = 1.0
        sph = inst >= 0
        n[sph] = _u(pt[sph] - self.cen[inst[sph]])
        skip = np.where(sph, inst, -1)
        E = np.zeros((P, 3)); vis_d = np.full(P, np.nan)
        h = np.flatnonzero(hit)
        if len(h):
            ph, nh, sk = pt[h] + n[h] * 1e-3, n[h], skip[h]
            M = len(h)
            # dominant light: sampled area/point light, shadow rays
            ld0, mult, dist0 = self.to_light(ph)
            acc = np.zeros((M, 3)); vacc = np.zeros(M)
            for k in range(c.n_light):
                off = rng.uniform(-1, 1, (M, 2)); off /= np.maximum(np.linalg.norm(off, axis=1, keepdims=True), 1) 
                a = np.where(np.abs(ld0[:, :1]) > 0.9, np.array([[0, 1.0, 0]]), np.array([[1.0, 0, 0]]))
                t1 = _u(np.cross(a, ld0)); t2 = np.cross(ld0, t1)
                if sp.dom_pos is None:
                    l = _u(ld0 + sp.soft * (t1 * off[:, :1] + t2 * off[:, 1:]))
                    dist = None
                else:
                    tgt = np.asarray(sp.dom_pos) + sp.soft * (t1 * off[:, :1] + t2 * off[:, 1:])
                    vec = tgt - ph; dist = np.linalg.norm(vec, axis=1); l = vec / dist[:, None]
                cos = np.maximum((nh * l).sum(1), 0)
                vis = ~occluded(ph, l, self.cen, self.rad, dist, sk)
                acc += (cos * vis * mult)[:, None] * np.asarray(sp.dom_rgb)
                vacc += vis
            E_dom = acc / c.n_light
            vis_d[h] = vacc / c.n_light
            # environment: cosine-weighted visibility rays
            a = np.where(np.abs(nh[:, :1]) > 0.9, np.array([[0, 1.0, 0]]), np.array([[1.0, 0, 0]]))
            t1 = _u(np.cross(a, nh)); t2 = np.cross(nh, t1)
            E_env = np.zeros((M, 3))
            for k in range(c.n_env):
                u1 = (k + rng.uniform(0, 1, M)) / c.n_env
                phi = 2 * np.pi * rng.uniform(0, 1, M)
                w = (t1 * (np.sqrt(u1) * np.cos(phi))[:, None] + t2 * (np.sqrt(u1) * np.sin(phi))[:, None]
                     + nh * np.sqrt(1 - u1)[:, None])
                free = ~occluded(ph, w, self.cen, self.rad, None, sk)
                E_env += self.env(w) * free[:, None]
            E[h] = E_dom + E_env / c.n_env
        alb = np.zeros((P, 3)); em = np.zeros((P, 3))
        alb[sph] = self.albedo[inst[sph]]; em[sph] = self.emis[inst[sph]]
        alb[inst == -2] = GROUND_ALBEDO
        rgb = (alb * E + em) * sp.exposure
        sky = inst == -1
        rgb[sky] = self.env(d[sky]) * sp.exposure
        yl = np.array([0.2126, 0.7152, 0.0722])
        bgm = (inst < 0)
        bg = float((np.clip(rgb[bgm], 0, 1) @ yl).mean()) if bgm.any() else 0.2
        surf = hit
        return Render(inst.reshape(H, W), rgb.reshape(H, W, 3), E.reshape(H, W, 3), vis_d.reshape(H, W),
                      E[surf].mean(0) if surf.any() else np.ones(3), bg)

    # ---- snapshot + truth ---------------------------------------------------------------------
    def _shadow_vis(self, i, rng):
        """Engine-side estimate: rays from the camera-facing surface point to the (soft) dominant light, other objects occlude."""
        sp = self.spec
        n = _u(-self.cen[i])
        p = self.cen[i] + n * self.rad[i] * 1.001
        K = 12
        ld, _, dist0 = self.to_light(p[None])
        ld = ld[0]
        a = np.array([0, 1.0, 0]) if abs(ld[0]) > 0.9 else np.array([1.0, 0, 0])
        t1 = _u(np.cross(a, ld)); t2 = np.cross(ld, t1)
        off = rng.uniform(-1, 1, (K, 2)); off /= np.maximum(np.linalg.norm(off, axis=1, keepdims=True), 1)
        if sp.dom_pos is None:
            l = _u(ld + sp.soft * (t1 * off[:, :1] + t2 * off[:, 1:])); tmax = None
        else:
            vec = np.asarray(sp.dom_pos) + sp.soft * (t1 * off[:, :1] + t2 * off[:, 1:]) - p
            tmax = np.linalg.norm(vec, axis=1); l = vec / tmax[:, None]
        occ = occluded(np.repeat(p[None], K, 0), l, self.cen, self.rad, tmax, np.full(K, i))
        return float(1 - occ.mean())

    def project(self, i):
        c = self.cen[i]
        dist = -c[2]
        ndc = np.array([c[0] / dist, c[1] / dist]) / self.tanf
        hw = self.rad[i] / max(dist, 1e-6) / self.tanf
        return (ndc[0] + 1) / 2, (1 - ndc[1]) / 2, hw     # viewport centre x, y (top-left origin), half-size

    def snapshot(self, rend: Render | None = None, frame=0) -> tuple[Snapshot, dict]:
        """-> (Snapshot, truth). Objects with < min_pixels visible pixels are not exported. truth: per-object rendered
        (clipped) lit colour, mean E, vis; scene white (rgb), background luminance."""
        c = self.cfg
        rend = rend or self.render()
        rng = np.random.default_rng([self.seed, 0x5AD0])
        b = SnapshotBuilder(c.observer_id, c.budget_ms, frame=frame)
        b.lighting = self.lighting()
        truth = dict(lit={}, E={}, vis={}, white_rgb=rend.white_rgb, bg_lum=rend.bg_lum, exposure=self.spec.exposure)
        cnt = np.bincount(rend.inst.ravel()[rend.inst.ravel() >= 0], minlength=len(self.objs))
        for i, o in enumerate(self.objs):
            if cnt[i] < c.min_pixels:
                continue
            m = rend.inst == i
            oid = f"o{i}"
            truth["lit"][oid] = np.clip(rend.rgb[m].mean(0), 0, 1)
            truth["E"][oid] = rend.E[m].mean(0)
            truth["vis"][oid] = float(np.nanmean(rend.vis[m]))
            cx, cy, hw = self.project(i)
            x0, y0, x1, y1 = [float(np.clip(q, 0, 1)) for q in (cx - hw, cy - hw, cx + hw, cy + hw)]
            area = (x1 - x0) * (y1 - y0)
            n = _u(-self.cen[i])
            b.add_object(oid, o["cls"], _r6(o["albedo"]), bbox=(round(x0, 6), round(y0, 6), round(x1 - x0, 6), round(y1 - y0, 6)),
                         screen_area=round(float(min(area, 1.0)), 6), depth=round(float(-self.cen[i][2]), 6), state=o["state"],
                         emissive_rgb=_r6(o["emis"]), task_tags=["hazard"] if o["cls"] in ("car", "pedestrian") else
                         (["signal"] if o["cls"] in ("light", "indicator") else []),
                         normal=tuple(_r6(n)), position=tuple(_r6(self.cen[i])), shadow_vis=round(self._shadow_vis(i, rng), 6))
        return b.build(), truth


def generate_world(cfg: WorldConfig | None, seed: int):
    """-> (Snapshot, truth, LitWorld). Deterministic in (cfg, seed)."""
    w = LitWorld(cfg or WorldConfig(), seed)
    s, t = w.snapshot()
    return s, t, w
