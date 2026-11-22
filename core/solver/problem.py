"""Scene -> palette-level problem; constraint projection; batched objective (SPEC 5.5).
Palette entries = semantic groups (same class+state+original colour cluster share one entry)."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.special import logsumexp
from core.perception import color as C, distance as Dst
from core.perception.graph import build_graph, bbox_adjacency
from core.lighting import LightField, object_lit, shadow_state

MEMORY_CLASSES = ("sky", "skin", "grass", "brand")


@dataclass
class SolverConfig:
    delta: float = 15.0          # max dE00 deviation per entry
    delta_mem: float = 3.0       # tight region for memory colours
    L_range: tuple = (8.0, 95.0)  # luminance (CIE L*) bounds
    cap: float = 2.0             # separation beyond cap*tau earns nothing
    beta: float = 1.0            # soft-min sharpness
    lam: float = 0.02            # naturalness penalty weight (mean dE00 deviation)
    grid: int = 5                # coarse-grid points per Lab axis
    sweeps: int = 2
    step0: float = 0.33          # initial CD step, fraction of delta
    step_min: float = 0.02
    max_iter: int = 60
    budget_ms: float = 1500.0
    retries: int = 3
    n_cues: int = 8


def memory_mask(colors, classes=None, tags=None, brand=None, tol=3.0):
    """Protect sky/skin/grass/brand: by class/tag name, brand colour proximity, or Lab hue region."""
    c = np.asarray(colors, float)
    lab = C.linear_to_lab(c)
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    h = np.degrees(np.arctan2(b, a)) % 360
    ch = np.hypot(a, b)
    m = ((h > 220) & (h < 270) & (L > 55) & (L < 90) & (ch > 12) & (ch < 50)) \
        | ((h > 110) & (h < 140) & (L > 35) & (L < 65) & (ch > 30) & (ch < 60)) \
        | ((h > 40) & (h < 70) & (L > 40) & (L < 80) & (ch > 15) & (ch < 35))
    if classes is not None:
        m |= np.array([str(x).split("/")[0] in MEMORY_CLASSES for x in classes], bool)
    if tags is not None:
        m |= np.array([bool(set(t) & set(MEMORY_CLASSES)) for t in tags], bool)
    if brand is not None and len(brand):
        bl = C.linear_to_lab(np.asarray(brand, float))
        m |= (C.ciede2000(lab[:, None], bl[None]) < tol).any(1)
    return m


def _cluster(lab, keys, tol):
    n = len(lab)
    par = list(range(n))

    def f(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for i in range(n):
        for j in range(i + 1, n):
            if keys[i] == keys[j] and float(C.ciede2000(lab[i], lab[j])) < tol:
                par[f(i)] = f(j)
    roots = {}
    return np.array([roots.setdefault(f(i), len(roots)) for i in range(n)])


@dataclass
class Scene:
    colors: np.ndarray            # (N,3) lit linear RGB
    ids: list
    keys: list                    # class/state
    groups: np.ndarray            # (N,) palette entry index
    size_deg: np.ndarray
    area: np.ndarray
    importance: np.ndarray
    bg_lum: float
    adjacency: np.ndarray
    protected: np.ndarray         # (N,) bool memory colour
    far_weight: float = 0.0
    illum_xyz: np.ndarray | None = None   # scene white point (Y=1) -> CAT16 before CVD; None = no adaptation
    refl: np.ndarray | None = None        # (N,3) reflected part of the lit colour (HDR)
    emis: np.ndarray | None = None        # (N,3) emitted part (HDR), added separately
    gain: np.ndarray | None = None        # (N,3) irradiance * exposure
    shadow: list | None = None            # (N,) lit|penumbra|shadow
    exposure: float = 1.0

    @property
    def n_groups(self):
        return int(self.groups.max()) + 1

    @classmethod
    def build(cls, colors, keys, size_deg, area, adjacency, importance=None, bg_lum=0.2, ids=None,
              protected=None, far_weight=0.0, cluster_tol=8.0, brand=None, tags=None):
        c = np.asarray(colors, float)
        n = len(c)
        lab = C.linear_to_lab(c)
        groups = _cluster(lab, list(keys), cluster_tol)
        prot = memory_mask(c, keys, tags, brand) if protected is None else np.asarray(protected, bool)
        return cls(c, list(range(n)) if ids is None else list(ids), list(keys), groups, np.asarray(size_deg, float),
                   np.asarray(area, float), np.ones(n) if importance is None else np.asarray(importance, float),
                   float(bg_lum), np.asarray(adjacency, bool), prot, far_weight)

    @classmethod
    def from_snapshot(cls, snap, fov=60.0, adj_margin=0.3, far_weight=0.0, bg_lum=0.2, brand=None,
                      lighting="lit", adapt=True, **kw):
        """lighting: 'lit' (SH-L2 + shadow + emissive, canonical) | 'albedo' (legacy: albedo as colour).
        adapt: CAT16 to the scene white point before CVD."""
        o = snap.objects
        fld = LightField.from_lighting(snap.lighting)
        parts = [object_lit(fld, x) for x in o]
        if lighting == "albedo":
            col = np.array([x.albedo_rgb for x in o], float)
        else:
            col = np.array([p[0] for p in parts], float)
        bb = np.array([(x.bbox.x, x.bbox.y, x.bbox.w, x.bbox.h) for x in o], float)              # x,y,w,h
        xyxy = np.c_[bb[:, 0], bb[:, 1], bb[:, 0] + bb[:, 2], bb[:, 1] + bb[:, 3]]
        area = np.maximum(bb[:, 2] * bb[:, 3], 1e-6)
        imp = np.array([2.0 if "hazard" in x.task_tags else 1.0 for x in o])
        sc = cls.build(col, [f"{x.cls}/{x.state}" for x in o], np.maximum(np.sqrt(area) * fov, 0.1), area,
                       bbox_adjacency(xyxy, adj_margin), imp, bg_lum, [x.id for x in o],
                       far_weight=far_weight, brand=brand, tags=[x.task_tags for x in o], **kw)
        if lighting != "albedo":
            sc.refl, sc.emis = np.array([p[1] for p in parts]), np.array([p[2] for p in parts])
            sc.gain = np.array([p[3] * fld.exposure for p in parts])
            sc.shadow = [shadow_state(x.shadow_vis) for x in o]
            sc.exposure = fld.exposure
            if adapt:
                sc.illum_xyz = fld.white_xyz()
        return sc

    def perceive_kw(self, obs):
        """Perception kwargs for an observer: CVD model + CAT16 to the scene white point."""
        kw = dict(obs.sim_kwargs())
        if self.illum_xyz is not None:
            kw["illum_xyz"] = self.illum_xyz
        return kw

    def to_overrides(self, colors):
        """Corrected display colours (N,3) -> (albedo', emissive') per object, inverting the lit model:
        emissive-dominated objects change their emission (never scaled by irradiance) and keep their albedo;
        others change albedo and keep their emission. Without lighting info: (colors, None)."""
        c = np.clip(np.asarray(colors, float), 0, 1)
        if self.refl is None:
            return c, None
        yl = np.array([0.2126, 0.7152, 0.0722])
        g = np.maximum(self.gain, 1e-9)
        use_e = ((self.emis @ yl) > (self.refl @ yl))[:, None]
        amp = np.maximum((self.emis @ yl) / np.maximum(np.minimum(self.emis, 1) @ yl, 1e-9), 1.0)[:, None]
        alb = np.where(use_e, np.clip(self.refl / g, 0, 1), np.clip((c - self.emis) / g, 0, 1))
        em = np.where(use_e, np.maximum(c - self.refl, 0) * amp / self.exposure, self.emis / self.exposure)
        return alb, em

    def graph(self, obs, colors=None):
        return build_graph(self.colors if colors is None else colors, self.ids, self.size_deg, self.area,
                           self.importance, self.bg_lum, adjacency=self.adjacency, far_weight=self.far_weight,
                           s_L=obs.s_L, s_C=obs.s_C, k=obs.k, **self.perceive_kw(obs))

    def palette(self):
        """Group-level arrays: colour (area-weighted mean), min size, summed area, max importance, adjacency."""
        E = self.n_groups
        col = np.zeros((E, 3)); sz = np.zeros(E); ar = np.zeros(E); im = np.zeros(E); prot = np.zeros(E, bool)
        for e in range(E):
            m = self.groups == e
            w = self.area[m] / self.area[m].sum()
            col[e] = (w[:, None] * self.colors[m]).sum(0)
            sz[e], ar[e], im[e], prot[e] = self.size_deg[m].min(), self.area[m].sum(), self.importance[m].max(), self.protected[m].any()
        G = np.eye(E, dtype=bool)[self.groups]            # (N,E)
        adj = (G.T.astype(int) @ self.adjacency.astype(int) @ G.astype(int)) > 0
        np.fill_diagonal(adj, False)
        return col, sz, ar, im, adj, prot


class Problem:
    """Palette-level objective under a set of (training) observers with per-observer thresholds."""

    def __init__(self, scene: Scene, observers, cfg: SolverConfig = SolverConfig()):
        self.scene, self.obs, self.cfg = scene, list(observers), cfg
        col, sz, ar, im, adj, prot = scene.palette()
        self.col0, self.prot = col, prot
        self.lab0 = C.linear_to_lab(col)
        self.E = len(col)
        self.delta = np.where(prot, cfg.delta_mem, cfg.delta)
        taus, W = [], None
        for o in self.obs:
            g = build_graph(col, None, sz, ar, im, scene.bg_lum, adjacency=adj, far_weight=scene.far_weight,
                            s_L=o.s_L, s_C=o.s_C, k=o.k, **scene.perceive_kw(o))
            taus.append(g.tau); W = g.weight
        iu = np.triu(W > 1e-9, 1)
        self.pairs = np.argwhere(iu)
        self.ia, self.ib = self.pairs[:, 0], self.pairs[:, 1]
        self.tau = np.array([t[iu] for t in taus]) if len(self.pairs) else np.zeros((len(self.obs), 0))
        w = W[iu] / max(W[iu].max(), 1e-12) if len(self.pairs) else np.zeros(0)
        self.w = np.maximum(w, 0.2)
        self.boost = np.ones(len(self.pairs))
        self.active = np.zeros(self.E, bool)
        self.active[self.ia] = True; self.active[self.ib] = True
        self.beta = cfg.beta

    # --- constraints -------------------------------------------------------------------------
    def _rgb(self, lab):
        return C.xyz_to_linear(C.lab_to_xyz(lab))

    def _dev(self, X):
        d = C.ciede2000(X, self.lab0)
        lo, hi = np.zeros(d.shape), np.ones(d.shape)
        bad = d > self.delta
        if not bad.any():
            return X
        for _ in range(14):
            mid = 0.5 * (lo + hi)
            ok = C.ciede2000(self.lab0 + mid[..., None] * (X - self.lab0), self.lab0) <= self.delta
            lo, hi = np.where(ok, mid, lo), np.where(ok, hi, mid)
        return np.where(bad[..., None], self.lab0 + lo[..., None] * (X - self.lab0), X)

    def _gamut(self, X):
        def inside(L):
            r = self._rgb(L)
            return ((r >= -1e-4) & (r <= 1 + 1e-4)).all(-1)
        bad = ~inside(X)
        if not bad.any():
            return X
        lo, hi = np.zeros(bad.shape), np.ones(bad.shape)
        for _ in range(14):
            mid = 0.5 * (lo + hi)
            Y = np.stack([X[..., 0], X[..., 1] * mid, X[..., 2] * mid], -1)
            ok = inside(Y)
            lo, hi = np.where(ok, mid, lo), np.where(ok, hi, mid)
        Y = np.stack([X[..., 0], X[..., 1] * lo, X[..., 2] * lo], -1)
        return np.where(bad[..., None], Y, X)

    def project(self, P):
        """(…,E,3) Lab -> feasible Lab: L bounds, deviation ball, gamut (alternating, deviation last)."""
        X = np.array(P, float)
        X[..., 0] = np.clip(X[..., 0], *self.cfg.L_range)
        X = self._dev(self._gamut(self._dev(X)))
        return X

    def to_rgb(self, lab):
        return np.clip(self._rgb(lab), 0, 1)

    # --- objective ---------------------------------------------------------------------------
    def separation_terms(self, X):
        """X (B,E,3) projected Lab -> capped distances (B, M*P) under each observer's CVD + threshold."""
        B = X.shape[0]
        flat = self.to_rgb(X).reshape(-1, 3)
        out = []
        for m, o in enumerate(self.obs):
            lab = C.linear_to_lab(Dst.perceived(flat, **self.scene.perceive_kw(o))).reshape(B, self.E, 3)
            D = C.ciede2000(lab[:, self.ia], lab[:, self.ib])
            out.append(np.minimum(D, self.cfg.cap * self.tau[m]))
        return np.concatenate(out, 1)

    def softmin(self, V):
        M = len(self.obs)
        lw = np.log(np.tile(self.w * self.boost, M))
        return -(logsumexp(-self.beta * V + lw, axis=1) - logsumexp(lw)) / self.beta

    def deviation(self, X):
        return C.ciede2000(X, self.lab0)

    def objective(self, P):
        """Maximise. P (B,E,3) Lab (unprojected ok) -> (B,)."""
        X = self.project(P)
        if not len(self.pairs):
            return -self.cfg.lam * self.deviation(X).mean(-1)
        return self.softmin(self.separation_terms(X)) - self.cfg.lam * self.deviation(X).mean(-1)
