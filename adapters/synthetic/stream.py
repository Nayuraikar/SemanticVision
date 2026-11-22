"""Scripted synthetic stream for the stability bench (SPEC 5.9): one critical confusable pair per event type
(collision, state_change, enter_view, lighting_change) at seeded frames, plus filler objects, light-state changes,
a grazing pair whose adjacency flaps, and sensor noise (bbox jitter, AR(1) exposure jitter). Only on-screen objects are
exported (enter_view = first appearance). Adjacency in the bench scene = bbox overlap (adj_margin 0)."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from core.perception import distance as Dst, color as C
from protocol import Snapshot, SnapshotBuilder
from .generator import LIGHTING, LIGHT_RGB

W = 0.09
EVENT_TYPES = ("collision", "state_change", "enter_view", "lighting_change")
BAND = {"collision": 0.08, "state_change": 0.30, "enter_view": 0.52, "lighting_change": 0.74}


@dataclass
class StreamConfig:
    n_frames: int = 96
    dt: float = 1 / 30
    kind: str = "deutan"            # CVD kind the critical pairs are confusable under (protan|deutan)
    first_event: int = 16
    spacing: int = 18
    jitter_bbox: float = 0.002
    jitter_exposure: float = 0.006  # AR(1) innovation std
    n_fillers: int = 3
    observer_id: str = "bench"
    budget_ms: float = 5000.0       # large: solver deadline never binds -> deterministic

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ScriptedEvent:
    type: str
    frame: int
    pair: tuple


WARM = ((0.60, 0.45, 0.30), (0.95, 0.70, 0.45), (0.3, 1.0, 0.2), 1.0)   # tinted illuminant for the lighting event
MIN_GAIN = 1.1   # per-channel irradiance*exposure floor: the solver's lit targets stay reachable with albedo <= 1


def irradiance(amb, dom, dr, ex):
    """Normal-averaged RGB gain (E * exposure) of an ambient+dominant lighting."""
    from core.lighting import LightField
    b = SnapshotBuilder("x", 1.0)
    b.ambient_dominant(_r(amb), _r(dom), _r(dr), exposure=ex)
    return np.asarray(LightField.from_lighting(b.build().lighting).irradiance(None), float) * ex


def confuser(c, kind: str, rng, n: int = 800, sev: float = 0.3, floor: float = 5.0, gain=1.0, apart_gain=None,
             apart_min: float = 8.0):
    """Albedo whose lit colour (x gain), CVD-simulated at a mild severity, is closest to c's while still different
    for normal vision (dE00 >= floor). apart_gain: a second lighting under which the pair must stay clearly
    distinguishable (dE00 >= apart_min), so a lighting change *creates* the confusion."""
    cand = rng.uniform(0.03, 0.95, size=(n, 3))
    c = np.asarray(c, float)
    kw = dict(model="machado", kind=kind, severity=sev)
    lit = lambda a, g: np.clip(a * g, 0, 1)
    p0, p1 = lit(c, gain), lit(cand, gain)
    d = Dst.distance(Dst.perceived(p0, **kw)[None], Dst.perceived(p1, **kw))
    ok = Dst.distance(p0[None], p1) >= floor
    if apart_gain is not None:
        a0, a1 = lit(c, apart_gain), lit(cand, apart_gain)
        ok &= Dst.distance(Dst.perceived(a0, **kw)[None], Dst.perceived(a1, **kw)) >= apart_min
    d = np.where(ok, d, np.inf)
    if not np.isfinite(d).any():
        d = Dst.distance(p0[None], p1)
    return cand[int(np.argmin(d))]


def _r(x):
    return [round(float(v), 6) for v in x]


def generate_stream(cfg: StreamConfig, seed: int, split: str = "heldout"):
    """-> (snapshots, scripted events). Deterministic in (cfg, seed, split)."""
    rng = np.random.default_rng([seed, 0x57AB if split == "heldout" else 0xDE5, 0x51E])
    order = list(rng.permutation(len(EVENT_TYPES)))
    when = {EVENT_TYPES[k]: cfg.first_event + cfg.spacing * i + int(rng.integers(0, 4)) for i, k in enumerate(order)}
    base = {t: rng.uniform(0.1, 0.9, 3) for t in EVENT_TYPES}
    norm = lambda L: (*L[:3], MIN_GAIN / float(irradiance(*L[:3], 1.0).min()))
    day, warm = norm(LIGHTING["day"]), norm(WARM)
    g_day, g_warm = irradiance(*day), irradiance(*warm)
    mate = {t: confuser(base[t], cfg.kind, rng, gain=g_day) for t in EVENT_TYPES if t != "lighting_change"}
    mate["lighting_change"] = confuser(base["lighting_change"], cfg.kind, rng, gain=g_warm, apart_gain=g_day)
    far = {t: rng.uniform(0.1, 0.9, 3) for t in EVENT_TYPES}
    graze = (rng.uniform(0.1, 0.9, 3),)
    graze += (confuser(graze[0], cfg.kind, rng, gain=g_day),)
    fill = [(rng.uniform(0.05, 0.9, 3), rng.uniform(0.05, 0.85), (0.20, 0.43, 0.66)[k % 3]) for k in range(cfg.n_fillers)]
    lights = [("ind0", 0.78, 0.08), ("ind1", 0.78, 0.30)]
    lstate = {n: "ok" for n, *_ in lights}
    lflip = {n: sorted(int(x) for x in rng.integers(4, max(cfg.n_frames - 4, 6), size=3)) for n, *_ in lights}
    exp_j, snaps = 0.0, []

    # geometry
    bx = 0.60
    xt = bx - W
    f_c = when["collision"]
    v_c = min(0.012, 0.9 * xt / (f_c - 0.5))
    state_x = 0.2
    f_e = when["enter_view"]

    events = [ScriptedEvent(t, when[t], p) for t, p in
              (("collision", ("col_a", "col_b")), ("state_change", ("sc_x", "sc_y")),
               ("enter_view", ("ev_e", "ev_f")), ("lighting_change", ("lt_p", "lt_q")))]
    amb_d, dom_d, dir_d, ex_d = day
    amb_n, dom_n, dir_n, ex_n = warm
    for f in range(cfg.n_frames):
        ev = []
        night = f >= when["lighting_change"]
        amb, dom, dr, ex = (amb_n, dom_n, dir_n, ex_n) if night else (amb_d, dom_d, dir_d, ex_d)
        if f == when["lighting_change"]:
            ev.append(("lighting_change", None, {"preset": "warm"}))
        exp_j = 0.9 * exp_j + rng.normal(0, cfg.jitter_exposure)
        b = SnapshotBuilder(cfg.observer_id, cfg.budget_ms, frame=f, t=round(f * cfg.dt, 9))
        b.ambient_dominant(_r(amb), _r(dom), _r(dr), exposure=round(float(ex * (1 + exp_j)), 6))
        j = lambda: rng.normal(0, cfg.jitter_bbox, 2)

        def add(id, cls, alb, x, y, state="default", emis=(0, 0, 0), tags=()):
            dx, dy = j()
            b.add_object(id, cls, _r(np.clip(alb, 0, 1)), bbox=(round(float(x + dx), 6), round(float(y + dy), 6), W, W),
                         screen_area=W * W, depth=10.0, state=state, emissive_rgb=_r(emis), task_tags=list(tags))

        # collision pair
        y = BAND["collision"]
        ax = min(xt + v_c * (f - f_c + 0.5), xt + 0.012)
        add("col_a", "car", base["collision"], ax, y, tags=("hazard",))
        add("col_b", "pedestrian", mate["collision"], bx, y, tags=("hazard",))   # distinct classes: never one palette entry
        if f == f_c:
            ev.append(("collision", "col_a", {"other": "col_b"}))
        # state change pair
        y = BAND["state_change"]
        sc_on = f >= when["state_change"]
        add("sc_x", "sign", mate["state_change"] if sc_on else far["state_change"], state_x + W - 0.012, y,
            state="alt" if sc_on else "default")
        add("sc_y", "wall", base["state_change"], state_x, y)
        if f == when["state_change"]:
            ev.append(("state_change", "sc_x", {"to": "alt"}))
        # enter view pair
        y = BAND["enter_view"]
        add("ev_f", "wall", base["enter_view"], 0.93, y)
        if f >= f_e:
            add("ev_e", "sign", mate["enter_view"], max(0.996 - 0.01 * (f - f_e), 0.94), y)
        if f == f_e:
            ev.append(("enter_view", "ev_e", {}))
        # lighting pair
        y = BAND["lighting_change"]
        add("lt_p", "wall", base["lighting_change"], 0.3, y)
        add("lt_q", "sign", mate["lighting_change"], 0.3 + W - 0.012, y)
        # grazing pair (adjacency flaps with the jitter)
        add("gz_a", "car", graze[0], 0.1, 0.90)
        add("gz_b", "wall", graze[1], 0.1 + W - 0.001, 0.90)
        for k, (alb, fx, fy) in enumerate(fill):
            add(f"fl{k}", "wall", alb, fx, fy)           # fl0 sits at y=.20: clear of every band
        # emissive lights with random state flips
        for n, lx, ly in lights:
            if f in lflip[n]:
                lstate[n] = ["ok", "warn", "fault"][(["ok", "warn", "fault"].index(lstate[n]) + 1) % 3]
                ev.append(("state_change", n, {"to": lstate[n]}))
            col = {"ok": LIGHT_RGB["green"], "warn": LIGHT_RGB["amber"], "fault": LIGHT_RGB["red"]}[lstate[n]]
            add(n, "indicator", np.array(col) * 0.3, lx, ly, state=lstate[n], emis=np.array(col) * 3.0, tags=("signal",))
        for t, oid, data in ev:
            b.add_event(t, oid, **data)
        snaps.append(b.build())
    return snaps, events
