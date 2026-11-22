"""Calibration stimulus pool: colour pairs along cone-confusion lines (protan=L, deutan=M, tritan=S) and a
lightness axis, across lightness levels and base chromas; each pair is shown at a patch size and background.
All colours are LINEAR sRGB. stimulus index = pair * N_COMBO + combo (combo = size_idx * len(BGS) + bg_idx)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from core.perception import color as C

AXES = ("protan", "deutan", "tritan", "lightness")
KINDS = AXES[:3]
LEVELS_Y = (0.08, 0.2, 0.4, 0.7)                 # base relative luminance
TINTS = {"gray": (1, 1, 1), "red": (1, .35, .35), "green": (.4, 1, .4), "blue": (.4, .4, 1)}
DELTAS = (0.04, 0.07, 0.12, 0.2, 0.33, 0.55)     # relative modulation (cone contrast / luminance ratio - 1)
SIZES = (0.5, 2.0, 6.0)                          # patch size, degrees
BGS = (0.02, 0.2, 0.6)                           # background relative luminance
N_COMBO = len(SIZES) * len(BGS)
REF_COMBO = 1 * len(BGS) + 1                     # 2 deg on 0.2 background


def base_colour(y: float, tint) -> np.ndarray:
    t = np.asarray(tint, float)
    return t * (y / float(C.linear_to_xyz(t)[1]))


def make_pair(axis: str, base, delta):
    """base (...,3), delta (...) -> (c1, c2) clipped to [0,1]. Confusion-line axes scale one cone (HPE LMS)."""
    base = np.asarray(base, float)
    delta = np.asarray(delta, float)
    if axis == "lightness":
        c2 = base * (1 + delta[..., None])
    else:
        lms = C.linear_to_lms(base)
        lms2 = lms.copy()
        lms2[..., AXES.index(axis)] = lms[..., AXES.index(axis)] * (1 + delta)
        c2 = C.lms_to_linear(lms2)
    return np.clip(base, 0, 1), np.clip(c2, 0, 1)


@dataclass(frozen=True)
class Pool:
    c1: np.ndarray          # (P,3)
    c2: np.ndarray          # (P,3)
    axis: np.ndarray        # (P,) index into AXES
    level: np.ndarray       # (P,) index into LEVELS_Y
    tint: np.ndarray        # (P,) index into TINTS
    delta: np.ndarray       # (P,)

    @property
    def n_pairs(self) -> int:
        return len(self.delta)

    @property
    def n_stim(self) -> int:
        return self.n_pairs * N_COMBO


def build_pool() -> Pool:
    c1, c2, ax, lv, tn, dl = [], [], [], [], [], []
    for a, axis in enumerate(AXES):
        for l, y in enumerate(LEVELS_Y):
            for t, tint in enumerate(TINTS.values()):
                b = base_colour(y, tint)
                if b.max() > 1:                       # base out of gamut at this luminance
                    continue
                for d in DELTAS:
                    p1, p2 = make_pair(axis, b, np.float64(d))
                    lms_ok = axis == "lightness" or np.all(np.asarray(C.linear_to_lms(b)) > 0)
                    raw_ok = axis == "lightness" and (b * (1 + d)).max() <= 1.0 or \
                        axis != "lightness" and np.all((C.lms_to_linear(_scaled(b, axis, d)) >= -1e-9) &
                                                       (C.lms_to_linear(_scaled(b, axis, d)) <= 1 + 1e-9))
                    if not (lms_ok and raw_ok):       # keep only unclipped (exact-contrast) pairs
                        continue
                    c1.append(p1); c2.append(p2); ax.append(a); lv.append(l); tn.append(t); dl.append(d)
    return Pool(np.array(c1), np.array(c2), np.array(ax), np.array(lv), np.array(tn), np.array(dl))


def _scaled(b, axis, d):
    lms = C.linear_to_lms(b)
    lms[AXES.index(axis)] *= 1 + d
    return lms


_POOL: Pool | None = None


def pool() -> Pool:
    global _POOL
    if _POOL is None:
        _POOL = build_pool()
    return _POOL


def decode(stim: int) -> tuple[int, float, float]:
    """stimulus index -> (pair index, size_deg, bg_lum)."""
    p, c = divmod(int(stim), N_COMBO)
    return p, SIZES[c // len(BGS)], BGS[c % len(BGS)]
