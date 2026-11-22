"""CVD simulation on LINEAR sRGB (last axis = channels): Machado 2009, Vienot 1999, Brettel 1997."""
from __future__ import annotations
import numpy as np
from . import color as C
from ._machado_table import MACHADO

_KIND = {"protan": "Protanomaly", "deutan": "Deuteranomaly", "tritan": "Tritanomaly"}
_AXIS = {"protan": 0, "deutan": 1, "tritan": 2}


def _kind(k):
    k = k.lower()
    if k not in _KIND:
        raise ValueError(f"kind must be one of {list(_KIND)}")
    return k


def machado_matrix(kind: str, severity: float) -> np.ndarray:
    """3x3 matrix on linear RGB; severity in [0,1], linear interpolation between published 0.1 steps."""
    if not 0.0 <= severity <= 1.0:
        raise ValueError("severity must be in [0,1]")
    tab = MACHADO[_KIND[_kind(kind)]]
    x = severity * 10
    i = min(int(np.floor(x + 1e-12)), 9)
    f = x - i
    return (1 - f) * tab[i] + f * tab[i + 1]


def machado(rgb_lin, kind: str, severity: float = 1.0, clip: bool = True):
    out = np.asarray(rgb_lin, float) @ machado_matrix(kind, severity).T
    return np.clip(out, 0, 1) if clip else out


# Vienot 1999 published LMS space and projection coefficients
VIENOT_RGB2LMS = np.array([[17.8824, 43.5161, 4.11935],
                           [3.45565, 27.1554, 3.86714],
                           [0.0299566, 0.184309, 1.46709]])
VIENOT_COEF = {  # missing-cone row of the LMS->LMS projector
    "protan": (0, np.array([0.0, 2.02344, -2.52581])),
    "deutan": (1, np.array([0.494207, 0.0, 1.24827])),
    "tritan": (2, np.array([-0.395913, 0.801109, 0.0])),
}


def vienot_matrix(kind: str) -> np.ndarray:
    """Full-dichromat 3x3 matrix on linear RGB."""
    i, row = VIENOT_COEF[_kind(kind)]
    S = np.eye(3)
    S[i] = row
    return np.linalg.inv(VIENOT_RGB2LMS) @ S @ VIENOT_RGB2LMS


def vienot(rgb_lin, kind: str, clip: bool = True):
    out = np.asarray(rgb_lin, float) @ vienot_matrix(kind).T
    return np.clip(out, 0, 1) if clip else out


_ANCHORS = {"protan": (475.0, 575.0), "deutan": (475.0, 575.0), "tritan": (485.0, 660.0)}


def _cmf_xyz(wl):
    import colour  # only for the CIE 1931 table (data); no colour maths
    return np.asarray(colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"][wl], float)


_brettel_cache: dict = {}


def _brettel_params(kind):
    if kind in _brettel_cache:
        return _brettel_cache[kind]
    i = _AXIS[kind]
    W = C.M_RGB2LMS @ np.ones(3)
    a1, a2 = (C.M_XYZ2LMS @ _cmf_xyz(w) for w in _ANCHORS[kind])
    n1, n2 = np.cross(W, a1), np.cross(W, a2)
    e = np.zeros(3)
    e[i] = 1
    nsep = np.cross(W, e)
    sign1 = np.sign(nsep @ a1)
    _brettel_cache[kind] = (i, n1, n2, nsep, sign1)
    return _brettel_cache[kind]


def brettel_matrices(kind: str):
    """(M_plane1, M_plane2, nsep, sign1): LMS->LMS projectors for the two half-planes."""
    i, n1, n2, nsep, s1 = _brettel_params(_kind(kind))
    mats = []
    for n in (n1, n2):
        P = np.eye(3)
        r = np.zeros(3)
        r[[k for k in range(3) if k != i]] = -np.delete(n, i) / n[i]
        P[i] = r
        mats.append(P)
    return mats[0], mats[1], nsep, s1


def brettel(rgb_lin, kind: str, clip: bool = True):
    """Full dichromat, two half-planes. Input linear sRGB."""
    kind = _kind(kind)
    rgb_lin = np.asarray(rgb_lin, float)
    P1, P2, nsep, s1 = brettel_matrices(kind)
    lms = C.linear_to_lms(rgb_lin)
    side = np.sign(lms @ nsep) == s1
    out = np.where(side[..., None], lms @ P1.T, lms @ P2.T)
    out = C.lms_to_linear(out)
    return np.clip(out, 0, 1) if clip else out


def simulate(rgb_lin, model: str, kind: str, severity: float = 1.0):
    """Dispatch. model in {machado, vienot, brettel}. vienot/brettel are full dichromacy;
    for severity<1 they blend linearly in linear RGB (Vienot 1999 sec.4 style)."""
    rgb_lin = np.asarray(rgb_lin, float)
    if model == "machado":
        return machado(rgb_lin, kind, severity)
    f = {"vienot": vienot, "brettel": brettel}[model]
    full = f(rgb_lin, kind)
    return np.clip((1 - severity) * rgb_lin + severity * full, 0, 1)
