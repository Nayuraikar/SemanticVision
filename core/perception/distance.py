"""Perceived colour and pair distance D(i,j) = dE(CVD(Ci), CVD(Cj))."""
from __future__ import annotations
import numpy as np
from . import color as C
from . import cvd as V


def perceived(rgb_lin, model: str | None = None, kind: str = "deutan", severity: float = 1.0,
              illum_xyz=None, L_A: float = 64 / np.pi * 0.2, F: float = 1.0):
    """Lit linear-sRGB -> CAT16 (illuminant white -> D65) -> CVD -> linear RGB (adapted, simulated)."""
    rgb = np.asarray(rgb_lin, float)
    if illum_xyz is not None:
        M = C.cat16_matrix(illum_xyz, C.D65_XYZ, float(np.clip(C.cat16_degree(L_A, F), 0, 1)))
        rgb = C.xyz_to_linear(C.linear_to_xyz(rgb) @ M.T)
    if model is not None:
        rgb = V.simulate(np.clip(rgb, 0, 1), model, kind, severity)
    return rgb


def to_space(rgb_lin, space: str = "de2000"):
    if space == "de2000":
        return C.linear_to_lab(rgb_lin)
    if space == "cam16":
        return C.xyz_to_cam16ucs(C.linear_to_xyz(rgb_lin))
    raise ValueError(space)


def distance(a, b, space="de2000"):
    """Distance between two sets of already-perceived linear RGB (broadcasting)."""
    if space == "de2000":
        return C.ciede2000(to_space(a, space), to_space(b, space))
    return C.cam16ucs_distance(to_space(a, space), to_space(b, space))


def pair_distance(ci, cj, space="de2000", **kw):
    return distance(perceived(ci, **kw), perceived(cj, **kw), space)


def pair_matrix(colors, space="de2000", **kw):
    """(N,3) lit linear RGB -> (N,N) symmetric D matrix."""
    p = perceived(colors, **kw)
    z = to_space(p, space)
    if space == "de2000":
        return C.ciede2000(z[:, None, :], z[None, :, :])
    return C.cam16ucs_distance(z[:, None, :], z[None, :, :])
