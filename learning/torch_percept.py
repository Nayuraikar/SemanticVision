"""Differentiable perception in PyTorch, mirroring core/perception (tested against it): Lab<->linear sRGB, CIEDE2000,
CAT16 (as a precomputed linear-RGB matrix) and CVD simulation (Machado/Vienot matrices, Brettel half-planes).
Only ONNX-exportable ops (no torch.atan2: own atan2 from atan + where)."""
from __future__ import annotations
import math
import numpy as np
import torch
from core.perception import color as C, cvd as V

_T = lambda a: torch.tensor(np.asarray(a, np.float64), dtype=torch.float32)
M_RGB2XYZ, M_XYZ2RGB = _T(C.M_RGB2XYZ), _T(C.M_XYZ2RGB)
D65 = _T(C.D65_XYZ)
EPS = 1e-12
_D = 6 / 29


def atan2(y, x):
    """Exportable atan2 (radians, (-pi, pi])."""
    small = x.abs() < 1e-12
    safe = torch.where(small, torch.ones_like(x), x)
    a = torch.atan(y / safe)
    a = torch.where(x < 0, torch.where(y >= 0, a + math.pi, a - math.pi), a)
    return torch.where(small, torch.where(y >= 0, torch.full_like(y, math.pi / 2), torch.full_like(y, -math.pi / 2)), a)


def xyz_to_lab(xyz):
    t = xyz / D65.to(xyz)
    f = torch.where(t > _D ** 3, torch.clamp(t, min=_D ** 3) ** (1 / 3), t / (3 * _D * _D) + 4 / 29)
    return torch.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab_to_xyz(lab):
    fy = (lab[..., 0] + 16) / 116
    f = torch.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    t = torch.where(f > _D, f ** 3, 3 * _D * _D * (f - 4 / 29))
    return t * D65.to(lab)


def linear_to_lab(rgb):
    return xyz_to_lab(rgb @ M_RGB2XYZ.to(rgb).T)


def lab_to_linear(lab):
    return lab_to_xyz(lab) @ M_XYZ2RGB.to(lab).T


def ciede2000(lab1, lab2):
    l1, a1, b1 = lab1.unbind(-1)
    l2, a2, b2 = lab2.unbind(-1)
    c1 = torch.sqrt(a1 * a1 + b1 * b1 + EPS)
    c2 = torch.sqrt(a2 * a2 + b2 * b2 + EPS)
    cb7 = ((c1 + c2) / 2) ** 7
    g = 0.5 * (1 - torch.sqrt(cb7 / (cb7 + 25.0 ** 7) + EPS))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p = torch.sqrt(a1p * a1p + b1 * b1 + EPS)
    c2p = torch.sqrt(a2p * a2p + b2 * b2 + EPS)
    h1p = torch.remainder(atan2(b1, a1p) * (180 / math.pi), 360.0)
    h2p = torch.remainder(atan2(b2, a2p) * (180 / math.pi), 360.0)
    zero = (c1p * c2p) < 1e-9
    dl, dc = l2 - l1, c2p - c1p
    dh = h2p - h1p
    dh = torch.where(dh > 180, dh - 360, torch.where(dh < -180, dh + 360, dh))
    dh = torch.where(zero, torch.zeros_like(dh), dh)
    dH = 2 * torch.sqrt(c1p * c2p) * torch.sin(dh * (math.pi / 360))
    lb, cbp, hs = (l1 + l2) / 2, (c1p + c2p) / 2, h1p + h2p
    hb = torch.where((h1p - h2p).abs() <= 180, hs / 2, torch.where(hs < 360, (hs + 360) / 2, (hs - 360) / 2))
    hb = torch.where(zero, hs, hb)
    r = math.pi / 180
    T = (1 - 0.17 * torch.cos((hb - 30) * r) + 0.24 * torch.cos(2 * hb * r) + 0.32 * torch.cos((3 * hb + 6) * r)
         - 0.20 * torch.cos((4 * hb - 63) * r))
    dth = 30 * torch.exp(-(((hb - 275) / 25) ** 2))
    cbp7 = cbp ** 7
    rc = 2 * torch.sqrt(cbp7 / (cbp7 + 25.0 ** 7) + EPS)
    sl = 1 + 0.015 * (lb - 50) ** 2 / torch.sqrt(20 + (lb - 50) ** 2)
    sc, sh = 1 + 0.045 * cbp, 1 + 0.015 * cbp * T
    rt = -torch.sin(2 * dth * r) * rc
    x, y, zz = dl / sl, dc / sc, dH / sh
    return torch.sqrt(torch.clamp(x * x + y * y + zz * zz + rt * y * zz, min=0) + EPS)


# ---- observer -------------------------------------------------------------------------------------------------------
def perceive_matrix(rgb, M_cat, M_cvd):
    """core.perception.distance.perceived for matrix models: CAT (no clip) -> clip -> CVD -> clip. rgb (...,N,3),
    M_* (...,3,3) broadcast over N."""
    x = torch.clamp(rgb @ M_cat.transpose(-1, -2), 0, 1)
    return torch.clamp(x @ M_cvd.transpose(-1, -2), 0, 1)


def brettel(rgb, kind):
    P1, P2, nsep, s1 = V.brettel_matrices(kind)
    lms = rgb @ _T(C.M_RGB2LMS).to(rgb).T
    side = torch.sign(lms @ _T(nsep).to(rgb)) == float(s1)
    out = torch.where(side[..., None], lms @ _T(P1).to(rgb).T, lms @ _T(P2).to(rgb).T)
    return torch.clamp(out @ _T(C.M_LMS2RGB).to(rgb).T, 0, 1)


def simulate(rgb, model, kind, severity=1.0):
    """Mirror of core.perception.cvd.simulate on linear sRGB."""
    rgb = torch.clamp(rgb, 0, 1)
    if model == "machado":
        return torch.clamp(rgb @ _T(V.machado_matrix(kind, severity)).to(rgb).T, 0, 1)
    full = torch.clamp(rgb @ _T(V.vienot_matrix(kind)).to(rgb).T, 0, 1) if model == "vienot" else brettel(rgb, kind)
    return torch.clamp((1 - severity) * rgb + severity * full, 0, 1)


def observer_lab(lab, M_cat, M_cvd):
    """Displayed Lab (...,N,3) -> Lab as perceived by a matrix-model observer (RGB->[CAT]->CVD->Lab)."""
    rgb = torch.clamp(lab_to_linear(lab), 0, 1)
    return linear_to_lab(perceive_matrix(rgb, M_cat, M_cvd))
