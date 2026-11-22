"""Colour-space primitives. Vectorised; last axis = channels. XYZ scale: Y=1 for white unless noted."""
from __future__ import annotations
import numpy as np

M_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                      [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]])
M_XYZ2RGB = np.linalg.inv(M_RGB2XYZ)
D65_XYZ = M_RGB2XYZ @ np.ones(3)  # Y=1
# Hunt-Pointer-Estevez (D65-normalised) cone space
M_XYZ2LMS = np.array([[0.38971, 0.68898, -0.07868],
                      [-0.22981, 1.18340, 0.04641],
                      [0.0, 0.0, 1.0]])
M_LMS2XYZ = np.linalg.inv(M_XYZ2LMS)
M_RGB2LMS = M_XYZ2LMS @ M_RGB2XYZ
M_LMS2RGB = np.linalg.inv(M_RGB2LMS)
M16 = np.array([[0.401288, 0.650173, -0.051461],
                [-0.250268, 1.204414, 0.045854],
                [-0.002079, 0.048952, 0.953127]])
M16_INV = np.linalg.inv(M16)


def _a(x):
    return np.asarray(x, dtype=float)


def srgb_to_linear(c):
    c = _a(c)
    return np.where(c <= 0.04045, c / 12.92, ((np.abs(c) + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = _a(c)
    a = np.abs(c)
    return np.sign(c) * np.where(a <= 0.0031308, a * 12.92, 1.055 * a ** (1 / 2.4) - 0.055)


def linear_to_xyz(rgb):
    return _a(rgb) @ M_RGB2XYZ.T


def xyz_to_linear(xyz):
    return _a(xyz) @ M_XYZ2RGB.T


def linear_to_lms(rgb):
    return _a(rgb) @ M_RGB2LMS.T


def lms_to_linear(lms):
    return _a(lms) @ M_LMS2RGB.T


def srgb_to_lms(c):
    return linear_to_lms(srgb_to_linear(c))


def lms_to_srgb(lms):
    return linear_to_srgb(lms_to_linear(lms))


def cat16_matrix(xyz_src, xyz_dst, D=1.0):
    """3x3 XYZ->XYZ CAT16 adaptation (any consistent XYZ scale), degree of adaptation D (scalar)."""
    ws = M16 @ _a(xyz_src)
    wd = M16 @ _a(xyz_dst)
    gain = D * wd / ws + (1.0 - D)
    return M16_INV @ np.diag(gain) @ M16


def cat16_degree(L_A, F=1.0):
    """CIECAM16 degree of adaptation."""
    return F * (1.0 - (1.0 / 3.6) * np.exp((-_a(L_A) - 42.0) / 92.0))


def cat16_adapt(xyz, xyz_src, xyz_dst=D65_XYZ, D=1.0):
    return _a(xyz) @ cat16_matrix(xyz_src, xyz_dst, D).T


def xyz_to_lab(xyz, white=D65_XYZ):
    t = _a(xyz) / _a(white)
    d = 6 / 29
    f = np.where(t > d ** 3, np.cbrt(t), t / (3 * d * d) + 4 / 29)
    L = 116 * f[..., 1] - 16
    return np.stack([L, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab_to_xyz(lab, white=D65_XYZ):
    lab = _a(lab)
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    f = np.stack([fx, fy, fz], -1)
    d = 6 / 29
    t = np.where(f > d, f ** 3, 3 * d * d * (f - 4 / 29))
    return t * _a(white)


def linear_to_lab(rgb, white=D65_XYZ):
    return xyz_to_lab(linear_to_xyz(rgb), white)


def ciede2000(lab1, lab2, kL=1.0, kC=1.0, kH=1.0):
    """CIEDE2000 (Sharma 2005), broadcasting over leading axes."""
    l1, a1, b1 = np.moveaxis(_a(lab1), -1, 0)
    l2, a2, b2 = np.moveaxis(_a(lab2), -1, 0)
    c1 = np.hypot(a1, b1)
    c2 = np.hypot(a2, b2)
    cb = (c1 + c2) / 2
    g = 0.5 * (1 - np.sqrt(cb ** 7 / (cb ** 7 + 25.0 ** 7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dl = l2 - l1
    dc = c2p - c1p
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(c1p * c2p == 0, 0.0, dh)
    dH = 2 * np.sqrt(c1p * c2p) * np.sin(np.radians(dh) / 2)
    lb = (l1 + l2) / 2
    cbp = (c1p + c2p) / 2
    hs = h1p + h2p
    hb = np.where(np.abs(h1p - h2p) <= 180, hs / 2, np.where(hs < 360, (hs + 360) / 2, (hs - 360) / 2))
    hb = np.where(c1p * c2p == 0, hs, hb)
    T = (1 - 0.17 * np.cos(np.radians(hb - 30)) + 0.24 * np.cos(np.radians(2 * hb))
         + 0.32 * np.cos(np.radians(3 * hb + 6)) - 0.20 * np.cos(np.radians(4 * hb - 63)))
    dth = 30 * np.exp(-(((hb - 275) / 25) ** 2))
    rc = 2 * np.sqrt(cbp ** 7 / (cbp ** 7 + 25.0 ** 7))
    sl = 1 + 0.015 * (lb - 50) ** 2 / np.sqrt(20 + (lb - 50) ** 2)
    sc = 1 + 0.045 * cbp
    sh = 1 + 0.015 * cbp * T
    rt = -np.sin(np.radians(2 * dth)) * rc
    x, y, z = dl / (kL * sl), dc / (kC * sc), dH / (kH * sh)
    return np.sqrt(x * x + y * y + z * z + rt * y * z)


# ---- CAM16 / CAM16-UCS (forward model) ----
SURROUND = {"average": (1.0, 0.69, 1.0), "dim": (0.9, 0.59, 0.9), "dark": (0.8, 0.525, 0.8)}


def xyz_to_cam16ucs(xyz, xyz_w=D65_XYZ, L_A=64 / np.pi * 0.2, Y_b=0.2, surround="average"):
    """XYZ (Y=1 white-scale) -> CAM16-UCS (J', a', b'). xyz_w is the adopted white (Y=1)."""
    F, c, Nc = SURROUND[surround]
    xyz = _a(xyz) * 100.0
    w = _a(xyz_w) / _a(xyz_w)[1] * 100.0
    D = np.clip(cat16_degree(L_A, F), 0, 1)
    rgb_w = M16 @ w
    Drgb = D * w[1] / rgb_w + 1 - D
    rgb_c = (xyz @ M16.T) * Drgb
    rgb_wc = rgb_w * Drgb
    k = 1 / (5 * L_A + 1)
    FL = 0.2 * k ** 4 * (5 * L_A) + 0.1 * (1 - k ** 4) ** 2 * (5 * L_A) ** (1 / 3)
    n = Y_b / w[1]
    z = 1.48 + np.sqrt(n)
    Nbb = 0.725 * n ** -0.2

    def nl(v):
        p = (FL * np.abs(v) / 100) ** 0.42
        return np.sign(v) * 400 * p / (27.13 + p) + 0.1

    ra = nl(rgb_c)
    raw = nl(rgb_wc)
    a = ra[..., 0] - 12 * ra[..., 1] / 11 + ra[..., 2] / 11
    b = (ra[..., 0] + ra[..., 1] - 2 * ra[..., 2]) / 9
    h = np.arctan2(b, a) % (2 * np.pi)
    et = 0.25 * (np.cos(h + 2) + 3.8)
    A = (2 * ra[..., 0] + ra[..., 1] + 0.05 * ra[..., 2] - 0.305) * Nbb
    Aw = (2 * raw[0] + raw[1] + 0.05 * raw[2] - 0.305) * Nbb
    J = 100 * np.sign(A / Aw) * np.abs(A / Aw) ** (c * z)
    t = (50000 / 13 * Nc * Nbb * et * np.hypot(a, b)) / (ra[..., 0] + ra[..., 1] + 21 / 20 * ra[..., 2])
    C = np.abs(t) ** 0.9 * np.sqrt(np.maximum(J, 0) / 100) * (1.64 - 0.29 ** n) ** 0.73
    M = C * FL ** 0.25
    Jp = 1.7 * J / (1 + 0.007 * J)
    Mp = np.log1p(0.0228 * M) / 0.0228
    return np.stack([Jp, Mp * np.cos(h), Mp * np.sin(h)], -1)


def cam16ucs_distance(u1, u2):
    return np.linalg.norm(_a(u1) - _a(u2), axis=-1)
