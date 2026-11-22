"""Real spherical harmonics up to L2 (9 coeffs). Radiance coefficients c_lm; normalised diffuse irradiance
E(n) = sum_lm A_l c_lm Y_lm(n), A = (1, 2/3, 1/4) (= Ramamoorthi-Hanrahan A_l / pi), so lit = albedo * E."""
from __future__ import annotations
import numpy as np

BAND = np.array([0, 1, 1, 1, 2, 2, 2, 2, 2])
A_L = np.array([1.0, 2 / 3, 1 / 4])[BAND]
Y00 = 0.282094791773878


def basis(n):
    """(...,3) unit vectors -> (...,9)."""
    n = np.asarray(n, float)
    x, y, z = n[..., 0], n[..., 1], n[..., 2]
    return np.stack([np.full_like(x, Y00), 0.488602511902920 * y, 0.488602511902920 * z, 0.488602511902920 * x,
                     1.092548430592079 * x * y, 1.092548430592079 * y * z, 0.315391565252520 * (3 * z * z - 1),
                     1.092548430592079 * x * z, 0.546274215296040 * (x * x - y * y)], -1)


def unit(v):
    v = np.asarray(v, float)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def irradiance(coeffs, n=None):
    """coeffs (9,3); n (...,3) or None (normal-averaged: L0 term only) -> (...,3) normalised irradiance, >= 0."""
    c = np.asarray(coeffs, float)
    if n is None:
        return np.maximum(Y00 * c[0], 0.0)
    return np.maximum((basis(unit(n)) * A_L) @ c, 0.0)


def delta_coeffs(direction, rgb):
    """SH of a directional light whose normalised irradiance on a facing surface is rgb:
    c_lm = pi * rgb * Y_lm(dir) (so E(dir) ~ rgb, the L2 truncation of max(0, cos))."""
    return np.pi * basis(unit(direction))[:, None] * np.asarray(rgb, float)[None, :]


def uniform_coeffs(rgb):
    """Uniform ambient of normalised irradiance rgb."""
    c = np.zeros((9, 3))
    c[0] = np.asarray(rgb, float) / Y00
    return c


def fibonacci_sphere(n=2048):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.cos(phi), np.sin(th) * np.sin(phi)], -1)


def project(radiance_fn, n=2048):
    """Project a radiance function L(dirs (M,3)) -> (M,3) RGB onto SH-L2 coefficients (9,3)."""
    d = fibonacci_sphere(n)
    return 4 * np.pi / n * (basis(d).T @ np.asarray(radiance_fn(d), float))
