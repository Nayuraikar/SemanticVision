"""LightField: protocol lighting -> irradiance by normal (+ position for a point dominant light, + shadow visibility)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from core.perception import color as C
from . import sh as S


@dataclass(frozen=True)
class Dominant:
    rgb: np.ndarray            # normalised irradiance on a facing surface (at ref_dist for a point light)
    direction: np.ndarray      # unit, towards the light (nominal)
    position: np.ndarray | None = None
    ref_dist: float = 1.0

    def at(self, p=None):
        """(direction, rgb) seen from world position p (point lights: direction to the light, 1/d^2 falloff)."""
        if self.position is None or p is None:
            return self.direction, self.rgb
        v = self.position - np.asarray(p, float)
        d = max(float(np.linalg.norm(v)), 1e-6)
        return v / d, self.rgb * (self.ref_dist / d) ** 2


@dataclass(frozen=True)
class LightField:
    sh: np.ndarray                      # (9,3) total radiance SH (includes the dominant term)
    dom: Dominant | None = None
    exposure: float = 1.0
    white_point: tuple = tuple(C.D65_XYZ)

    @classmethod
    def from_lighting(cls, L):
        if L.kind == "sh2":
            d = L.dominant
            dom = None if d is None else Dominant(np.array(d.rgb, float), S.unit(d.direction),
                                                  None if d.position is None else np.array(d.position, float), d.ref_dist)
            return cls(np.array(L.coeffs, float), dom, float(L.exposure), tuple(L.white_point))
        dr = S.unit(L.dominant_dir)
        rgb = np.array(L.dominant_rgb, float)
        pos = None if L.dominant_pos is None else np.array(L.dominant_pos, float)
        sh = S.uniform_coeffs(L.ambient_rgb) + S.delta_coeffs(dr, rgb)
        return cls(sh, Dominant(rgb, dr, pos), float(L.exposure), tuple(L.white_point))

    def dominant_sh(self, p=None):
        d, rgb = self.dom.at(p)
        return S.delta_coeffs(d, rgb)

    def irradiance(self, n=None, p=None, vis=None):
        """Normalised irradiance (...,3). n None = normal-averaged. vis/p None = plain SH-L2 (no shadow/position);
        with a dominant light: E = SH(n) - dom_nominal(n) + vis * dom(n, p)  (vis defaults to 1 when only p is given)."""
        E = S.irradiance(self.sh, n) if (vis is None and p is None) or self.dom is None else None
        if E is not None:
            return E
        v = 1.0 if vis is None else np.asarray(vis, float)[..., None] if np.ndim(vis) else float(vis)
        if n is None:
            base = S.Y00 * self.sh[0]
            dom_nom = S.Y00 * S.delta_coeffs(self.dom.direction, self.dom.rgb)[0]
            dom_p = S.Y00 * self.dominant_sh(p)[0]
        else:
            basis = S.basis(S.unit(n)) * S.A_L
            base = basis @ self.sh
            dom_nom = basis @ S.delta_coeffs(self.dom.direction, self.dom.rgb)
            dom_p = basis @ self.dominant_sh(p)
        return np.maximum(base - dom_nom + v * dom_p, 0.0)

    def white_rgb(self):
        """Scene illuminant colour: normal-averaged irradiance."""
        return S.irradiance(self.sh, None)

    def white_xyz(self):
        """Illuminant XYZ normalised to Y=1 (chromaticity for CAT16)."""
        return white_xyz_from_rgb(self.white_rgb())


def white_xyz_from_rgb(rgb):
    xyz = C.linear_to_xyz(np.maximum(np.asarray(rgb, float), 1e-6))
    return xyz / xyz[1]
