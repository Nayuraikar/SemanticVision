"""Confusion threshold tau(size, background luminance, lightness/chroma sensitivity, k)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ThresholdParams:
    tau0: float = 2.3        # dE00 just-noticeable difference for a large field, ref background, unit sensitivity
    size_ref_deg: float = 2.0  # field size (deg) at which size factor is 1+size_gain*1
    size_gain: float = 1.0
    size_exp: float = 0.5
    size_cap: float = 8.0
    bg_ref: float = 0.2       # reference background relative luminance (Y, 0-1)
    bg_gain: float = 0.25     # per decade of luminance away from reference
    w_L: float = 0.5          # weight of lightness sensitivity in combined sensitivity
    w_C: float = 0.5


def tau(size_deg, bg_lum, s_L=1.0, s_C=1.0, k=1.0, p: ThresholdParams = ThresholdParams()):
    """tau = k * tau0 * f_size * f_bg / S.  Larger when smaller, on a darker/brighter-than-reference
    background (|log10| Weber-like), or for a less sensitive observer (S = w_L s_L + w_C s_C, 1 = normal).
    k is a calibrated global scale. size_deg: projected angular size; bg_lum: relative luminance Y."""
    size = np.maximum(np.asarray(size_deg, float), 1e-6)
    f_size = np.minimum(1.0 + p.size_gain * (p.size_ref_deg / size) ** p.size_exp, p.size_cap)
    f_bg = 1.0 + p.bg_gain * np.abs(np.log10(np.maximum(np.asarray(bg_lum, float), 1e-4) / p.bg_ref))
    S = np.maximum(p.w_L * np.asarray(s_L, float) + p.w_C * np.asarray(s_C, float), 1e-3)
    return k * p.tau0 * f_size * f_bg / S
