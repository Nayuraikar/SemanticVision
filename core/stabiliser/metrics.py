"""Perceptual distance on linear-RGB colours (HDR allowed) and the flicker metric (SPEC 5.9)."""
from __future__ import annotations
import numpy as np
from core.perception import color as C


def de00(a, b):
    """CIEDE2000 between linear-RGB colours (broadcasting, (...,3)); negatives clipped, HDR values kept."""
    a = np.maximum(np.asarray(a, float), 0.0)
    b = np.maximum(np.asarray(b, float), 0.0)
    return C.ciede2000(C.linear_to_lab(a), C.linear_to_lab(b))


def flicker(series):
    """series (T,N,3) displayed colours -> per-entry sum over frames of dE00(c_t, c_{t-1}), shape (N,)."""
    x = np.asarray(series, float)
    if len(x) < 2:
        return np.zeros(x.shape[1] if x.ndim == 3 else 0)
    return de00(x[1:], x[:-1]).sum(0)
