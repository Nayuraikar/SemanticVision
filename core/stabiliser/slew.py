"""Perceptual slew limiter: dE00(p(t), p(t-1)) <= r_max * dt per entry (SPEC 5.9)."""
from __future__ import annotations
import numpy as np
from .metrics import de00


def slew_limit(prev, target, max_de, iters: int = 30):
    """prev, target (N,3) linear RGB; max_de scalar or (N,). Moves each row from prev toward target along the RGB
    segment, as far as dE00(prev, out) <= max_de. Bisection keeps the last *verified* point, so the bound holds by
    construction (out is evaluated by the same expression as the check). Returns (out (N,3), clamped (N,) bool)."""
    prev = np.asarray(prev, float).reshape(-1, 3)
    target = np.asarray(target, float).reshape(-1, 3)
    if not len(prev):
        return prev.copy(), np.zeros(0, bool)
    lim = np.broadcast_to(np.asarray(max_de, float), (len(prev),))
    clamped = de00(prev, target) > lim
    lo, hi = np.zeros(len(prev)), np.ones(len(prev))
    d = target - prev
    for _ in range(iters):
        mid = (lo + hi) / 2
        ok = de00(prev, prev + mid[:, None] * d) <= lim
        lo, hi = np.where(ok, mid, lo), np.where(ok, hi, mid)
    t = np.where(clamped, lo, 1.0)
    return prev + t[:, None] * d, clamped
