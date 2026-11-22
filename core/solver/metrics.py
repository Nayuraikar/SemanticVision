"""Naturalness and success metrics (all under an observer's own CVD model and threshold tau)."""
from __future__ import annotations
import numpy as np
from core.perception import color as C, distance as Dst


def deviation_de00(orig, new):
    return C.ciede2000(C.linear_to_lab(np.asarray(orig, float)), C.linear_to_lab(np.asarray(new, float)))


def naturalness(orig, new, protected=None):
    """mean dE00 from original (lower = more natural), max dE00, memory-colour drift (mean/max over protected)."""
    d = deviation_de00(orig, new)
    p = np.zeros(len(d), bool) if protected is None else np.asarray(protected, bool)
    return dict(mean_de00=float(d.mean()), max_de00=float(d.max()),
                mem_drift=float(d[p].mean()) if p.any() else 0.0, mem_max=float(d[p].max()) if p.any() else 0.0)


def pair_scores(colors, obs, edges, tau, illum_xyz=None):
    """Perceived dE00 and success (D >= tau) of the given pairs under observer obs. edges (E,2), tau (N,N)."""
    if not len(edges):
        return None
    kw = obs.sim_kwargs() if illum_xyz is None else dict(obs.sim_kwargs(), illum_xyz=illum_xyz)
    D = Dst.pair_matrix(np.asarray(colors, float), **kw)
    d = D[edges[:, 0], edges[:, 1]]
    return dict(min=float(d.min()), mean=float(d.mean()), success=float(np.mean(d >= tau[edges[:, 0], edges[:, 1]])),
                failing=int(np.sum(d < tau[edges[:, 0], edges[:, 1]])))
