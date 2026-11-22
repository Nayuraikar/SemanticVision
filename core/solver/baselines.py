"""Baselines on (N,3) lit linear RGB. Assumed observer = full dichromat (Machado sev 1) of the given kind."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from core.perception import color as C, cvd
from core.perception.graph import build_graph

_SHIFT = {"protan": [[0, 0, 0], [.7, 1, 0], [.7, 0, 1]], "deutan": [[0, 0, 0], [.7, 1, 0], [.7, 0, 1]],
          "tritan": [[1, 0, .7], [0, 1, .7], [0, 0, 0]]}


def none(colors, kind=None, **kw):
    return np.array(colors, float)


def _dalt(c, kind, strength):
    """Classic daltonisation in sRGB-encoded space: shift the CVD error into the intact channels."""
    s = C.linear_to_srgb(c)
    err = s - C.linear_to_srgb(cvd.machado(c, kind, 1.0))
    out = s + np.asarray(strength, float).reshape(-1, 1) * (err @ np.array(_SHIFT[kind], float).T)
    return C.srgb_to_linear(np.clip(out, 0, 1))


def global_daltonisation(colors, kind, **kw):
    c = np.asarray(colors, float)
    return _dalt(c, kind, np.ones(len(c)))


def local_daltonisation(colors, kind, scene=None, severity=1.0, **kw):
    """Daltonise only objects that have a confusable neighbour (assumed observer), strength scaled by how
    confusable the worst neighbour is (1 - D/tau, doubled, clipped). Without a scene: all-pairs graph."""
    c = np.asarray(colors, float)
    if scene is None:
        g = build_graph(c, model="machado", kind=kind, severity=severity)
    else:
        g = build_graph(c, scene.ids, scene.size_deg, scene.area, scene.importance, scene.bg_lum,
                        adjacency=scene.adjacency, far_weight=scene.far_weight, model="machado", kind=kind, severity=severity)
    st = np.zeros(len(c))
    for (i, j), mg, T in zip(g.edges, g.edge_margin, g.tau[tuple(g.edges.T)] if len(g.edges) else []):
        s = min(1.0, 2.0 * mg / T)
        st[i], st[j] = max(st[i], s), max(st[j], s)
    return _dalt(c, kind, st)


def kuhn2008(colors, kind, anchor=0.01, maxiter=40, **kw):
    """Kuhn, Oliveira, Fernandes 2008 (contrast-preserving recolouring), stress-minimisation form: choose
    Lab positions q_i so that the pairwise dE00 of the *simulated* recoloured colours matches the pairwise
    dE00 of the original colours (what a trichromat sees); small anchor term keeps colours near the original."""
    c = np.asarray(colors, float)
    n = len(c)
    lab0 = C.linear_to_lab(c)
    iu = np.triu_indices(n, 1)
    target = C.ciede2000(lab0[:, None], lab0[None])[iu]

    def rgb(q):
        return np.clip(C.xyz_to_linear(C.lab_to_xyz(q.reshape(n, 3))), 0, 1)

    def f(q):
        L = C.linear_to_lab(cvd.machado(rgb(q), kind, 1.0))
        d = C.ciede2000(L[:, None], L[None])[iu]
        return float(np.sum((d - target) ** 2) + anchor * np.sum((q.reshape(n, 3) - lab0) ** 2))

    bnd = [(5, 98), (-110, 110), (-110, 110)] * n
    r = minimize(f, lab0.ravel(), method="L-BFGS-B", bounds=bnd, options=dict(maxiter=maxiter))
    return rgb(r.x)


METHODS = {"none": none, "global_dalt": global_daltonisation, "local_dalt": local_daltonisation, "kuhn2008": kuhn2008}
