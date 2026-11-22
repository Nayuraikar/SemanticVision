"""Confusion graph: nodes (objects), edges (pairs) with D, tau, adjacency, weight."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from . import distance as D
from .threshold import tau as tau_fn, ThresholdParams


@dataclass
class ConfusionGraph:
    ids: list
    n: int
    D: np.ndarray            # (N,N) perceived distance
    tau: np.ndarray          # (N,N) threshold
    adjacency: np.ndarray    # (N,N) bool
    size_weight: np.ndarray  # (N,N)
    weight: np.ndarray       # (N,N) edge weight (0 for non-edges)
    confusable: np.ndarray   # (N,N) bool, D<tau, off-diagonal
    edges: np.ndarray        # (E,2) int, i<j, confusable & weight>0
    edge_weight: np.ndarray  # (E,)
    edge_margin: np.ndarray  # (E,) tau-D (>0)

    def neighbours(self, i):
        m = self.weight[i] > 0
        m[i] = False
        return np.flatnonzero(m & self.confusable[i])


def bbox_adjacency(bbox, margin=0.0):
    """bbox (N,4) x0,y0,x1,y1 -> bool (N,N): boxes closer than margin (touching/overlapping/near)."""
    b = np.asarray(bbox, float)
    gx = np.maximum(b[:, None, 0] - b[None, :, 2], b[None, :, 0] - b[:, None, 2])
    gy = np.maximum(b[:, None, 1] - b[None, :, 3], b[None, :, 1] - b[:, None, 3])
    adj = np.maximum(gx, gy) <= margin
    np.fill_diagonal(adj, False)
    return adj


def build_graph(colors, ids=None, size_deg=None, area=None, importance=None, bg_lum=0.2,
                adjacency=None, bbox=None, adj_margin=0.0, far_weight=0.0,
                s_L=1.0, s_C=1.0, k=1.0, params: ThresholdParams = ThresholdParams(),
                space="de2000", min_weight=1e-9, **perceive_kw) -> ConfusionGraph:
    """colors: (N,3) lit linear RGB. size_deg: (N,) projected angular size; area: (N,) screen area
    (size weighting; defaults to size_deg**2). Pair tau uses the smaller object's size and mean bg.
    Weight = adj_factor * size_w * sqrt(imp_i imp_j); adj_factor=1 if adjacent else far_weight."""
    c = np.asarray(colors, float)
    N = len(c)
    ids = list(range(N)) if ids is None else list(ids)
    sz = np.full(N, 2.0) if size_deg is None else np.asarray(size_deg, float)
    ar = sz ** 2 if area is None else np.asarray(area, float)
    imp = np.ones(N) if importance is None else np.asarray(importance, float)
    bg = np.broadcast_to(np.asarray(bg_lum, float), (N,))
    if adjacency is None:
        adjacency = bbox_adjacency(bbox, adj_margin) if bbox is not None else np.ones((N, N), bool)
    adj = np.asarray(adjacency, bool) | np.asarray(adjacency, bool).T
    np.fill_diagonal(adj, False)
    Dm = D.pair_matrix(c, space=space, **perceive_kw)
    pair_sz = np.minimum(sz[:, None], sz[None, :])
    pair_bg = 0.5 * (bg[:, None] + bg[None, :])
    sL = np.broadcast_to(np.asarray(s_L, float), (N,)).min() if np.ndim(s_L) else s_L
    sC = np.broadcast_to(np.asarray(s_C, float), (N,)).min() if np.ndim(s_C) else s_C
    T = tau_fn(pair_sz, pair_bg, sL, sC, k, params)
    sw = np.sqrt(ar[:, None] * ar[None, :]) / max(ar.max(), 1e-12)
    adjf = np.where(adj, 1.0, far_weight)
    W = adjf * sw * np.sqrt(imp[:, None] * imp[None, :])
    off = ~np.eye(N, dtype=bool)
    conf = (Dm < T) & off
    W = np.where(off, W, 0.0)
    iu = np.triu(conf & (W > min_weight), 1)
    e = np.argwhere(iu)
    return ConfusionGraph(ids, N, Dm, T, adj, sw, W, conf, e,
                          W[iu], (T - Dm)[iu])
