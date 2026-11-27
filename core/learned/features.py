"""Scene + z -> learned-solver inputs (numpy only; shared by training and the ONNX tier).
Objects carry their palette group's colour (labg) and deviation bound, so the shared-palette constraint is exact: the model
shares its residual through S (row-normalised same-group matrix) and the projection is a per-object function of (labg, delta)."""
from __future__ import annotations
import numpy as np
from core.perception import color as C
from core.perception.threshold import tau as tau_fn
from .embed import proxy_matrix, tau_params

F_DIM = 24
LAB_SCALE = np.array([100.0, 128.0, 128.0])
PAD_LAB = np.array([50.0, 0.0, 0.0])


def cat_rgb_matrix(illum_xyz, L_A=64 / np.pi * 0.2, F=1.0) -> np.ndarray:
    """Linear-RGB 3x3 of core.perception.distance.perceived's CAT16 step (identity without a white point)."""
    if illum_xyz is None:
        return np.eye(3)
    M = C.cat16_matrix(illum_xyz, C.D65_XYZ, float(np.clip(C.cat16_degree(L_A, F), 0, 1)))
    return C.M_XYZ2RGB @ M @ C.M_RGB2XYZ


def proxy_lab(rgb, z, illum_xyz=None):
    M = cat_rgb_matrix(illum_xyz)
    r = np.clip(np.clip(np.asarray(rgb, float) @ M.T, 0, 1) @ proxy_matrix(z).T, 0, 1)
    return C.linear_to_lab(r)


def group_arrays(scene, delta=15.0, delta_mem=3.0):
    """-> labg (N,3) group Lab per object, delta (N,), S (N,N), A (N,N) adjacency between objects of different groups."""
    col, sz, ar, im, adj, prot = scene.palette()
    g = scene.groups
    labg = C.linear_to_lab(col)[g]
    dl = np.where(prot, delta_mem, delta)[g]
    same = g[:, None] == g[None, :]
    S = same * scene.area[None, :]
    S = S / S.sum(1, keepdims=True)
    A = np.asarray(scene.adjacency, bool) & ~same
    return labg, dl, S, A, col


def scene_inputs(scene, z, delta=15.0, delta_mem=3.0) -> dict:
    """-> dict(x (N,F), z (8,), labg (N,3), delta (N,), S (N,N), A (N,N) float32)."""
    labg, dl, S, A, col = group_arrays(scene, delta, delta_mem)
    N = len(labg)
    pl = proxy_lab(col, z, scene.illum_xyz)[scene.groups]
    tp = tau_params(z)
    psz = np.minimum(scene.size_deg[:, None], scene.size_deg[None, :])
    T = tau_fn(psz, scene.bg_lum, tp["s_L"], tp["s_C"], tp["k"], tp["p"])
    R = np.where(A, C.ciede2000(pl[:, None], pl[None]) / T, np.inf)
    deg = A.sum(1)
    j = np.argmin(R, 1)
    rmin = np.where(deg > 0, R[np.arange(N), j], 3.0)
    has = (deg > 0)[:, None]
    dpl = np.where(has, pl[j] - pl, 0.0) / 20.0
    dlab = np.where(has, labg[j] - labg, 0.0) / 20.0
    tmin = np.where(deg > 0, np.where(A, T, np.inf).min(1), T.max())
    deficit = np.where(A, np.maximum(0.0, 1.0 - np.where(np.isfinite(R), R, 9.0)), 0.0).sum(1) / np.maximum(deg, 1)
    w = np.zeros(3) if scene.illum_xyz is None else np.asarray(scene.illum_xyz, float)
    wxy = w[:2] / max(w.sum(), 1e-9) - np.array([0.3127, 0.3290]) if w.sum() > 0 else np.zeros(2)
    x = np.c_[labg / LAB_SCALE, pl / LAB_SCALE, np.log(np.maximum(scene.size_deg, 1e-3)),
              np.sqrt(scene.area / scene.area.max()), scene.importance, scene.protected.astype(float), dl / 15.0,
              np.log1p(deg), np.log(tmin), np.minimum(rmin, 3.0), dpl, dlab, deficit,
              np.full(N, scene.bg_lum), np.tile(wxy * 10, (N, 1))]
    assert x.shape[1] == F_DIM
    f = lambda a: np.asarray(a, np.float32)
    return dict(x=f(x), z=f(z), labg=f(labg), delta=f(dl), S=f(S), A=f(A))
