"""K3 helpers (SPEC 5.2): rung predictions vs rendered truth; CIEDE2000 vs CAM16-UCS. No thresholds are tuned here:
tau uses ThresholdParams defaults for every rung, metric and split; the only fitted constant (CAM16/dE00 scale) is
calibrated on seeded surface colours, never on evaluation worlds."""
from __future__ import annotations
from functools import lru_cache
import numpy as np
from adapters.synthetic import WorldConfig, generate_world, TRAIN_LIGHTS, HELDOUT_LIGHTS, LIGHT_PRESETS
from core.lighting import LightField, rung_colors, rung_illum, RUNGS, white_xyz_from_rgb
from core.perception import color as C, distance as Dst
from core.perception.graph import build_graph, bbox_adjacency

HELD_SEED_BASE = 1_000_000       # held-out scene seeds; train scenes use seed*100+k
FOV = 60.0


def world_seed(seed, k, split="heldout"):
    return (HELD_SEED_BASE if split == "heldout" else 0) + int(seed) * 100 + int(k)


def geometry(snap, adj_margin=0.3):
    o = snap.objects
    bb = np.array([(x.bbox.x, x.bbox.y, x.bbox.w, x.bbox.h) for x in o], float)
    area = np.maximum(bb[:, 2] * bb[:, 3], 1e-6)
    xyxy = np.c_[bb[:, 0], bb[:, 1], bb[:, 0] + bb[:, 2], bb[:, 1] + bb[:, 3]]
    return dict(size=np.maximum(np.sqrt(area) * FOV, 0.1), area=area, adj=bbox_adjacency(xyxy, adj_margin))


def confusable_pairs(colors, geo, obs, bg_lum, illum_xyz=None, space="de2000", tau_scale=1.0):
    """-> (pairs (E,2) adjacent i<j, D (E,), tau (E,), confusable bool (E,)) under the observer."""
    kw = dict(obs.sim_kwargs())
    if illum_xyz is not None:
        kw["illum_xyz"] = illum_xyz
    g = build_graph(colors, None, geo["size"], geo["area"], None, bg_lum, adjacency=geo["adj"], far_weight=0.0,
                    s_L=obs.s_L, s_C=obs.s_C, k=obs.k, space=space, **kw)
    iu = np.triu(g.weight > 1e-9, 1)
    pr = np.argwhere(iu)
    d, t = g.D[iu], g.tau[iu] * tau_scale
    return pr, d, t, d < t


def truth_colors(snap, truth):
    return np.array([truth["lit"][o.id] for o in snap.objects]).reshape(-1, 3)


def truth_illum(truth):
    return white_xyz_from_rgb(truth["white_rgb"])


def counts(pred, true):
    return dict(tp=int((pred & true).sum()), fp=int((pred & ~true).sum()), fn=int((~pred & true).sum()),
                n_true=int(true.sum()), n_pairs=int(len(true)))


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    r = tp / (tp + fn) if tp + fn else 1.0
    f = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    return dict(precision=p, recall=r, f1=f)


def rung_vs_truth(snap, truth, observers, rungs=RUNGS, adj_margin=0.3):
    """-> {rung: {tp,fp,fn,n_true,n_pairs}} pooled over observers. Truth = rendered colours + CAT16 to the rendered
    white. All rungs share geometry, observers and tau (background luminance from the render), so only colour differs."""
    fld = LightField.from_lighting(snap.lighting)
    geo = geometry(snap, adj_margin)
    tc, ti = truth_colors(snap, truth), truth_illum(truth)
    cols = {r: (rung_colors(fld, snap.objects, r), rung_illum(fld, r)) for r in rungs}
    out = {r: dict(tp=0, fp=0, fn=0, n_true=0, n_pairs=0) for r in rungs}
    for o in observers:
        _, _, _, true = confusable_pairs(tc, geo, o, truth["bg_lum"], ti)
        for r, (c, ill) in cols.items():
            _, _, _, pred = confusable_pairs(c, geo, o, truth["bg_lum"], ill)
            for k, v in counts(pred, true).items():
                out[r][k] += v
    return out


@lru_cache(maxsize=None)
def cam16_tau_scale(seed=0, n=600):
    """Median CAM16-UCS / dE00 distance ratio for near-threshold (dE00 1..6) pairs of seeded reflective colours.
    Puts the dE00-calibrated tau on CAM16-UCS's scale. Calibration data: seeded random colours (no evaluation world)."""
    r = np.random.default_rng([seed, 0xCA16])
    rgb = r.uniform(0.03, 0.9, (n, 3))
    lab = C.linear_to_lab(rgb)
    v = r.normal(size=(n, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
    lab2 = lab + v * r.uniform(1, 6, (n, 1))
    rgb2 = np.clip(C.xyz_to_linear(C.lab_to_xyz(lab2)), 0, 1)
    d00 = C.ciede2000(C.linear_to_lab(rgb), C.linear_to_lab(rgb2))
    dc = Dst.distance(rgb, rgb2, "cam16")
    return float(np.median(dc / np.maximum(d00, 1e-6)))
