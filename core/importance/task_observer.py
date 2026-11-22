"""INDEPENDENT computational task observer (SPEC 5.7.3): signal-detection d'(size, eccentricity) + reaction-time penalty.
Label = task value of colour = task weight x [P(success | colour intact) - P(success | colour degraded)].
It must NOT import the confusion model, CVD simulation, thresholds or the solver (asserted in tests by an AST scan):
only numpy/scipy and the engine-free Geometry. Labels therefore cannot be graded by the system's own model.

Parameters (P = published, A = assumed/tuned, flagged honestly):
 P  E2 = 2.5 deg: eccentricity at which acuity size-threshold doubles, s_thr(E) = S0 (1 + E/E2)
    (Levi, Klein & Aitchison 1985, Vision Res 25; E2 ~ 0.5-3 deg by task, Strasburger et al. 2011, J Vis 11(5):13).
 P  S0 = 0.15 deg: foveal size for reliable identification of a colour-coded target; letter acuity 0.083 deg
    (Snellen 5 arcmin) x ~2 for chromatic acuity (Mullen 1985, J Physiol 359).
 P  d' = D_MAX (1 - 2^-(s/s_thr)^BETA): Weibull psychometric, slope BETA = 3.5 (Quick 1974; Strasburger 2001),
    D_MAX = 4.65 (yes/no d' at 99% correct, Macmillan & Creelman 2005). Hit prob = Phi(d'/2) (unbiased criterion; Green & Swets 1966).
 P  Crowding: flankers within Bouma spacing 0.5 E reduce d' (Bouma 1970 Nature 226; Pelli & Tillman 2008 Nat Neurosci);
    A: d'_eff = d' / (1 + 0.4 n), n <= 3.
 P  RT = RT0 + HICK log2(1 + n_distractors) + K_D / max(d', .3) (+ saccade latency 0.2 s beyond SACC_ECC deg):
    Hick 1952 / Hyman 1953 slope ~0.15 s/bit; saccade latency ~200 ms (Carpenter 1988); A: K_D = 0.35, RT0 = 0.25 s.
 A  Colour-degraded d' multiplier GAMMA: 0.2 for objects whose state is carried by colour (light, indicator), else 0.6.
 A  Deadline T: 2.0 s; 0.8 s right after a state change (transient must be acted on); success = Phi(d'/2) * sigmoid((T - RT)/0.15).
 A  Stakes of a miss grow as a collision looms: x (1 + 2 exp(-ttc/1 s)); time-to-contact does not shorten T (a shortened deadline
    makes both arms fail and wrongly drives the label to 0).
 A  Task weights per tag (profile 'driving'): hazard 1.0, signal 0.7, info 0.3, none 0.1 (wall 0.05)."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.stats import norm
from .features import Geometry

PROFILES = {"driving": {"hazard": 1.0, "signal": 0.7, "info": 0.3, "none": 0.1},
            "monitoring": {"hazard": 0.5, "signal": 1.0, "info": 0.6, "none": 0.1}}
STATE_CODED = {"light", "indicator"}


@dataclass
class TaskParams:
    E2: float = 2.5
    S0: float = 0.15
    beta: float = 3.5
    d_max: float = 4.65
    crowd_k: float = 0.4
    rt0: float = 0.25
    hick: float = 0.15
    k_d: float = 0.35
    sacc_ecc: float = 5.0
    sacc: float = 0.2
    gamma_coded: float = 0.2
    gamma_other: float = 0.6
    T: float = 2.0
    T_change: float = 0.8
    change_window: float = 0.5
    rt_slope: float = 0.15
    profile: str = "driving"


def dprime(size, ecc, crowd, p: TaskParams = TaskParams()):
    thr = p.S0 * (1.0 + np.asarray(ecc, float) / p.E2)
    d = p.d_max * (1.0 - 2.0 ** (-(np.asarray(size, float) / thr) ** p.beta))
    return d / (1.0 + p.crowd_k * np.minimum(crowd, 3))


def reaction_time(d, ecc, n_distr, p: TaskParams = TaskParams()):
    return p.rt0 + p.hick * np.log2(1.0 + n_distr) + p.k_d / np.maximum(d, 0.3) + np.where(np.asarray(ecc) > p.sacc_ecc, p.sacc, 0.0)


def success(d, rt, T, p: TaskParams = TaskParams()):
    return norm.cdf(d / 2.0) / (1.0 + np.exp(-(T - rt) / p.rt_slope))


def deadline(g: Geometry, p: TaskParams = TaskParams()):
    return np.where(g.change_age < p.change_window, p.T_change, p.T)


def task_weights(snap, p: TaskParams = TaskParams()):
    w = PROFILES[p.profile]
    out = []
    for o in snap.objects:
        t = [w[x] for x in o.task_tags if x in w]
        v = max(t) if t else w["none"]
        out.append(0.5 * v if o.cls == "wall" else v)
    return np.array(out)


def labels(snap, g: Geometry, p: TaskParams = TaskParams()):
    """-> (N,) task value of colour in [0,1] per object."""
    n = g.n
    if n == 0:
        return np.zeros(0)
    coded = np.array([o.cls in STATE_CODED for o in snap.objects])
    gamma = np.where(coded, p.gamma_coded, p.gamma_other)
    d1 = dprime(g.size, g.ecc, g.crowd, p)
    d0 = d1 * gamma
    T = deadline(g, p)
    nd = max(n - 1, 0)
    s1 = success(d1, reaction_time(d1, g.ecc, nd, p), T, p)
    s0 = success(d0, reaction_time(d0, g.ecc, nd, p), T, p)
    stakes = 1.0 + 2.0 * np.exp(-np.where(np.isfinite(g.ttc), g.ttc, 1e9))
    return np.clip(task_weights(snap, p) * stakes / 3.0 * (s1 - s0), 0.0, 1.0)
