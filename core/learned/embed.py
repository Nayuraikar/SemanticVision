"""Observer embedding z(theta) for the learned solver (SPEC 5.6). numpy only.
z = [w_protan, w_deutan, w_tritan, severity, log k, s_L, s_C, bg_gain]; w is a type mixture on the simplex, so a calibrated
profile (soft posterior over types) and a preset Observer (one-hot) share one space. The CVD model family is never encoded:
the network must generalise across families (held-out Brettel/Vienot)."""
from __future__ import annotations
import numpy as np
from core.observers import Observer, KINDS
from core.perception import cvd as V
from core.perception.threshold import ThresholdParams

Z_DIM = 8
Z_NAMES = ("w_protan", "w_deutan", "w_tritan", "severity", "log_k", "s_L", "s_C", "bg_gain")
BG_DEFAULT = ThresholdParams().bg_gain


def embed(w, severity, k=1.0, s_L=1.0, s_C=1.0, bg_gain=BG_DEFAULT) -> np.ndarray:
    w = np.clip(np.asarray(w, float), 0, None)
    w = w / w.sum() if w.sum() > 1e-12 else np.full(3, 1 / 3)
    return np.r_[w, float(np.clip(severity, 0, 1)), np.log(max(float(k), 1e-6)), float(s_L), float(s_C),
                 float(bg_gain)].astype(np.float32)


def embed_observer(obs: Observer, bg_gain=BG_DEFAULT) -> np.ndarray:
    return embed(np.eye(3)[KINDS.index(obs.kind)], obs.severity, obs.k, obs.s_L, obs.s_C, bg_gain)


def embed_profile(summary: dict) -> np.ndarray:
    """Calibrated profile summary (calibration Session.summary(): params[name]['mean']) -> z of the posterior mean.
    Same conventions as core.profiles.profile_observer: severity = max_i e_i, k rescaled by S=(s_L+s_C)/2."""
    p = summary["params"]
    e = np.array([p[f"e_{k}"]["mean"] for k in KINDS], float)
    sL, sC = float(p["s_L"]["mean"]), float(p["s_C"]["mean"])
    bg = float(p["bg_gain"]["mean"]) if "bg_gain" in p else BG_DEFAULT
    return embed(e, float(e.max()), float(p["k"]["mean"]) * 0.5 * (sL + sC), sL, sC, bg)


def embed_samples(samples: dict) -> np.ndarray:
    """Posterior samples (Calibrator.samples() format) -> z of the posterior mean (k geometric mean)."""
    m = lambda key: {"mean": float(np.mean(samples[key]))}
    params = {f"e_{k}": m(f"e_{k}") for k in KINDS} | {"s_L": m("s_L"), "s_C": m("s_C")}
    params["k"] = {"mean": float(np.exp(np.mean(np.log(samples["k"]))))}
    if "bg_gain" in samples:
        params["bg_gain"] = m("bg_gain")
    return embed_profile({"params": params})


def embed_many(observers) -> np.ndarray:
    """Several observers (e.g. the solver's train set) -> mean embedding."""
    return np.mean([embed_observer(o) for o in observers], 0).astype(np.float32)


def proxy_matrix(z) -> np.ndarray:
    """z -> 3x3 linear-RGB CVD proxy: type-mixture of Machado matrices at z's severity (the only simulator the deployed
    model can assume; the true observer family is unknown)."""
    z = np.asarray(z, float)
    return sum(z[i] * V.machado_matrix(k, float(np.clip(z[3], 0, 1))) for i, k in enumerate(KINDS))


def tau_params(z) -> dict:
    z = np.asarray(z, float)
    return dict(k=float(np.exp(z[4])), s_L=float(z[5]), s_C=float(z[6]), p=ThresholdParams(bg_gain=float(z[7])))
