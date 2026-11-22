"""Importance: rules vs MLP (trained on the independent task observer's labels) on HELD-OUT scenes
(different palettes, lighting, object counts, seeds from the training ones). Metrics per seed: mean per-scene Spearman rho
with the labels and top-20% precision."""
from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr
from bench.registry import experiment
from adapters.synthetic import SynthConfig, generate
from core.importance import Tracker, rule_score, task_labels, train_mlp, FEATURE_NAMES, IDX

TRAIN = dict(palette=["confusable", "random"], lighting=["day", "night", "cycle"], n=[10, 14, 20])
HELD = dict(palette=["traffic", "dashboard"], lighting=["fog", "rain"], n=[30, 40])


def scenes(dist, n_scenes, seed, frames=(25, 40, 55), n_frames=56, motion=0.25):
    """-> list of (X, y) per snapshot, via a sequential Tracker per stream."""
    r = np.random.default_rng([seed, 0x1F0])
    out = []
    for _ in range(n_scenes):
        cfg = SynthConfig(n_objects=int(r.choice(dist["n"])), n_frames=n_frames, palette=str(r.choice(dist["palette"])),
                          lighting=str(r.choice(dist["lighting"])), motion=motion, state_rate=0.06)
        tr = Tracker()
        for snap in generate(cfg, int(r.integers(1 << 30))):
            X, g = tr.update(snap)
            if snap.frame in frames and len(X) > 3:
                out.append((X, task_labels(snap, g)))
    return out


def metrics(score_fn, data):
    rho, top = [], []
    for X, y in data:
        s = score_fn(X)
        if np.ptp(y) < 1e-9 or np.ptp(s) < 1e-9:
            rho.append(0.0)
        else:
            rho.append(float(spearmanr(s, y).statistic))
        k = max(1, len(y) // 5)
        top.append(len(set(np.argsort(-s)[:k]) & set(np.argsort(-y)[:k])) / k)
    return dict(spearman=float(np.mean(rho)), top20=float(np.mean(top)))


def stack(data):
    return np.vstack([d[0] for d in data]), np.concatenate([d[1] for d in data])


@experiment("importance_eval")
def run(params, seed):
    tr = scenes(TRAIN, int(params.get("n_train", 14)), seed * 1000)
    te = scenes(HELD, int(params.get("n_test", 10)), seed * 1000 + 500)
    X, y = stack(tr)
    noecc = [i for i in range(len(FEATURE_NAMES)) if i != IDX["ecc"]]
    mlp = train_mlp(X, y, seed)
    mlp_ne = train_mlp(X, y, seed, cols=noecc)
    area = lambda X: X[:, IDX["log_area"]]
    rnd = np.random.default_rng(seed)
    arms = {"rules": rule_score, "mlp": mlp, "mlp_noecc": mlp_ne, "area_only": area,
            "random": lambda X: rnd.random(len(X))}
    return {a: metrics(f, te) for a, f in arms.items()} | {"_n": {"train_samples": float(len(y))}}
