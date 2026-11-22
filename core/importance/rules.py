"""Rule-based importance in [0,1] from class, state, state-change rate, motion, collision proximity,
screen area, distance, task tags. Hand-set weights; no eccentricity (not in the SPEC 5.7 list)."""
from __future__ import annotations
import numpy as np
from .features import CLASSES, IDX

CLASS_W = {"pedestrian": 1.0, "car": 0.8, "light": 0.8, "indicator": 0.6, "sign": 0.4, "wall": 0.1}


def rule_score(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    wc = X[:, :len(CLASSES)] @ np.array([CLASS_W[c] for c in CLASSES])
    base = wc * (0.5 + 0.5 * X[:, IDX["state_crit"]])
    dyn = 0.3 * np.minimum(X[:, IDX["change_rate"]], 1.0) + 0.2 * np.minimum(X[:, IDX["speed"]] / 0.3, 1.0) \
        + 0.5 * X[:, IDX["collision"]]
    tag = 0.3 * X[:, IDX["hazard"]] + 0.18 * X[:, IDX["signal"]] + 0.06 * X[:, IDX["info"]]
    area = np.sqrt(np.exp(X[:, IDX["log_area"]]))
    vis = (0.5 + 0.5 * np.minimum(area / 0.15, 1.0)) / np.sqrt(1.0 + X[:, IDX["depth"]] / 30.0)
    return np.clip((base + dyn + tag) * vis / 1.6, 0.0, 1.0)
