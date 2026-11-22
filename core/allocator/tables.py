"""Measured cost/benefit tables (written by `python -m bench.allocator_calibrate` from real runs of core/solver; machine-specific).
cost[t]   : ms PER GROUP = coef . [1, E, P, deg, 1/k]  (E palette entries, P critical-graph pairs, deg incident pairs, k groups
            batched in the same tier call: a whole tier runs in one call, which amortises the descent)
gain[t]   : min dE00 gain on the group's critical pairs = max(coef . GAIN_FEATS, 0)   (t = 2,3,4)
reuse     : decay_de = dE00 colour change at which a cached correction keeps 1/e of its benefit."""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

PATH = Path(__file__).with_name("tables.json")
GAIN_FEATS = ("1", "deficit", "deficit^2", "log1p(deg)", "log(size_deg)", "protected")
COST_FEATS = ("1", "E", "P", "deg", "1/k")
N_TIERS = 5


@dataclass
class Tables:
    cost: dict = field(default_factory=dict)      # "1".."4" -> 4 coefs
    gain: dict = field(default_factory=dict)      # "2".."4" -> 6 coefs
    reuse: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def cost_matrix(self, E, P, deg, kinv=1.0):
        deg = np.asarray(deg, float)
        out = np.zeros((len(deg), N_TIERS))
        X = np.stack([np.ones_like(deg), np.broadcast_to(np.asarray(E, float), deg.shape),
                      np.broadcast_to(np.asarray(P, float), deg.shape), deg,
                      np.broadcast_to(np.asarray(kinv, float), deg.shape)], 1)
        for t in range(1, N_TIERS):
            out[:, t] = np.maximum(X @ np.asarray(self.cost[str(t)]), 1e-3)
        return out

    def gain_matrix(self, F):
        """F (n,6) -> (n,5) estimated gain per tier (tier 0 and 1 columns 0; reuse handled by the allocator)."""
        out = np.zeros((len(F), N_TIERS))
        for t in (2, 3, 4):
            out[:, t] = np.maximum(F @ np.asarray(self.gain[str(t)]), 0.0)
        return out

    def save(self, path=PATH):
        Path(path).write_text(json.dumps(dict(cost=self.cost, gain=self.gain, reuse=self.reuse, meta=self.meta), indent=1))

    @classmethod
    def load(cls, path=PATH):
        d = json.loads(Path(path).read_text())
        return cls(d["cost"], d["gain"], d["reuse"], d.get("meta", {}))


def load_tables(path=PATH):
    """Measured tables, or None if calibration has not been run (the allocator then falls back to exact tier 4 for all)."""
    try:
        return Tables.load(path)
    except FileNotFoundError:
        return None
