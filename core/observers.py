"""Observer sampler. Train: Machado, severity in TRAIN_SEV, nominal thresholds.
Held-out: Brettel (all kinds) or Vienot (protan/deutan), severity in HELD_SEV (disjoint from TRAIN_SEV),
plus threshold noise (k, s_L, s_C jitter). Disjoint by construction (model family differs)."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np

KINDS = ("protan", "deutan", "tritan")
TRAIN_SEV = (0.5, 1.0)
HELD_SEV = (0.1, 0.5)   # half-open at the top: sampled in [0.1, 0.5)
HELD_MODELS = {"protan": ("brettel", "vienot"), "deutan": ("brettel", "vienot"), "tritan": ("brettel",)}
NOISE_SIGMA = 0.25


@dataclass(frozen=True)
class Observer:
    id: str
    split: str          # "train" | "heldout"
    model: str          # machado | brettel | vienot
    kind: str
    severity: float
    k: float = 1.0      # threshold scale (noise)
    s_L: float = 1.0
    s_C: float = 1.0

    def sim_kwargs(self) -> dict:
        return dict(model=self.model, kind=self.kind, severity=self.severity)

    def key(self) -> tuple:
        return (self.model, self.kind, round(self.severity, 6))

    def to_dict(self) -> dict:
        return asdict(self)


def _rng(seed, split):
    return np.random.default_rng([int(seed), 0 if split == "train" else 1, 0x0B5])


def sample_observers(n: int, seed: int, split: str = "train") -> list[Observer]:
    if split not in ("train", "heldout"):
        raise ValueError("split must be 'train' or 'heldout'")
    r = _rng(seed, split)
    out, seen = [], set()
    while len(out) < n:
        kind = KINDS[int(r.integers(3))]
        if split == "train":
            o = Observer("", split, "machado", kind, round(float(r.uniform(*TRAIN_SEV)), 4))
        else:
            model = HELD_MODELS[kind][int(r.integers(len(HELD_MODELS[kind])))]
            sev = round(float(r.uniform(*HELD_SEV)), 4)
            if sev >= HELD_SEV[1]:
                continue
            o = Observer("", split, model, kind, sev, k=float(np.exp(r.normal(0, NOISE_SIGMA))),
                         s_L=float(np.exp(r.normal(0, NOISE_SIGMA))), s_C=float(np.exp(r.normal(0, NOISE_SIGMA))))
        if o.key() in seen:
            continue
        seen.add(o.key())
        out.append(Observer(**{**o.to_dict(), "id": f"{split[:2]}-{seed}-{len(out):03d}"}))
    return out


def train_observers(n, seed=0): return sample_observers(n, seed, "train")
def heldout_observers(n, seed=0): return sample_observers(n, seed, "heldout")
