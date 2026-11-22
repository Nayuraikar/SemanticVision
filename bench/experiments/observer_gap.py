"""Sanity experiment: confusable-pair rate on a synthetic stream under train vs held-out observers
(no solver). Shows generator + observer sampler + CLI wiring; real experiments register the same way."""
from __future__ import annotations
import numpy as np
from bench.registry import experiment
from adapters.synthetic import SynthConfig, generate
from core.observers import train_observers, heldout_observers
from core.perception import distance, threshold


def _rate(snaps, obs, tau0=2.3):
    hits = tot = 0
    for s in snaps:
        cols = np.array([o.albedo_rgb for o in s.objects])
        D = distance.pair_matrix(cols, **obs.sim_kwargs())
        iu = np.triu_indices(len(cols), 1)
        hits += int(np.sum(D[iu] < tau0 * obs.k)); tot += len(iu[0])
    return hits / max(tot, 1)


@experiment("observer_gap")
def run(params, seed):
    cfg = SynthConfig.from_dict(params.get("scene", {}))
    snaps = list(generate(cfg, seed))[:: max(1, int(params.get("stride", 10)))]
    n = int(params.get("n_observers", 6))
    return {"train": {"confusable_rate": float(np.mean([_rate(snaps, o) for o in train_observers(n, seed)]))},
            "heldout": {"confusable_rate": float(np.mean([_rate(snaps, o) for o in heldout_observers(n, seed)]))}}
