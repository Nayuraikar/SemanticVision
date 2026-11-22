"""K3 ablation ladder: albedo -> SH-L2 -> +shadow -> +CAT16 -> +shadow+CAT16, scored as confusion-pair prediction
(precision/recall/F1) against the deterministic software-render truth, per lighting condition, under HELD-OUT observers
on HELD-OUT scenes. Arms: '<light>|<rung>'. Counts are pooled per seed (CIs/paired tests are over seeds)."""
from __future__ import annotations
from bench.registry import experiment
from bench.k3 import rung_vs_truth, world_seed, prf
from adapters.synthetic import WorldConfig, generate_world, LIGHT_PRESETS
from core.lighting import RUNGS
from core.observers import heldout_observers


@experiment("k3_ablation")
def run(params, seed):
    lights = params.get("lights", list(LIGHT_PRESETS))
    n_worlds, n_obs = int(params.get("n_worlds", 4)), int(params.get("n_obs", 6))
    split = params.get("scene_split", "heldout")
    rungs = params.get("rungs", list(RUNGS))
    obs = heldout_observers(n_obs, seed)
    out = {}
    for light in lights:
        acc = {r: dict(tp=0, fp=0, fn=0, n_true=0, n_pairs=0) for r in rungs}
        for k in range(n_worlds):
            cfg = WorldConfig.from_dict({**params.get("world", {}), "lighting": light})
            snap, truth, _ = generate_world(cfg, world_seed(seed, k, split))
            if len(snap.objects) < 2:
                continue
            res = rung_vs_truth(snap, truth, obs, rungs, float(params.get("adj_margin", 0.3)))
            for r in rungs:
                for key, v in res[r].items():
                    acc[r][key] += v
        for r in rungs:
            a = acc[r]
            out[f"{light}|{r}"] = {**{k: float(v) for k, v in a.items()}, **prf(a["tp"], a["fp"], a["fn"])}
    return out
