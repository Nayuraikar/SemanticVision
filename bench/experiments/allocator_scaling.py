"""Allocator scaling on synthetic workloads built from measured tier rows (held-out scenes: seeds 9000+, not used to fit the
tables). Per (n items, budget): realised importance-weighted gain (true measured values) at EQUAL budget for every method,
fraction of the LP upper bound on the TRUE values, quality per ms, allocator overhead (ms and fraction of the budget).
Arms: <method>|n<N>|<budget>, budget f<frac of the all-fine cost> or t<absolute ms>."""
from __future__ import annotations
import numpy as np
from bench.registry import experiment
from bench import alloc_workloads as W
from core.allocator import mckp, load_tables
from core.allocator.mckp import RoundRobin

METHODS = ("uniform", "round_robin", "greedy", "dp", "lagrangian")


@experiment("allocator_scaling")
def run(params, seed):
    pool = W.load_pool(params.get("pool", "allocator_heldout_rows"))
    tb = load_tables()
    out = {}
    for n in params.get("sizes", [50, 200, 1000, 5000]):
        rng = np.random.default_rng([seed, n, 0xA11])
        wl = W.make(pool, n, rng, tb)
        full = float(wl.true_cost[:, 4].sum())
        buds = [(f"f{f}", f * full) for f in params.get("fracs", [0.1, 0.25, 0.5])] + \
               [(f"t{int(t)}", float(t)) for t in params.get("abs_ms", [33, 100])]
        rr = RoundRobin(); rr.ptr = int(rng.integers(n))
        for bn, B in buds:
            trueB = (wl.true_ben, wl.true_cost)
            lp = mckp.lp_bound(wl.true_ben, wl.true_cost, B)
            ora = mckp.lagrangian(wl.true_ben, wl.true_cost, B)
            og, _, _ = W.execute(W._oracle_view(wl), ora, B)
            arms = {}
            for m in METHODS + (("bnb",) if n <= int(params.get("bnb_max", 50)) and bn.startswith("f") else ()):
                ch, ov = W.run_method(m, wl, B, rr)
                got, spent, done = W.execute(wl, ch, B)
                arms[m] = dict(benefit=got, frac_lp=got / max(lp, 1e-12), per_ms=got / max(spent, 1e-9), overhead_ms=ov,
                               overhead_frac=ov / B, spent_ms=spent, items_done=done)
            arms["oracle"] = dict(benefit=og, frac_lp=og / max(lp, 1e-12), per_ms=0.0, overhead_ms=0.0, overhead_frac=0.0,
                                  spent_ms=0.0, items_done=0)
            for m, d in arms.items():
                out[f"{m}|n{n}|{bn}"] = d
    return out


def _oracle(wl):
    """Workload view whose estimates equal the truth (planning with perfect information)."""
    import copy
    o = copy.copy(wl)
    o.est_ben, o.est_cost = wl.true_ben, wl.true_cost
    return o
