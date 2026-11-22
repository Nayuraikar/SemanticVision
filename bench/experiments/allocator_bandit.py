"""Static MCKP (measured tables) vs MCKP with LinUCB-learned benefit, over sequential episodes of held-out workloads (n=200,
budget 25% of the all-fine cost). 'shift' scales the TRUE per-tier gains (tier 3 x0.4, tier 4 x1.6) -- something the static tables
cannot know and the bandit can learn from realised gains. Metrics: fraction of the oracle's benefit, early (first 10 episodes)
and late (last 20) -- the early gap is the bandit's exploration cost."""
from __future__ import annotations
import numpy as np
from bench.registry import experiment
from bench import alloc_workloads as W
from core.allocator import mckp, load_tables, LinUCB

SHIFT = np.array([1, 1, 1.0, 0.4, 1.6])


@experiment("allocator_bandit")
def run(params, seed):
    pool = W.load_pool(params.get("pool", "allocator_heldout_rows"))
    tb = load_tables()
    n, K, f = int(params.get("n", 200)), int(params.get("episodes", 30)), float(params.get("frac", 0.25))
    out = {}
    for name, scale in (("noshift", None), ("shift", SHIFT)):
        rng = np.random.default_rng([seed, 0xBA2])
        bandit = LinUCB(6, alpha=float(params.get("alpha", 1.0)), decay=float(params.get("decay", 0.95)))
        res = {"static": [], "bandit": [], "oracle": []}
        for ep in range(K):
            wl = W.make(pool, n, rng, tb, gain_scale=scale)
            B = f * float(wl.true_cost[:, 4].sum())
            act = (wl.F[:, 1] > 0)[:, None]
            est = bandit.estimate(wl.F) * wl.imp[:, None] * act
            est[:, 1] = np.where(wl.hit, est[:, 1], 0.0)
            ch_b, _ = W.run_method("lagrangian", wl, B, bandit_est=est)
            got_b, _, done = W.execute_idx(wl, ch_b, B)
            for i in done:                                    # online feedback: realised (unweighted) gain of executed items
                bandit.update(wl.F[i], int(ch_b[i]), wl.true_ben[i, ch_b[i]] / wl.imp[i])
            bandit.alpha *= bandit.decay
            ch_s, _ = W.run_method("lagrangian", wl, B)
            got_s, _, _ = W.execute(wl, ch_s, B)
            o = W._oracle_view(wl)
            got_o, _, _ = W.execute(o, mckp.lagrangian(wl.true_ben, wl.true_cost, B), B)
            for k, v in (("static", got_s), ("bandit", got_b), ("oracle", got_o)):
                res[k].append(v)
        o = np.array(res["oracle"])
        for m in ("static", "bandit"):
            v = np.array(res[m])
            out[f"{m}|{name}"] = dict(late=float(v[-20:].sum() / o[-20:].sum()), early=float(v[:10].sum() / o[:10].sum()),
                                      overall=float(v.sum() / o.sum()))
    return out
