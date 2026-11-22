"""Synthetic allocator workloads built by bootstrapping MEASURED per-group rows (bench/results/allocator_*_rows.csv, from real
core/solver tier runs). A workload of n items = n groups resampled from the pool (i.e. many independent regions of measured size);
true cost/gain are the measured values (noise included); the allocator only sees table estimates (or the bandit's)."""
from __future__ import annotations
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from core.allocator import mckp
from core.allocator.tables import Tables, N_TIERS

RES = Path(__file__).resolve().parent / "results"


@dataclass
class Pool:
    F: np.ndarray        # (m,6) group features
    E: np.ndarray
    P: np.ndarray
    deg: np.ndarray
    ms: np.ndarray       # (m,5) measured ms per tier (col 0 = 0)
    gain: np.ndarray     # (m,5) measured min dE00 gain per tier (col 0 = 0; col 1 = reuse at no scene change)

    def __len__(self):
        return len(self.F)


def load_pool(name="allocator_heldout_rows") -> Pool:
    """Items = groups measured in the 'all' regime (whole tier in one call). Reuse (tier 1) at no scene change replays the
    tier-4 solution: gain = the item's tier-4 gain, ms = mean measured tier-1 ms."""
    df = pd.read_csv(RES / f"{name}.csv")
    fc = [f"f{i}" for i in range(6)]
    keys = ["seed", "E", "P", "deg"] + fc
    ms1 = float(df[df.tier == 1].ms.mean())
    F, E, P, deg, ms, gn = [], [], [], [], [], []
    for k, g in df[(df.tier != 1) & (df.regime == "all")].groupby(keys):
        t = {int(r.tier): r for _, r in g.iterrows()}
        if not all(x in t for x in (2, 3, 4)):
            continue
        F.append(list(k[4:])); E.append(k[1]); P.append(k[2]); deg.append(k[3])
        ms.append([0, ms1] + [t[x].ms for x in (2, 3, 4)])
        gn.append([0, max(t[4].gain, 0.0)] + [max(t[x].gain, 0.0) for x in (2, 3, 4)])
    return Pool(np.array(F), np.array(E, float), np.array(P, float), np.array(deg, float), np.array(ms), np.array(gn))


@dataclass
class Workload:
    F: np.ndarray
    imp: np.ndarray
    est_ben: np.ndarray
    est_cost: np.ndarray
    true_ben: np.ndarray
    true_cost: np.ndarray
    hit: np.ndarray
    shift: np.ndarray | None = None

    @property
    def n(self):
        return len(self.imp)


def make(pool: Pool, n, rng, tables: Tables, hit_frac=0.3, gain_scale=None) -> Workload:
    """gain_scale: (5,) multipliers on TRUE gains per tier (distribution shift the static tables do not know about)."""
    i = rng.integers(len(pool), size=n)
    imp = 0.5 + 1.5 * rng.beta(1.2, 3.0, size=n)
    hit = rng.random(n) < hit_frac
    tg = pool.gain[i].copy()
    if gain_scale is not None:
        tg = tg * np.asarray(gain_scale)[None]
    tg[~hit, 1] = 0.0
    tc = pool.ms[i].copy(); tc[~hit, 1] = 0.0
    F = pool.F[i]
    est = tables.gain_matrix(F)
    est[:, 1] = np.where(hit, est[:, 4], 0.0)
    active = (F[:, 1] > 0) & (pool.deg[i] > 0)
    ec = tables.cost_matrix(pool.E[i], pool.P[i], pool.deg[i], 2.0 / max(int(active.sum()), 1))
    ec[:, 1] = np.where(hit, ec[:, 1], 0.0)
    est = est * imp[:, None] * active[:, None]
    ec = ec * active[:, None]
    est[:, 0] = ec[:, 0] = 0.0
    return Workload(F, imp, est, ec, tg * imp[:, None] * active[:, None], tc * active[:, None], hit)


def execute(wl: Workload, choice, B):
    """Run the plan under a hard budget: tier asc, estimated benefit desc; stop at the first item that would overrun B (true ms).
    -> (realised benefit, ms spent, items completed)."""
    got, spent, done = execute_idx(wl, choice, B)
    return got, spent, len(done)


def execute_idx(wl: Workload, choice, B):
    """As execute(), but returns the list of completed item indices."""
    idx = np.flatnonzero(choice > 0)
    order = idx[np.lexsort((-wl.est_ben[idx, choice[idx]], choice[idx]))]
    spent, got, done = 0.0, 0.0, []
    for i in order:
        c = wl.true_cost[i, choice[i]]
        if spent + c > B:
            break
        spent += c; got += wl.true_ben[i, choice[i]]; done.append(int(i))
    return got, spent, done


def _oracle_view(wl: Workload) -> Workload:
    """Workload whose estimates equal the truth (planning with perfect information)."""
    import copy
    o = copy.copy(wl)
    o.est_ben, o.est_cost = wl.true_ben, wl.true_cost
    return o


def run_method(name, wl: Workload, B, rr=None, bandit_est=None):
    """-> (choice, overhead ms). The overhead covers only the MCKP/baseline decision (estimates are table lookups)."""
    ben = wl.est_ben if bandit_est is None else bandit_est
    t0 = time.perf_counter()
    if name == "round_robin":
        ch = rr(ben, wl.est_cost, B)
    elif name == "bnb":
        ch = mckp.bnb(ben, wl.est_cost, B, node_limit=60_000, time_limit=0.5)[0]
    else:
        ch = mckp.SOLVERS[name](ben, wl.est_cost, B)
    return ch, (time.perf_counter() - t0) * 1e3
