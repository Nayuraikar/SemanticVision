"""Multiple-choice knapsack: one tier per item, maximise sum benefit s.t. sum cost <= B. Arrays b, c: (n, T), tier 0 = (0, 0).
solve(b, c, B) -> (n,) int tier indices. Every solver returns a FEASIBLE choice (sum cost <= B)."""
from __future__ import annotations
import time
import numpy as np


def total(b, c, ch):
    i = np.arange(len(ch))
    return float(b[i, ch].sum()), float(c[i, ch].sum())


def hull_steps(b, c):
    """Per item, the upper convex hull of (cost, benefit) points -> list of (item, tier, dcost, dbenefit) upgrade steps with
    strictly decreasing density within an item (dominated and LP-dominated tiers pruned)."""
    steps = []
    for i in range(len(b)):
        order = np.lexsort((-b[i], c[i]))
        h, last = [], (0.0, 0.0, 0)          # tier 0 baseline
        pts = [(0.0, 0.0, 0)] + [(c[i, t], b[i, t], t) for t in order if t != 0 and b[i, t] > 0]
        for p in pts[1:]:
            while h and (p[1] - h[-1][1]) * (h[-1][0] - (h[-2][0] if len(h) > 1 else 0.0)) >= \
                    (h[-1][1] - (h[-2][1] if len(h) > 1 else 0.0)) * (p[0] - h[-1][0]):
                h.pop()
            if p[1] > (h[-1][1] if h else 0.0) + 1e-12:
                h.append(p)
        prev = (0.0, 0.0)
        for cc, bb, t in h:
            steps.append((i, t, cc - prev[0], bb - prev[1]))
            prev = (cc, bb)
    return steps


def uniform(b, c, B):
    n, T = b.shape
    for t in range(T - 1, -1, -1):
        if c[:, t].sum() <= B + 1e-12:
            return np.full(n, t)
    return np.zeros(n, int)


class RoundRobin:
    """Stateful: frame k serves items k*stride.. cyclically at the best tier that still fits; the rest get tier 0."""

    def __init__(self):
        self.ptr = 0

    def __call__(self, b, c, B):
        n, T = b.shape
        ch, rem, served = np.zeros(n, int), float(B), 0
        for k in range(n):
            i = (self.ptr + k) % n
            for t in range(T - 1, 0, -1):
                if c[i, t] <= rem + 1e-12 and b[i, t] > 0:
                    ch[i], rem, served = t, rem - c[i, t], served + 1
                    break
        self.ptr = (self.ptr + max(served, 1)) % n
        return ch


def greedy(b, c, B):
    """Naive greedy by benefit density: every (item, tier) option ranked by b/c; take if its item is unassigned and it fits."""
    n, T = b.shape
    dens = np.where(c > 1e-12, b / np.maximum(c, 1e-12), np.where(b > 0, np.inf, 0.0))
    dens[:, 0] = 0.0
    order = np.argsort(-dens, axis=None, kind="stable")
    ch, rem, done = np.zeros(n, int), float(B), np.zeros(n, bool)
    for f in order:
        i, t = divmod(int(f), T)
        if dens[i, t] <= 0:
            break
        if not done[i] and c[i, t] <= rem + 1e-12:
            ch[i], done[i], rem = t, True, rem - c[i, t]
    return ch


def lagrangian(b, c, B, iters=40):
    """Convex-hull pruning + Lagrangian relaxation: bisect lambda so the per-item argmax(b - lambda c) fits B; then fill the
    residual budget with hull upgrade steps by density."""
    n, T = b.shape
    pick = lambda lam: np.argmax(b - lam * c, axis=1)
    cost = lambda ch: c[np.arange(n), ch].sum()
    if cost(pick(0.0)) <= B:
        return pick(0.0)
    hi = float(np.max(np.where(c > 1e-12, b / np.maximum(c, 1e-12), 0.0))) * 1.001 + 1e-9
    lo = 0.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if cost(pick(mid)) > B else (lo, mid)
    ch = pick(hi)
    rem = B - cost(ch)
    steps = [s for s in hull_steps(b, c) if s[1] > ch[s[0]] and s[3] > 0]
    steps.sort(key=lambda s: -s[3] / max(s[2], 1e-12))
    for i, t, _, _ in steps:
        extra = c[i, t] - c[i, ch[i]]
        if t > ch[i] and b[i, t] > b[i, ch[i]] and extra <= rem + 1e-12:
            rem -= extra; ch[i] = t
    return ch


def dp(b, c, B, bins=None):
    """Exact on the cost grid: costs rounded UP to B/bins units (so the result is feasible); O(n * bins * T)."""
    n, T = b.shape
    bins = bins or int(min(max(400, 2 * n), 4000))      # resolution grows with n but is capped (DP cost O(n*bins*T))
    if B <= 0:
        return np.zeros(n, int)
    u = B / bins
    ci = np.ceil(c / u - 1e-9).astype(int)
    f = np.zeros(bins + 1)
    back = np.zeros((n, bins + 1), np.int8)
    for i in range(n):
        best, arg = f + 0.0, np.zeros(bins + 1, np.int8)
        for t in range(1, T):
            k = ci[i, t]
            if k > bins or b[i, t] <= 0:
                continue
            cand = np.full(bins + 1, -np.inf)
            cand[k:] = f[:bins + 1 - k] + b[i, t]
            m = cand > best
            best, arg = np.where(m, cand, best), np.where(m, t, arg).astype(np.int8)
        f, back[i] = best, arg
    ch, w = np.zeros(n, int), bins
    for i in range(n - 1, -1, -1):
        t = int(back[i, w]); ch[i] = t; w -= ci[i, t] if t else 0
    return ch


def lp_bound(b, c, B):
    """Fractional (LP) relaxation value: upper bound on any integral solution."""
    steps = sorted(hull_steps(b, c), key=lambda s: -s[3] / max(s[2], 1e-12))
    v, rem = 0.0, float(B)
    for _, _, dc_, db_ in steps:
        if dc_ <= rem:
            v, rem = v + db_, rem - dc_
        else:
            return v + db_ * rem / dc_
    return v


def bnb(b, c, B, node_limit=300_000, time_limit=5.0):
    """Branch and bound (offline optimum for small n). Bound at a node = LP relaxation (fractional hull-step fill) over the
    items not yet decided, with the remaining budget. Returns (choice, proven_optimal)."""
    n, T = b.shape
    inc = lagrangian(b, c, B)
    best, best_ch = total(b, c, inc)[0], inc.copy()
    order = np.argsort(-b.max(1), kind="stable")
    pos = np.empty(n, int); pos[order] = np.arange(n)
    st = sorted(hull_steps(b, c), key=lambda s: -s[3] / max(s[2], 1e-12))
    sp = np.array([pos[s[0]] for s in st]); sc = np.array([s[2] for s in st]); sb = np.array([s[3] for s in st])
    cur, nodes, t0, proven = np.zeros(n, int), [0], time.perf_counter(), [True]

    def bound(k, rem):
        m = sp >= k
        cc, bb = sc[m], sb[m]
        cs = np.cumsum(cc)
        j = int(np.searchsorted(cs, rem, side="right"))
        v = bb[:j].sum()
        if j < len(cc):
            v += bb[j] * (rem - (cs[j - 1] if j else 0.0)) / max(cc[j], 1e-12)
        return v

    def rec(k, rem, val):
        nonlocal best, best_ch
        if nodes[0] > node_limit or (nodes[0] % 1024 == 0 and time.perf_counter() - t0 > time_limit):
            proven[0] = False
            return
        nodes[0] += 1
        if val > best + 1e-12:
            best, best_ch = val, cur.copy()
        if k == n or val + bound(k, rem) <= best + 1e-12:
            return
        i = order[k]
        for t in np.argsort(-b[i], kind="stable"):
            if c[i, t] <= rem + 1e-12:
                cur[i] = t
                rec(k + 1, rem - c[i, t], val + b[i, t])
        cur[i] = 0

    rec(0, float(B), 0.0)
    return best_ch, proven[0]


SOLVERS = {"uniform": uniform, "greedy": greedy, "lagrangian": lagrangian, "dp": dp}
