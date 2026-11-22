"""Non-colour cue CSP (SPEC 5.5.1): adjacent confusable (infeasible) groups must get different cues."""
from __future__ import annotations
import numpy as np

VOCAB = [("outline", "dashed"), ("outline", "double"), ("hatch", "diag"), ("hatch", "cross"),
         ("icon", "triangle"), ("icon", "circle"), ("label", "A"), ("label", "B")]


def assign_cues(pairs, vocab=VOCAB):
    """pairs: iterable of (g_i, g_j) infeasible by colour. Greedy vertex cover (an uncued group differs from
    any cued one), then DSATUR colouring of the cover so adjacent cued groups differ.
    Returns (cues {group: (type, variant)}, unresolved pairs (cue clash))."""
    pairs = [tuple(sorted(p)) for p in pairs if p[0] != p[1]]
    pairs = sorted(set(pairs))
    adj = {}
    for a, b in pairs:
        adj.setdefault(a, set()).add(b); adj.setdefault(b, set()).add(a)
    cover, left = set(), set(pairs)
    while left:
        deg = {}
        for a, b in left:
            deg[a] = deg.get(a, 0) + 1; deg[b] = deg.get(b, 0) + 1
        v = max(sorted(deg), key=lambda x: deg[x])
        cover.add(v)
        left = {p for p in left if v not in p}
    colour = {}
    order = set(cover)
    while order:
        sat = {v: len({colour[u] for u in adj[v] if u in colour}) for v in order}
        v = max(sorted(order), key=lambda x: (sat[x], len(adj[x] & cover)))
        used = {colour[u] for u in adj[v] if u in colour}
        free = [k for k in range(len(vocab)) if k not in used]
        colour[v] = free[0] if free else min(range(len(vocab)), key=lambda k: sum(colour.get(u) == k for u in adj[v]))
        order.remove(v)
    bad = [p for p in pairs if p[0] in colour and p[1] in colour and colour[p[0]] == colour[p[1]]]
    return {g: vocab[k] for g, k in colour.items()}, bad
