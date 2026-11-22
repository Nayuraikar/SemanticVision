"""Hysteresis on confusion decisions (pair ratio dE00/tau) and on cue assignment (SPEC 5.9)."""
from __future__ import annotations


class PairHysteresis:
    """A pair becomes confusing when ratio < enter and stops when ratio > exit (exit >= enter), after at least
    `dwell` frames active. A pair missing from the analysis (adjacency flap) keeps its state for `hold` frames."""

    def __init__(self, enter: float = 1.0, exit: float = 1.3, dwell: int = 0, hold: int = 0):
        if exit < enter:
            raise ValueError("exit threshold must be >= enter threshold")
        self.enter, self.exit, self.dwell, self.hold = enter, exit, dwell, hold
        self.state: dict[tuple, list] = {}   # pair -> [active, since_frame, last_seen]

    def reset(self):
        self.state.clear()

    def update(self, ratios: dict, frame: int):
        """-> (active pairs set, newly entered pairs set)."""
        new = set()
        for k, r in ratios.items():
            st = self.state.get(k)
            if st is None:
                st = self.state[k] = [False, frame, frame]
            st[2] = frame
            if not st[0] and r < self.enter:
                st[0], st[1] = True, frame
                new.add(k)
            elif st[0] and r > self.exit and frame - st[1] >= self.dwell:
                st[0], st[1] = False, frame
        for k in [k for k, st in self.state.items() if k not in ratios and frame - st[2] > self.hold]:
            del self.state[k]
        return {k for k, st in self.state.items() if st[0]}, new


def _key(c):
    return (c.type, c.parameters.get("variant"))


class CueHysteresis:
    """Cues do not flicker: a cue disappears only after `hold` frames without the solver asking for it, and an
    object keeps its previous cue (type, variant) while it is clash-free against its confusable partners."""

    def __init__(self, hold: int = 15):
        self.hold = hold
        self.cur: dict = {}        # id -> Cue
        self.seen: dict = {}       # id -> last frame the solver asked for a cue

    def reset(self):
        self.cur.clear(); self.seen.clear()

    def update(self, new: dict, pairs, frame: int, ids=None) -> dict:
        """new: id -> Cue from the solver; pairs: id pairs that need distinct cues. Returns id -> Cue."""
        for i in new:
            self.seen[i] = frame
        keep = {i: c for i, c in self.cur.items() if i in new or frame - self.seen.get(i, -10**9) <= self.hold}
        if ids is not None:
            keep = {i: c for i, c in keep.items() if i in ids}
        out = {i: (keep.get(i) or new[i]) for i in new}
        pairs = [p for p in pairs if p[0] in out and p[1] in out]
        for _ in range(3):
            bad = {i for a, b in pairs if _key(out[a]) == _key(out[b]) for i in (a, b)}
            if not bad:
                break
            for i in bad:
                out[i] = new[i]
        else:
            if any(_key(out[a]) == _key(out[b]) for a, b in pairs):
                out = dict(new)
        for i, c in keep.items():       # solver stopped asking: hold the previous cue for `hold` frames
            out.setdefault(i, c)
        self.cur = dict(out)
        return out
