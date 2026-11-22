"""Double-buffered palette: the renderer reads the last *validated* palette; the next one is built off to the side
and swapped in (one reference assignment under a lock) only after verification (SPEC 5.9)."""
from __future__ import annotations
import hashlib
import threading
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping, Optional


def _digest(version, frame, dt, albedo, emissive, cues) -> str:
    h = hashlib.sha1(repr((version, frame, round(dt, 12))).encode())
    for d in (albedo, emissive):
        h.update(repr(sorted(d.items())).encode())
    h.update(repr(sorted((i, c.model_dump_json()) for i, c in cues.items())).encode())
    return h.hexdigest()


@dataclass(frozen=True)
class Palette:
    """Immutable validated palette: id -> albedo rgb, id -> emissive rgb, id -> Cue."""
    version: int
    frame: int
    dt: float
    albedo: Mapping
    emissive: Mapping
    cues: Mapping
    digest: str
    ref: Mapping = None             # id -> (albedo, emissive) slew reference the candidate was limited against

    @classmethod
    def build(cls, version, frame, dt, albedo, emissive, cues=None, ref=None):
        a = {i: tuple(float(x) for x in v) for i, v in albedo.items()}
        e = {i: tuple(float(x) for x in v) for i, v in emissive.items()}
        c = dict(cues or {})
        return cls(version, frame, dt, MappingProxyType(a), MappingProxyType(e), MappingProxyType(c),
                   _digest(version, frame, dt, a, e, c), MappingProxyType(dict(ref or {})))

    def intact(self) -> bool:
        return self.digest == _digest(self.version, self.frame, self.dt, self.albedo, self.emissive, self.cues)


class PaletteBuilder:
    """Mutable back-buffer under construction; invisible to readers until frozen, staged and committed."""

    def __init__(self):
        self.albedo, self.emissive, self.cues = {}, {}, {}

    def set(self, id, albedo, emissive):
        self.albedo[id], self.emissive[id] = tuple(albedo), tuple(emissive)

    def set_cue(self, id, cue):
        self.cues[id] = cue

    def freeze(self, version, frame, dt) -> Palette:
        return Palette.build(version, frame, dt, self.albedo, self.emissive, self.cues)


class DoubleBuffer:
    def __init__(self):
        self._lock = threading.Lock()
        self._front: Optional[Palette] = None
        self._back: Optional[Palette] = None
        self.swaps = 0
        self.rejected = 0

    def read(self) -> Optional[Palette]:
        with self._lock:
            return self._front

    def stage(self, p: Palette):
        with self._lock:
            self._back = p

    def commit(self, verify: Optional[Callable[[Palette, Optional[Palette]], bool]] = None) -> bool:
        """Verify the staged palette against the current front; swap on success, else keep the front."""
        with self._lock:
            back, self._back = self._back, None
            if back is None:
                return False
            if not back.intact() or (verify is not None and not verify(back, self._front)):
                self.rejected += 1
                return False
            self._front = back
            self.swaps += 1
            return True
