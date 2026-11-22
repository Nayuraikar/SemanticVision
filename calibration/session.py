"""Interactive calibration session: next stimulus (max EIG), answer, stop rule, replay from stored trials."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
from . import stimuli as S
from .bayes import Calibrator, MODES
from .model import Prior, SelfReport


@dataclass(frozen=True)
class CalibConfig:
    seed: int = 0
    mode: str = "mixed"            # mixed (alternate same/different and slider) | same_different | slider
    target_width: float = 0.4    # stop when max normalised 90% width of correction-relevant params <= this
    min_trials: int = 12
    max_trials: int = 50
    n_particles: int = 1200

    @staticmethod
    def from_dict(d: dict | None) -> "CalibConfig":
        d = {k: v for k, v in (d or {}).items() if k in CalibConfig.__dataclass_fields__}
        c = CalibConfig(**d)
        if c.mode not in ("mixed",) + MODES:
            raise ValueError(f"mode must be mixed|same_different|slider, got {c.mode}")
        if not (c.min_trials >= 1 and c.max_trials >= c.min_trials and c.target_width > 0 and c.n_particles >= 100):
            raise ValueError("bad calibration config")
        return c

    def to_dict(self) -> dict:
        return asdict(self)


class Session:
    def __init__(self, cfg: CalibConfig | None = None, self_report: dict | None = None):
        self.cfg = cfg or CalibConfig()
        self.cal = Calibrator(self.cfg.seed, self.cfg.n_particles, Prior.from_self_report(SelfReport.from_dict(self_report)))
        self.pending: dict | None = None
        self._width = self.cal.width()

    @classmethod
    def replay(cls, cfg: CalibConfig, self_report: dict | None, trials: list[tuple[int, str, object]]) -> "Session":
        s = cls(cfg, self_report)
        for stim, mode, resp in trials:
            s.cal.update(stim, mode, resp)
        s._width = s.cal.width()
        return s

    @property
    def n(self) -> int:
        return self.cal.n

    @property
    def done(self) -> bool:
        return self.n >= self.cfg.max_trials or (self.n >= self.cfg.min_trials and self._width <= self.cfg.target_width)

    def progress(self) -> dict:
        return dict(n=self.n, max_trials=self.cfg.max_trials, min_trials=self.cfg.min_trials, width=self._width,
                    target_width=self.cfg.target_width, done=self.done)

    def _mode(self) -> str:
        return ("same_different", "slider")[self.n % 2] if self.cfg.mode == "mixed" else self.cfg.mode

    def next(self) -> dict | None:
        if self.done:
            return None
        if self.pending is None:
            mode = self._mode()
            stim, gain = self.cal.select(mode)
            p, size, bg = S.decode(stim)
            pl = self.cal.pool
            self.pending = dict(trial_id=self.n, stim=stim, mode=mode, c1=[float(x) for x in pl.c1[p]],
                                c2=[float(x) for x in pl.c2[p]], size_deg=size, bg_lum=bg, axis=S.AXES[pl.axis[p]],
                                swap=bool(self.n % 3 == 1), eig_bits=gain)
        return {k: v for k, v in self.pending.items() if k != "stim"}

    def answer(self, trial_id: int, response) -> tuple[int, str, object]:
        """Returns the recorded (stim, mode, response). Raises KeyError (no pending / id mismatch), ValueError (bad response)."""
        if self.pending is None or self.pending["trial_id"] != trial_id:
            raise KeyError("no such pending trial")
        mode, stim = self.pending["mode"], self.pending["stim"]
        if mode == "same_different":
            if response not in ("same", "different"):
                raise ValueError("response must be 'same' or 'different'")
        else:
            if isinstance(response, bool) or not isinstance(response, (int, float)) or not 0 <= float(response) <= 1:
                raise ValueError("slider response must be a number in [0,1]")
        self.cal.update(stim, mode, response)
        self.pending = None
        self._width = self.cal.width()
        return stim, mode, self.cal.history[-1][2]

    def summary(self, full: bool = False) -> dict:
        return self.cal.summary(full)
