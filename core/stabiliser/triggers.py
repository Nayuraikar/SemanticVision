"""Event-driven recompute triggers (SPEC 5.9): detected from snapshot events AND from snapshot diffs, so a missing
event does not hide a change."""
from __future__ import annotations
from dataclasses import dataclass

ORDER = ("profile_change", "enter_view", "collision", "state_change", "lighting_change", "importance_jump", "new_pair")


def importance(o) -> float:
    return 2.0 if "hazard" in o.task_tags else 1.0


def in_view(o) -> bool:
    b = o.bbox
    return b.x + b.w > 0 and b.x < 1 and b.y + b.h > 0 and b.y < 1


def _flat(x, out):
    if isinstance(x, dict):
        for k in sorted(x):
            _flat(x[k], out)
    elif isinstance(x, (list, tuple)):
        for v in x:
            _flat(v, out)
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        out.append(float(x))


def light_vec(lighting):
    out: list = []
    _flat(lighting.model_dump(), out)
    return out


@dataclass
class TriggerConfig:
    importance_jump: float = 0.5
    lighting_eps: float = 0.05      # relative change of any lighting coefficient


class TriggerDetector:
    def __init__(self, cfg: TriggerConfig | None = None):
        self.cfg = cfg or TriggerConfig()
        self.reset()

    def reset(self):
        self.prev = None

    def detect(self, snap, new_pairs=()) -> list[str]:
        """Returns fired trigger names (ORDER order); ['init'] on the first frame after reset."""
        cur = dict(ids={o.id for o in snap.objects}, view={o.id for o in snap.objects if in_view(o)},
                   state={o.id: o.state for o in snap.objects}, imp={o.id: importance(o) for o in snap.objects},
                   light=light_vec(snap.lighting), obs=snap.observer_id)
        p, self.prev = self.prev, cur
        if p is None:
            return ["init"]
        fired = set()
        if cur["obs"] != p["obs"]:
            fired.add("profile_change")
        if cur["view"] - p["view"]:
            fired.add("enter_view")
        if any(cur["state"][i] != p["state"][i] for i in cur["state"] if i in p["state"]):
            fired.add("state_change")
        if any(abs(cur["imp"][i] - p["imp"][i]) >= self.cfg.importance_jump for i in cur["imp"] if i in p["imp"]):
            fired.add("importance_jump")
        a, b = cur["light"], p["light"]
        if len(a) != len(b) or any(abs(x - y) / (abs(y) + 0.05) > self.cfg.lighting_eps for x, y in zip(a, b)):
            fired.add("lighting_change")
        for e in snap.events:
            if e.type in ("enter_view", "collision", "state_change", "lighting_change"):
                fired.add(e.type)
            elif e.type == "trigger_enter":     # physics trigger volume: a semantic state change
                fired.add("state_change")
        if new_pairs:
            fired.add("new_pair")
        return [t for t in ORDER if t in fired]
