"""Snapshot builder helper."""
from __future__ import annotations
from .models import (Snapshot, SceneObject, BBox, Event, SHLighting,
                     AmbientDominantLighting, Lighting)


class SnapshotBuilder:
    def __init__(self, observer_id: str, budget_ms: float, lighting: Lighting | None = None,
                 frame: int = 0, t: float = 0.0):
        self.observer_id, self.budget_ms, self.frame, self.t = observer_id, budget_ms, frame, t
        self.lighting = lighting
        self._objs: list[SceneObject] = []
        self._events: list[Event] = []

    def ambient_dominant(self, ambient, dominant, direction=(0, 1, 0), **kw):
        self.lighting = AmbientDominantLighting(ambient_rgb=ambient, dominant_rgb=dominant,
                                                dominant_dir=direction, **kw)
        return self

    def sh2(self, coeffs, **kw):
        self.lighting = SHLighting(coeffs=coeffs, **kw)
        return self

    def add_object(self, id, cls, albedo_rgb, bbox=(0, 0, 0.1, 0.1), screen_area=None,
                   depth=1.0, state="default", emissive_rgb=(0, 0, 0), velocity=(0, 0),
                   task_tags=(), lit_color=None, normal=None, position=None, shadow_vis=None):
        b = BBox(x=bbox[0], y=bbox[1], w=bbox[2], h=bbox[3])
        self._objs.append(SceneObject(
            id=id, **{"class": cls}, state=state, albedo_rgb=albedo_rgb, emissive_rgb=emissive_rgb,
            screen_area=b.w * b.h if screen_area is None else screen_area, bbox=b, depth=depth,
            velocity=velocity, task_tags=list(task_tags), lit_color=lit_color,
            normal=normal, position=position, shadow_vis=shadow_vis))
        return self

    def add_event(self, type, object_id=None, **data):
        self._events.append(Event(type=type, object_id=object_id, data=data))
        return self

    def build(self) -> Snapshot:
        if self.lighting is None:
            raise ValueError("lighting not set")
        return Snapshot(frame=self.frame, t=self.t, objects=self._objs, lighting=self.lighting,
                        events=self._events, observer_id=self.observer_id, budget_ms=self.budget_ms)
