"""Versioned snapshot/override contract (SPEC 4.2-4.3). Engine-agnostic."""
from __future__ import annotations
import math
from typing import Annotated, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0"

RGB = Annotated[tuple[float, float, float], Field(description="linear RGB in [0,1]")]
EventType = Literal["state_change", "collision", "enter_view", "exit_view", "lighting_change"]
CueType = Literal["outline", "hatch", "icon", "label"]
SolverTier = Literal["exact", "learned", "cached", "fallback", "none", "allocated"]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _check_rgb(v, allow_hdr=False):
    for c in v:
        if not math.isfinite(c) or c < 0 or (not allow_hdr and c > 1):
            raise ValueError(f"rgb component out of range: {c}")
    return v


class BBox(_Base):
    x: float
    y: float
    w: float = Field(ge=0)
    h: float = Field(ge=0)


class SceneObject(_Base):
    id: str = Field(min_length=1)
    cls: str = Field(alias="class", min_length=1)
    state: str = "default"
    albedo_rgb: RGB
    emissive_rgb: RGB = (0.0, 0.0, 0.0)
    screen_area: float = Field(ge=0, le=1, description="fraction of viewport")
    bbox: BBox
    depth: float = Field(ge=0)
    velocity: tuple[float, float] = (0.0, 0.0)
    task_tags: list[str] = Field(default_factory=list)
    lit_color: Optional[RGB] = None
    normal: Optional[tuple[float, float, float]] = Field(None, description="world surface normal facing the camera")
    position: Optional[tuple[float, float, float]] = Field(None, description="world position (same frame as lights)")
    shadow_vis: Optional[float] = Field(None, ge=0, le=1, description="visibility of the dominant light (1 lit, 0 shadow)")

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @field_validator("albedo_rgb", "lit_color")
    @classmethod
    def _rgb01(cls, v):
        return v if v is None else _check_rgb(v)

    @field_validator("emissive_rgb")
    @classmethod
    def _emis(cls, v):
        return _check_rgb(v, allow_hdr=True)


class DominantLight(_Base):
    """Dominant (sun/lamp) term of the lighting. rgb = normalised irradiance on a surface facing it (lit = albedo*rgb*cos).
    direction = direction towards the light (nominal); position set => point light, rgb is its value at ref_dist."""
    rgb: RGB
    direction: tuple[float, float, float]
    position: Optional[tuple[float, float, float]] = None
    ref_dist: float = Field(1.0, gt=0)

    @field_validator("rgb")
    @classmethod
    def _hdr(cls, v):
        return _check_rgb(v, allow_hdr=True)


class SHLighting(_Base):
    """coeffs: radiance SH-L2 (real basis, l<=2) of ALL lighting incl. the dominant term. Normalised diffuse
    irradiance E(n) = sum_l (1, 2/3, 1/4)_l c_lm Y_lm(n); lit = albedo*E*exposure + emissive*exposure."""
    kind: Literal["sh2"] = "sh2"
    coeffs: list[RGB] = Field(min_length=9, max_length=9, description="9 SH-L2 coeffs x RGB")
    dominant: Optional[DominantLight] = None
    white_point: tuple[float, float, float] = (0.9505, 1.0, 1.089)  # XYZ, D65
    exposure: float = Field(1.0, gt=0)


class AmbientDominantLighting(_Base):
    kind: Literal["ambient_dominant"] = "ambient_dominant"
    ambient_rgb: RGB
    dominant_rgb: RGB
    dominant_dir: tuple[float, float, float]
    dominant_pos: Optional[tuple[float, float, float]] = None
    white_point: tuple[float, float, float] = (0.9505, 1.0, 1.089)
    exposure: float = Field(1.0, gt=0)

    @field_validator("ambient_rgb", "dominant_rgb")
    @classmethod
    def _hdr(cls, v):
        return _check_rgb(v, allow_hdr=True)

    @field_validator("dominant_dir")
    @classmethod
    def _dir(cls, v):
        n = math.sqrt(sum(c * c for c in v))
        if not math.isfinite(n) or n < 1e-9:
            raise ValueError("dominant_dir must be non-zero")
        return v


Lighting = Annotated[Union[SHLighting, AmbientDominantLighting], Field(discriminator="kind")]


class Event(_Base):
    type: EventType
    object_id: Optional[str] = None
    data: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _needs_obj(self):
        if self.type != "lighting_change" and not self.object_id:
            raise ValueError(f"{self.type} requires object_id")
        return self


class Snapshot(_Base):
    schema_version: str = SCHEMA_VERSION
    frame: int = Field(0, ge=0)
    t: float = Field(0.0, ge=0)
    objects: list[SceneObject] = Field(default_factory=list)
    lighting: Lighting
    events: list[Event] = Field(default_factory=list)
    observer_id: str = Field(min_length=1)
    budget_ms: float = Field(gt=0)
    session_id: Optional[str] = Field(None, min_length=1, max_length=128, description="stabiliser session; omitted = new session")

    @field_validator("schema_version")
    @classmethod
    def _ver(cls, v):
        if v.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
            raise ValueError(f"unsupported schema_version {v}; expected {SCHEMA_VERSION}.x")
        return v

    @model_validator(mode="after")
    def _refs(self):
        ids = [o.id for o in self.objects]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate object ids")
        s = set(ids)
        for e in self.events:
            if e.object_id is not None and e.object_id not in s:
                raise ValueError(f"event references unknown object {e.object_id}")
        return self


class PaletteOverride(_Base):
    id: str
    albedo: RGB
    emissive: Optional[RGB] = None

    @field_validator("albedo")
    @classmethod
    def _a(cls, v):
        return _check_rgb(v)

    @field_validator("emissive")
    @classmethod
    def _e(cls, v):
        return v if v is None else _check_rgb(v, allow_hdr=True)


class Cue(_Base):
    id: str
    type: CueType
    parameters: dict = Field(default_factory=dict)


class ConfusionPair(_Base):
    ids: tuple[str, str]
    de00_before: float = Field(ge=0)
    de00_after: float = Field(ge=0)
    feasible: bool

    @field_validator("ids")
    @classmethod
    def _distinct(cls, v):
        if v[0] == v[1]:
            raise ValueError("pair ids must differ")
        return v


class Diagnostics(_Base):
    stage_latency_ms: dict[str, float] = Field(default_factory=dict)
    solver_tier: SolverTier = "none"
    fallback_flags: list[str] = Field(default_factory=list)
    triggers: list[str] = Field(default_factory=list, description="recompute triggers that fired this frame")
    slew_clamped: int = Field(0, ge=0, description="palette channels clamped by the slew limiter this frame")
    session_id: Optional[str] = None
    tier_counts: dict[str, int] = Field(default_factory=dict, description="palette groups per allocator tier (skip/reuse/learned/coarse/fine)")
    allocator_ms: float = Field(0.0, ge=0, description="allocator's own latency (features + MCKP), inside stage_latency_ms.solve")
    entering: list[str] = Field(default_factory=list, description="object ids predicted to enter view (correction precomputed)")

    @field_validator("stage_latency_ms")
    @classmethod
    def _lat(cls, v):
        if any((not math.isfinite(x)) or x < 0 for x in v.values()):
            raise ValueError("latency must be finite and >= 0")
        return v


class Overrides(_Base):
    schema_version: str = SCHEMA_VERSION
    frame: int = Field(0, ge=0)
    palette_overrides: list[PaletteOverride] = Field(default_factory=list)
    cues: list[Cue] = Field(default_factory=list)
    confusion_pairs: list[ConfusionPair] = Field(default_factory=list)
    diagnostics: Diagnostics = Field(default_factory=Diagnostics)

    _ver = field_validator("schema_version")(Snapshot._ver.__func__)  # type: ignore[attr-defined]

    @model_validator(mode="after")
    def _uniq(self):
        ids = [p.id for p in self.palette_overrides]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate palette override ids")
        return self
