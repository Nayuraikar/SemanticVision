import json
import pytest
from pydantic import ValidationError
from protocol import (Snapshot, Overrides, SnapshotBuilder, write_json, read_json,
                      write_ndjson, read_ndjson, export_schemas, SCHEMA_VERSION)


def snap(frame=0, **kw):
    b = SnapshotBuilder("obs1", 4.0, frame=frame).ambient_dominant((0.2,) * 3, (1, 1, 0.9))
    b.add_object("a", "cup", (0.8, 0.1, 0.1), bbox=(0.1, 0.1, 0.2, 0.2), task_tags=["grab"])
    b.add_object("b", "lamp", (0.1, 0.8, 0.1), emissive_rgb=(2, 2, 2))
    b.add_event("state_change", "a", to="on")
    return b.build()


def test_builder_defaults():
    s = snap()
    assert s.schema_version == SCHEMA_VERSION
    assert s.objects[0].screen_area == pytest.approx(0.04)


def test_sh2_and_alias():
    s = SnapshotBuilder("o", 1).sh2([(0.1, 0.1, 0.1)] * 9).add_object("x", "c", (0, 0, 0)).build()
    d = json.loads(s.model_dump_json(by_alias=True))
    assert d["objects"][0]["class"] == "c" and d["lighting"]["kind"] == "sh2"


def test_sh2_wrong_len():
    with pytest.raises(ValueError):
        SnapshotBuilder("o", 1).sh2([(0, 0, 0)] * 4)


def test_no_lighting():
    with pytest.raises(ValueError):
        SnapshotBuilder("o", 1).build()


@pytest.mark.parametrize("bad", [(1.5, 0, 0), (-0.1, 0, 0), (float("nan"), 0, 0)])
def test_bad_albedo(bad):
    with pytest.raises(ValueError):
        SnapshotBuilder("o", 1).add_object("x", "c", bad)


def test_dup_ids_and_event_ref():
    b = SnapshotBuilder("o", 1).ambient_dominant((0,) * 3, (1,) * 3)
    b.add_object("x", "c", (0, 0, 0)).add_object("x", "c", (0, 0, 0))
    with pytest.raises(ValueError, match="duplicate"):
        b.build()
    b = SnapshotBuilder("o", 1).ambient_dominant((0,) * 3, (1,) * 3).add_event("collision", "zz")
    with pytest.raises(ValueError, match="unknown object"):
        b.build()
    with pytest.raises(ValueError):
        SnapshotBuilder("o", 1).add_event("collision")


def test_budget_and_version():
    d = json.loads(snap().model_dump_json(by_alias=True))
    d["budget_ms"] = 0
    with pytest.raises(ValidationError):
        Snapshot.model_validate(d)
    d["budget_ms"] = 1
    d["schema_version"] = "2.0"
    with pytest.raises(ValidationError):
        Snapshot.model_validate(d)
    d["schema_version"] = "1.7"
    Snapshot.model_validate(d)
    d["bogus"] = 1
    with pytest.raises(ValidationError):
        Snapshot.model_validate(d)


def test_overrides():
    o = Overrides.model_validate({
        "palette_overrides": [{"id": "a", "albedo": [0.5, 0.5, 0.5]}],
        "cues": [{"id": "a", "type": "outline", "parameters": {"w": 2}}],
        "confusion_pairs": [{"ids": ["a", "b"], "de00_before": 3, "de00_after": 12, "feasible": True}],
        "diagnostics": {"stage_latency_ms": {"solve": 1.2}, "solver_tier": "exact"}})
    assert o.diagnostics.solver_tier == "exact"
    with pytest.raises(ValidationError):
        Overrides.model_validate({"cues": [{"id": "a", "type": "sparkle"}]})
    with pytest.raises(ValidationError):
        Overrides.model_validate({"confusion_pairs": [{"ids": ["a", "a"], "de00_before": 1,
                                                       "de00_after": 1, "feasible": False}]})
    with pytest.raises(ValidationError):
        Overrides.model_validate({"palette_overrides": [{"id": "a", "albedo": [0, 0, 0]}] * 2})
    with pytest.raises(ValidationError):
        Overrides.model_validate({"diagnostics": {"stage_latency_ms": {"x": -1}}})


def test_json_roundtrip(tmp_path):
    s = snap()
    write_json(s, tmp_path / "s.json")
    assert read_json(tmp_path / "s.json") == s
    o = Overrides()
    write_json(o, tmp_path / "o.json")
    assert read_json(tmp_path / "o.json", Overrides) == o


def test_ndjson_roundtrip(tmp_path):
    ss = [snap(i) for i in range(5)]
    assert write_ndjson(ss, tmp_path / "r.ndjson") == 5
    assert list(read_ndjson(tmp_path / "r.ndjson")) == ss


def test_ndjson_bad_line(tmp_path):
    p = tmp_path / "r.ndjson"
    p.write_text(snap().model_dump_json(by_alias=True) + "\n{bad\n")
    with pytest.raises(ValueError, match=":2:"):
        list(read_ndjson(p))


def test_schema_export(tmp_path):
    ps = export_schemas(tmp_path)
    assert len(ps) == 2
    for p in ps:
        j = json.loads(p.read_text())
        assert "properties" in j
    s = json.loads(ps[0].read_text())
    assert "schema_version" in s["properties"] and "class" in json.dumps(s)
