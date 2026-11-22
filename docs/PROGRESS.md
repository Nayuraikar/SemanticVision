# Progress

## Phase: protocol (SPEC 4.1-4.5)
- Done: protocol/{models,builder,io,schema}.py, protocol/schemas/*.v1.0.schema.json, docs/SCOPE.md, tests/test_protocol.py (14 pass).
- API: Snapshot, Overrides, SceneObject, SHLighting|AmbientDominantLighting, Event, PaletteOverride, Cue, ConfusionPair, Diagnostics; SCHEMA_VERSION="1.0" (major-version check).
- API: SnapshotBuilder(observer_id,budget_ms).{sh2,ambient_dominant,add_object,add_event,build}; write_json/read_json(path,model); write_ndjson(items,path)/read_ndjson(path,model); export_schemas(dir).
- Wire: JSON uses key "class" (alias of SceneObject.cls); dump with by_alias=True. RGB linear; albedo in [0,1], emissive HDR >=0.
- Gaps: no protobuf; no per-class state enums; lit_color unvalidated vs lighting.
