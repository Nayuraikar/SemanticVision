"""JSON Schema export."""
from __future__ import annotations
import json
from pathlib import Path
from .models import Snapshot, Overrides, SCHEMA_VERSION


def export_schemas(out_dir="protocol/schemas") -> list[Path]:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, m in (("snapshot", Snapshot), ("overrides", Overrides)):
        p = d / f"{name}.v{SCHEMA_VERSION}.schema.json"
        p.write_text(json.dumps(m.model_json_schema(by_alias=True), indent=2))
        paths.append(p)
    return paths


if __name__ == "__main__":
    for p in export_schemas():
        print(p)
