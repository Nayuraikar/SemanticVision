"""JSON / NDJSON read-write for record/replay."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable, Iterator, Type, TypeVar
from pydantic import BaseModel
from .models import Snapshot, Overrides

M = TypeVar("M", bound=BaseModel)


def _dump(m: BaseModel) -> str:
    return m.model_dump_json(by_alias=True)


def write_json(m: BaseModel, path) -> None:
    Path(path).write_text(json.dumps(json.loads(_dump(m)), indent=2))


def read_json(path, model: Type[M] = Snapshot) -> M:
    return model.model_validate_json(Path(path).read_text())


def write_ndjson(items: Iterable[BaseModel], path) -> int:
    n = 0
    with open(path, "w") as f:
        for m in items:
            f.write(_dump(m) + "\n")
            n += 1
    return n


def read_ndjson(path, model: Type[M] = Snapshot) -> Iterator[M]:
    with open(path) as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield model.model_validate_json(line)
                except ValueError as e:
                    raise ValueError(f"{path}:{i}: {e}") from e
