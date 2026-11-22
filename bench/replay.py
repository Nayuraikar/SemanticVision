"""Record / replay of snapshot streams as NDJSON (engine-free)."""
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Iterator
from protocol import Snapshot, write_ndjson, read_ndjson


def record(snapshots: Iterable[Snapshot], path) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return write_ndjson(snapshots, path)


def replay(path) -> Iterator[Snapshot]:
    return read_ndjson(path, Snapshot)


def record_synthetic(cfg, seed: int, path) -> int:
    from adapters.synthetic import generate
    return record(generate(cfg, seed), path)
