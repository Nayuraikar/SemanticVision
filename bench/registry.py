"""Experiment registry. An experiment: fn(params: dict, seed: int) -> {arm: {metric: float}}.
Later phases add bench/experiments/<name>.py with @experiment("name") plus a config JSON; no CLI change."""
from __future__ import annotations
import importlib, pkgutil
from typing import Callable

_REG: dict[str, Callable] = {}


def experiment(name: str):
    def deco(fn):
        if name in _REG and _REG[name] is not fn:
            raise ValueError(f"experiment '{name}' already registered")
        _REG[name] = fn
        return fn
    return deco


def discover():
    import bench.experiments as pkg
    for m in pkgutil.iter_modules(pkg.__path__):
        importlib.import_module(f"bench.experiments.{m.name}")


def get(name: str) -> Callable:
    discover()
    if name not in _REG:
        raise KeyError(f"unknown experiment '{name}'; known: {sorted(_REG)}")
    return _REG[name]


def names() -> list[str]:
    discover()
    return sorted(_REG)
