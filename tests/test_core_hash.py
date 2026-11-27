"""Frozen-core proof: the held-out dashboard environment runs on the UNCHANGED core. Baseline written before the
dashboard was built (tests/fixtures/core_hash.json); any changed, added or removed file fails. No regenerate switch."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "tests/fixtures/core_hash.json"
DIRS = ("core", "protocol", "calibration", "learning")
EXTRA = ("core/allocator/tables.json", "core/allocator/learned.onnx", "core/importance/mlp.npz")


def current_hashes():
    files = [p for d in DIRS for p in (ROOT / d).rglob("*.py")] + [ROOT / f for f in EXTRA if (ROOT / f).exists()]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def diff_against_baseline():
    base, cur = json.loads(BASELINE.read_text()), current_hashes()
    return dict(added=sorted(set(cur) - set(base)), removed=sorted(set(base) - set(cur)),
                changed=sorted(k for k in set(base) & set(cur) if base[k] != cur[k]))


def assert_core_unchanged():
    d = diff_against_baseline()
    assert d == dict(added=[], removed=[], changed=[]), f"frozen core differs from baseline: {d}"


def test_core_matches_baseline():
    assert set(EXTRA) <= set(json.loads(BASELINE.read_text()))
    assert_core_unchanged()
