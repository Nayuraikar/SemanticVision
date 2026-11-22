import ast
import json
from pathlib import Path

import numpy as np
import pytest

from adapters.synthetic import SynthConfig, generate
from core.importance import (Tracker, rule_score, task_labels, geometry, dprime, reaction_time, TaskParams, train_mlp,
                             NumpyMLP, FEATURE_NAMES, IDX, load_scorer, MODEL_PATH)
from core.importance import task_observer

ROOT = Path(__file__).resolve().parents[1]


def test_task_observer_is_independent():
    """Non-circular labels: the observer module (and its geometry input) must not import the confusion model or solver."""
    banned = ("core.perception", "core.solver", "core.lighting", "core.allocator", "calibration", "core.observers")
    for f in ("task_observer.py", "features.py"):
        tree = ast.parse((ROOT / "core/importance" / f).read_text())
        mods = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)] + \
               [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        assert not [m for m in mods if m.startswith(banned)], (f, mods)


def test_dprime_psychophysics_shape():
    s = np.array([0.05, 0.2, 0.8, 3.0])
    d = dprime(s, 0.0, 0)
    assert np.all(np.diff(d) >= 0) and d[1] > d[0] and d[-1] <= 4.65 + 1e-9
    de = dprime(0.4, np.array([0.0, 5.0, 20.0, 40.0]), 0)
    assert np.all(np.diff(de) <= 0) and de[0] > de[-1]
    assert dprime(0.5, 5.0, 3) < dprime(0.5, 5.0, 0)                      # crowding
    assert reaction_time(2.0, 10.0, 5) > reaction_time(2.0, 0.0, 5) > reaction_time(2.0, 0.0, 1)
    assert reaction_time(1.0, 0.0, 3) > reaction_time(3.0, 0.0, 3)         # weaker signal -> slower


def _stream(seed=1, n=12, frames=40):
    return list(generate(SynthConfig(n_objects=n, n_frames=frames, palette="traffic", motion=0.3, state_rate=0.1), seed))


def test_labels_and_rules_ranges():
    tr = Tracker()
    for s in _stream():
        X, g = tr.update(s)
    y = task_labels(s, g)
    assert X.shape == (len(s.objects), len(FEATURE_NAMES)) and y.shape == (len(s.objects),)
    assert np.all((y >= 0) & (y <= 1)) and y.max() > 0
    r = rule_score(X)
    assert np.all((r >= 0) & (r <= 1))
    walls = [i for i, o in enumerate(s.objects) if o.cls == "wall"]
    peds = [i for i, o in enumerate(s.objects) if o.cls in ("pedestrian", "car")]
    if walls and peds:
        assert r[walls].mean() < r[peds].mean()


def test_tracker_state_change_rate():
    snaps = _stream(frames=120)
    tr = Tracker()
    rate = None
    for s in snaps:
        X, _ = tr.update(s)
        rate = X[:, IDX["change_rate"]]
    changed = {e.object_id for s in snaps for e in s.events if e.type == "state_change"}
    assert changed and rate.max() > 0
    ids = [o.id for o in snaps[-1].objects]
    assert all(rate[ids.index(i)] == 0 for i in ids if i not in changed)


def test_mlp_train_save_load(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = np.clip(0.5 + 0.2 * X[:, 0] - 0.1 * X[:, 1] ** 2, 0, 1)
    m = train_mlp(X, y, seed=0, epochs=200)
    assert np.mean((m(X) - y) ** 2) < 0.02
    m.save(tmp_path / "m.npz")
    m2 = NumpyMLP.load(tmp_path / "m.npz")
    assert np.allclose(m(X), m2(X))


def test_scorer_matches_decision():
    dec = ROOT / "bench/results/importance_decision.json"
    if not dec.exists():
        pytest.skip("run python -m bench.importance_report")
    keep = json.loads(dec.read_text())["keep_mlp"]
    assert MODEL_PATH.exists() == keep
    assert (load_scorer() is rule_score) == (not keep)
