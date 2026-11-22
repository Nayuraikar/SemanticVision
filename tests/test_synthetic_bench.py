import json, numpy as np
from adapters.synthetic import SynthConfig, generate
from core.observers import train_observers, heldout_observers
from bench import registry, runner, stats
from bench.replay import record_synthetic, replay
from protocol import Snapshot

CFG = SynthConfig(n_objects=8, n_frames=60, lighting="cycle")


def dump(seed, cfg=CFG):
    return [s.model_dump_json(by_alias=True) for s in generate(cfg, seed)]


def test_same_seed_identical():
    assert dump(3) == dump(3)


def test_diff_seed_differs():
    assert dump(3) != dump(4)


def test_valid_and_events():
    snaps = list(generate(SynthConfig(n_objects=12, n_frames=300, lighting="cycle"), 1))
    assert all(isinstance(s, Snapshot) for s in snaps)
    types = {e.type for s in snaps for e in s.events}
    assert {"state_change", "lighting_change"} <= types
    assert all(abs(s.t - i * (1 / 30)) < 1e-6 for i, s in enumerate(snaps))


def test_record_replay_roundtrip(tmp_path):
    p = tmp_path / "s.ndjson"
    n = record_synthetic(CFG, 7, p)
    got = [s.model_dump_json(by_alias=True) for s in replay(p)]
    assert n == 60 and got == dump(7)


def test_observers_disjoint():
    for seed in range(5):
        tr, ho = train_observers(30, seed), heldout_observers(30, seed)
        assert not {o.key() for o in tr} & {o.key() for o in ho}
        assert not {o.id for o in tr} & {o.id for o in ho}
        assert {o.model for o in tr} == {"machado"} and "machado" not in {o.model for o in ho}
        assert max(o.severity for o in ho) < min(o.severity for o in tr)
        assert all(o.model != "vienot" or o.kind != "tritan" for o in ho)
        assert any(o.k != 1.0 for o in ho) and all(o.k == 1.0 for o in tr)
    assert train_observers(5, 1) == train_observers(5, 1)


def test_stats():
    a = np.array([1.0, 2, 3, 4, 5, 6]); b = a - 0.5 + np.array([0, .1, -.1, .05, 0, .02])
    p = stats.paired(a, b)
    assert p["mean_diff"] > 0.4 and p["t_p"] < 0.01
    s = stats.summarize(a)
    assert s["ci_lo"] < s["mean"] < s["ci_hi"]


def test_cli_end_to_end(tmp_path):
    from bench.__main__ import main
    assert "observer_gap" in registry.names()
    cfgp = tmp_path / "c.json"
    cfgp.write_text(json.dumps({"experiment": "observer_gap", "n_seeds": 4, "baseline": "train",
                                "params": {"scene": {"n_objects": 6, "n_frames": 20}, "stride": 10, "n_observers": 3}}))
    assert main(["run", "--config", str(cfgp), "--seed", "5", "--out", str(tmp_path)]) == 0
    for k in ("raw", "summary", "paired"):
        assert (tmp_path / f"c_{k}.csv").exists()
    r1 = runner.run(runner.load_config(cfgp), 5, out_dir=tmp_path)
    r2 = runner.run(runner.load_config(cfgp), 5, out_dir=tmp_path)
    assert r1["rows"] == r2["rows"] and r1["seeds"] == [5, 6, 7, 8]
