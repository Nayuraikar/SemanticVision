"""Transfer bench: censoring is explicit, test choice, Holm, worker cap, and the pipeline wrapper on a fixture."""
import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from bench.experiments import transfer as X
from protocol.io import read_ndjson
from protocol.models import Snapshot

FIX = Path("tests/fixtures")


def test_config_caps_workers_and_uses_all_conditions():
    cfg = json.loads(Path("bench/configs/transfer.json").read_text())
    assert cfg["params"]["workers"] <= X.MAX_WORKERS == 4
    assert {k: tuple(v) for k, v in cfg["params"]["envs"].items()} == X.ENVS


def test_observer_draw_is_environment_independent_and_split():
    tr, held = X.observers_for(3, {})
    assert (tr, held) == X.observers_for(3, {})
    assert {o.split for o in tr} == {"train"} and {o.split for o in held} == {"heldout"}
    assert len({o.kind for o in tr + held}) == 1


def test_gap_paired_when_matched_welch_otherwise():
    g = X.gap({0: 3.0, 1: 5.0, 2: 4.5}, {0: 1.0, 1: 2.0, 2: 2.0})
    assert g["test"] == "paired" and g["n_matched"] == 3 and g["gap"] == pytest.approx(2.5) and g["ci_lo"] < 2.5 < g["ci_hi"]
    assert math.isnan(g["welch_p"])
    w = X.gap({0: 3.0, 1: 5.0}, {2: 1.0, 3: 2.0})
    assert w["test"] == "welch" and w["n_matched"] == 0 and w["gap"] == pytest.approx(2.5) and 0 < w["p"] < 1
    empty = X.gap({}, {0: 1.0, 1: 2.0})
    assert math.isnan(empty["gap"]) and math.isnan(empty["p"])          # nothing to compare is not a zero gap


def test_report_censors_instead_of_zero(tmp_path):
    rows = []
    for s in range(3):
        for env, cs in X.ENVS.items():
            for c in cs:
                crit = not (env == "dashboard" and s == 0) and not (c == "night")
                m = dict(no_critical=float(not crit), flicker_excess=1.0 + s + len(c), ms_total=10.0 + s)
                if crit:
                    m["min_de00"] = 4.0 + s + (2.0 if env == "dashboard" else 0.0) + 0.1 * s * (env == "dashboard")
                rows += [dict(seed=s, arm="pipeline", metric=f"{env}_{c}|{k}", value=v) for k, v in m.items()]
    raw = tmp_path / "transfer_raw.csv"
    with open(raw, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    table, tests, censor = X.report(raw)
    T = {(r["group"], r["metric"]): r for r in table}
    assert T[("dashboard", "min_de00")]["n"] == 2 and T[("dashboard", "min_de00")]["censored_seeds"] == 1
    assert T[("dashboard", "min_de00")]["mean"] == pytest.approx(7.65)       # censored seed not averaged in as 0
    assert T[("traffic_night", "min_de00")]["n"] == 0 and math.isnan(T[("traffic_night", "min_de00")]["mean"])
    C = {c["group"]: c for c in censor}
    assert C["dashboard"]["seeds_without_critical"] == 1 and C["traffic"]["seeds_without_critical"] == 0
    assert C["traffic"]["condition_seeds_without_critical"] == 3
    assert C["dashboard"]["condition_seeds_without_tier2_calls"] == 9
    g = next(t for t in tests if t["comparison"] == "dashboard - traffic" and t["metric"] == "min_de00")
    assert g["test"] == "paired" and g["n_matched"] == 2 and g["n_traffic"] == 3 and g["p_holm"] >= g["p"]
    assert (tmp_path / "transfer_tests.csv").exists() and (tmp_path / "transfer_censoring.csv").exists()


def test_pipeline_wrapper_uses_onnx_tier_and_stage_latencies_add_up():
    tr, held = X.observers_for(1, {"n_heldout": 2})
    snaps = [s.model_copy(update={"frame": i}) for i, s in enumerate(read_ndjson(FIX / "dashboard_normal.ndjson", Snapshot))]
    table = {o.id: o for o in tr} | {held[0].id: held[0]}
    res = X.run_pipeline(snaps, 0.1, held[0], table)
    assert len(res["frames"]) == len(res["shown"]) == 4 and res["solves"] >= 1 and res["infeasible"]
    for f in res["frames"]:
        assert set(f) == set(X.STAGES)
        assert sum(f[s] for s in X.STAGES if s != "total") == pytest.approx(f["total"])
        assert all(np.isfinite(v) and v > -0.5 for v in f.values())
    assert res["tier2_fallbacks"] <= res["tier2_calls"]


def test_task_pairs_are_lit_different_state_indicators_regardless_of_adjacency():
    for env, name in (("traffic", "traffic_day"), ("dashboard", "dashboard_normal")):
        s = next(iter(read_ndjson(FIX / f"{name}.ndjson", Snapshot)))
        by = {o.id: o for o in s.objects}
        pairs = X.task_pairs(env, s)
        assert pairs
        for (a, ra), (b, rb) in pairs:
            assert ra != rb and max(by[a].emissive_rgb) > 0 and max(by[b].emissive_rgb) > 0
            assert (ra in ("brake", "hazard")) == (rb in ("brake", "hazard"))
        lamps = {(i, r) for i, r in ((o.id, X.task_role(env, o)) for o in s.objects) if r}
        n = sum(a[1][0] == b[1][0] and a[1][1] != b[1][1] for a in lamps for b in lamps) // 2
        assert len(pairs) == n


@pytest.mark.parametrize("env,name", [("traffic", "traffic_day"), ("dashboard", "dashboard_normal")])
def test_score_runs_end_to_end_with_secondary_task_metrics(env, name):
    tr, held = X.observers_for(1, {"n_heldout": 2})
    snaps = [s.model_copy(update={"frame": i}) for i, s in enumerate(read_ndjson(FIX / f"{name}.ndjson", Snapshot))]
    m = X.score(snaps, 1, tr, held, 0.1, env)
    assert m["task_n_pairs"] > 0 and {"solve_analysis_ms", "solve_correction_ms", *X.TASK} <= set(m)
    for k in ("task_success", "task_uncorrected_success", "task_unresolved"):
        assert 0 <= m[k] <= 1
    assert m["task_min_de00"] >= 0 and m["task_uncorrected_min_de00"] >= 0


def test_escalation_window_anchor():
    snaps = list(read_ndjson(FIX / "dashboard_dim.ndjson", Snapshot))
    cut, k0 = X.escalation_window(snaps, 1, 2)
    assert [s.frame for s in cut] == list(range(len(cut)))
    assert any(e.object_id == "master_warning" for e in cut[k0].events)
