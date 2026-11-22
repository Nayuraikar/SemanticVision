import csv
import numpy as np
from bench import registry, runner
from bench.ablation import build_tables, holm
from bench.k3 import cam16_tau_scale, prf, world_seed, HELD_SEED_BASE
from bench.experiments.metric_disagreement import collect
from bench.metric_report import bins_table
from adapters.synthetic import TRAIN_LIGHTS, HELDOUT_LIGHTS
from core.perception.threshold import ThresholdParams

SMALL = {"lights": ["night", "tinted"], "n_worlds": 2, "n_obs": 3, "world": {"n_objects": 12, "palette": "mixed"}}


def test_prf_edge_cases():
    assert prf(0, 0, 0) == dict(precision=1.0, recall=1.0, f1=1.0)
    assert prf(0, 2, 0)["precision"] == 0.0 and prf(0, 0, 2)["recall"] == 0.0
    assert abs(prf(6, 2, 2)["f1"] - 0.75) < 1e-12


def test_ablation_experiment_ladder_and_determinism():
    fn = registry.get("k3_ablation")
    a, b = fn(SMALL, 0), fn(SMALL, 0)
    assert a == b
    assert {k.split("|")[1] for k in a} == {"albedo", "sh2", "sh2_shadow", "sh2_cat16", "sh2_full"}
    f1 = lambda l, r: np.mean([fn(SMALL, s)[f"{l}|{r}"]["f1"] for s in range(3)])
    for l in ("night", "tinted"):
        assert f1(l, "sh2_full") > f1(l, "albedo")       # lighting-aware beats albedo-only under dark/tinted light


def test_runner_csv_and_table(tmp_path):
    cfg = {"name": "k3_small", "experiment": "k3_ablation", "n_seeds": 4, "baseline": "night|albedo", "params": SMALL}
    res = runner.run(cfg, 0, out_dir=tmp_path)
    for suf in ("raw", "summary", "paired"):
        assert (tmp_path / f"k3_small_{suf}.csv").exists()
    table, tests = build_tables(res["rows"])
    assert {r["light"] for r in table} == {"night", "tinted", "ALL"}
    r0 = table[0]
    assert r0["f1_lo"] <= r0["f1"] <= r0["f1_hi"] and r0["n_seeds"] == 4
    t = [x for x in tests if x["light"] == "ALL" and x["metric"] == "f1" and x["rung"] == "sh2_full" and x["vs"] == "albedo"][0]
    assert t["mean_diff"] > 0 and 0 <= t["t_p_holm"] <= 1 and "wilcoxon_p" in t
    assert holm([0.01, 0.04, 0.03])[0] == 0.03 and max(holm([0.5, 0.9])) <= 1.0


def test_heldout_scenes_lights_and_no_tuned_thresholds():
    assert not set(TRAIN_LIGHTS) & set(HELDOUT_LIGHTS)
    assert world_seed(3, 1, "heldout") >= HELD_SEED_BASE > world_seed(3, 1, "train")
    assert world_seed(0, 0, "heldout") != world_seed(0, 0, "train")
    import inspect, bench.k3 as k3, bench.experiments.k3_ablation as ab
    assert "ThresholdParams" not in inspect.getsource(ab) and "params=" not in inspect.getsource(k3.confusable_pairs).split("build_graph")[1].split(")")[0]
    assert ThresholdParams() == ThresholdParams()        # defaults only; nothing fitted on the evaluation worlds


def test_cam16_scale_deterministic_and_sane():
    s = cam16_tau_scale()
    assert s == cam16_tau_scale(0) and 0.2 < s < 5


def test_disagreement_report():
    p = {"lights": ["day", "night"], "n_worlds": 2, "n_obs": 3, "world": {"n_objects": 14, "palette": "mixed"}}
    rows = collect(p, 0)
    assert rows and {"emissive", "reflective"} <= {r["kind"] for r in rows}
    for r in rows:
        assert r["disagree"] == (r["conf_de00"] != r["conf_cam16"]) and r["disagree"] == (r["cam16_only"] or r["de00_only"])
    bt = bins_table(rows)
    assert all(b["n"] >= b["n_disagree"] for b in bt) and sum(b["n"] for b in bt if b["light"] == "ALL") == len(rows)
    em = [r for r in rows if r["kind"] == "emissive"]
    assert em and min(r["size_deg"] for r in em) < 2.0           # small-field emissive cases present
    assert registry.get("metric_disagreement")(p, 0)["emissive"]["n_pairs"] > 0
