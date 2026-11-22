"""K3 ablation table: python -m bench.ablation [--config k3_ablation] [--seed 0] [--n-seeds N] [--out DIR].
Writes k3_ablation_{raw,summary,paired}.csv (runner) and k3_ablation_table.csv / k3_ablation_tests.csv:
per lighting condition (and pooled), per rung: precision/recall/F1 with 95% CIs over seeds, and paired tests between rungs."""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np
from . import runner, stats
from .k3 import prf
from adapters.synthetic import TRAIN_LIGHTS, HELDOUT_LIGHTS

COMPARE = [("sh2", "albedo"), ("sh2_shadow", "sh2"), ("sh2_cat16", "sh2"), ("sh2_full", "sh2_shadow"),
           ("sh2_full", "sh2_cat16"), ("sh2_full", "albedo")]
METRICS = ("precision", "recall", "f1")


def holm(ps):
    ps = np.array([1.0 if np.isnan(p) else p for p in ps])
    order = np.argsort(ps)
    adj, run = np.zeros(len(ps)), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        adj[i] = min(run, 1.0)
    return adj


def series(rows):
    """-> {(light, rung, metric): np.array aligned by seed} plus pooled 'ALL' (counts summed over lights per seed)."""
    seeds = sorted({r["seed"] for r in rows})
    d = {}
    for r in rows:
        light, rung = r["arm"].split("|")
        d.setdefault((light, rung, r["metric"]), {})[r["seed"]] = r["value"]
    out = {k: np.array([v[s] for s in seeds]) for k, v in d.items()}
    lights = sorted({k[0] for k in out})
    rungs = sorted({k[1] for k in out})
    for rung in rungs:
        cnt = {m: sum(out[(l, rung, m)] for l in lights) for m in ("tp", "fp", "fn")}
        pr = [prf(cnt["tp"][i], cnt["fp"][i], cnt["fn"][i]) for i in range(len(seeds))]
        for m in METRICS:
            out[("ALL", rung, m)] = np.array([p[m] for p in pr])
    return out, lights, rungs, seeds


def build_tables(rows):
    ser, lights, rungs, seeds = series(rows)
    order = [r for r in ("albedo", "sh2", "sh2_shadow", "sh2_cat16", "sh2_full") if r in rungs]
    split = lambda l: "train" if l in TRAIN_LIGHTS else ("heldout" if l in HELDOUT_LIGHTS else "all")
    table = []
    for l in lights + ["ALL"]:
        for r in order:
            row = dict(light=l, split=split(l), rung=r, n_seeds=len(seeds))
            for m in METRICS:
                s = stats.summarize(ser[(l, r, m)])
                row.update({m: s["mean"], f"{m}_lo": s["ci_lo"], f"{m}_hi": s["ci_hi"]})
            table.append(row)
    tests = []
    for l in lights + ["ALL"]:
        for m in METRICS:
            part = []
            for a, b in COMPARE:
                if a in order and b in order:
                    part.append(dict(light=l, metric=m, rung=a, vs=b, **stats.paired(ser[(l, a, m)], ser[(l, b, m)])))
            adj = holm([p["t_p"] for p in part])
            for p, q in zip(part, adj):
                p["t_p_holm"] = float(q)
            tests += part
    return table, tests


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m bench.ablation")
    ap.add_argument("--config", default="k3_ablation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-seeds", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    cfg = runner.load_config(a.config)
    res = runner.run(cfg, a.seed, a.n_seeds, a.out)
    table, tests = build_tables(res["rows"])
    out = Path(res["out"])
    write_csv(out / "k3_ablation_table.csv", table)
    write_csv(out / "k3_ablation_tests.csv", tests)
    for r in table:
        if r["light"] == "ALL":
            print(f"{r['rung']:>11} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} [{r['f1_lo']:.3f},{r['f1_hi']:.3f}]")
    print("->", out)
    return 0


if __name__ == "__main__":
    main()
