"""python -m bench.allocator_report [--skip-run]: run the allocator benches (each < 5 min) and print/write paired tests of the MCKP
solvers vs greedy / uniform / round-robin at EQUAL budget, per workload cell and on the real solver; overhead table; bandit; predictor.
Writes bench/results/allocator_tests.csv."""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from . import runner, stats

RES = runner.ROOT / "results"
BENCHES = ("allocator_scaling", "allocator_bandit", "allocator_quality", "allocator_quality_50", "allocator_predictor")


def holm(ps):
    ps = np.asarray(ps, float)
    o = np.argsort(ps); adj = np.empty_like(ps); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(ps) - r) * ps[i]); adj[i] = min(run, 1.0)
    return adj


def tests(raw, metric, a_methods, b_methods, key_fn):
    rows = []
    piv = raw[raw.metric == metric].pivot_table(index="seed", columns="arm", values="value")
    cells = sorted({key_fn(c) for c in piv.columns if "|" in c})
    for cell in cells:
        for a in a_methods:
            for b in b_methods:
                ca, cb = f"{a}|{cell}" if cell else a, f"{b}|{cell}" if cell else b
                if ca in piv and cb in piv:
                    rows.append(dict(cell=cell, a=a, b=b, metric=metric, mean_a=piv[ca].mean(), mean_b=piv[cb].mean(),
                                     **stats.paired(piv[ca].values, piv[cb].values)))
    df = pd.DataFrame(rows)
    if len(df):
        df["holm_p"] = holm(df.t_p.fillna(1.0).values)
    return df


def main(skip_run=False):
    if not skip_run:
        for b in BENCHES:
            runner.run(runner.load_config(b), seed=0)
    out = []
    sc = pd.read_csv(RES / "allocator_scaling_raw.csv")
    out.append(tests(sc, "benefit", ("lagrangian", "dp"), ("greedy", "uniform", "round_robin"), lambda c: c.split("|", 1)[1]))
    ql = []
    for name in ("allocator_quality", "allocator_quality_50"):
        q = pd.read_csv(RES / f"{name}_raw.csv")
        t = tests(q, "wsuccess", ("lagrangian", "dp"), ("greedy", "uniform", "round_robin"), lambda c: c.split("|", 1)[1])
        t["cell"] = name + ":" + t["cell"]
        ql.append(t)
    out += ql
    df = pd.concat(out)
    df.to_csv(RES / "allocator_tests.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n== MCKP vs baselines, equal budget (paired over seeds; benefit = importance-weighted gain / weighted success) ==")
    cols = ["cell", "a", "b", "mean_a", "mean_b", "mean_diff", "t_p", "holm_p"]
    print(df[cols].round(4).to_string(index=False))
    s = pd.read_csv(RES / "allocator_scaling_summary.csv")
    ov = s[s.metric.isin(["overhead_ms", "overhead_frac"])].copy()
    ov[["method", "n", "budget"]] = ov.arm.str.split("|", expand=True)
    print("\n== allocator overhead (mean over seeds) ==")
    print(ov.pivot_table(index=["n", "budget"], columns=["metric", "method"], values="mean").round(4).to_string())
    return df


if __name__ == "__main__":
    main("--skip-run" in sys.argv)
