"""K2 profile-error report: python -m bench.robust_report [--config profile_error] [--seed 0] [--n-seeds N] [--out DIR].
Writes profile_error_{raw,summary}.csv (runner) and profile_error_table.csv (per level x method: mean, 95% CI) and
profile_error_tests.csv (per level: cvar/worst vs point, paired over seeds, mean diff + CI, t/Wilcoxon p, Holm)."""
from __future__ import annotations
import argparse
import numpy as np
from . import runner, stats
from .ablation import holm, write_csv

METRICS = ("min_de00", "success_rate", "worst_deg", "deviation")


def tables(rows):
    seeds = sorted({r["seed"] for r in rows})
    d = {}
    for r in rows:
        d.setdefault((r["arm"], r["metric"]), {})[r["seed"]] = r["value"]
    ser = {k: np.array([v[s] for s in seeds if s in v]) for k, v in d.items()}
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    table = []
    for a in arms:
        lv, me = a.split("|")
        row = dict(level=lv, method=me, n_seeds=len(seeds))
        for m in METRICS:
            s = stats.summarize(ser[(a, m)])
            row.update({m: s["mean"], f"{m}_lo": s["ci_lo"], f"{m}_hi": s["ci_hi"]})
        table.append(row)
    tests = []
    for lv in dict.fromkeys(a.split("|")[0] for a in arms if a != "ref|none"):
        part = []
        for me in ("cvar", "worst"):
            for m in METRICS:
                a, b = ser[(f"{lv}|{me}", m)], ser[(f"{lv}|point", m)]
                p = stats.paired(a, b)
                dd = a - b
                h = float(stats.st.t.ppf(0.975, len(dd) - 1) * dd.std(ddof=1) / np.sqrt(len(dd))) if len(dd) > 1 else float("nan")
                part.append(dict(level=lv, method=me, metric=m, **p, diff_lo=p["mean_diff"] - h, diff_hi=p["mean_diff"] + h))
        adj = holm([p["t_p"] for p in part])
        for p, q in zip(part, adj):
            p["t_p_holm"] = float(q)
        tests += part
    return table, tests


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m bench.robust_report")
    ap.add_argument("--config", default="profile_error")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-seeds", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = runner.run(runner.load_config(a.config), a.seed, a.n_seeds, a.out)
    table, tests = tables(res["rows"])
    write_csv(f"{res['out']}/profile_error_table.csv", table)
    write_csv(f"{res['out']}/profile_error_tests.csv", tests)
    for t in table:
        print(f"{t['level']:>15} {t['method']:<6} min_de00 {t['min_de00']:.2f} [{t['min_de00_lo']:.2f},{t['min_de00_hi']:.2f}] "
              f"succ {t['success_rate']:.2f} worst_deg {t['worst_deg']:.2f} dev {t['deviation']:.2f}")
    for t in tests:
        if t["metric"] == "min_de00":
            print(f"{t['level']:>15} {t['method']:<6} vs point min_de00 diff {t['mean_diff']:+.2f} [{t['diff_lo']:+.2f},{t['diff_hi']:+.2f}] "
                  f"t_p={t['t_p']:.3g} holm={t['t_p_holm']:.3g}")
    print("->", res["out"])


if __name__ == "__main__":
    main()
