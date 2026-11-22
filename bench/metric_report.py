"""Disagreement report: python -m bench.metric_report [--config metric_disagreement] [--seed 0] [--n-seeds N] [--out DIR].
Writes metric_disagreement_{raw,summary,paired}.csv (runner), metric_disagreement_pairs.csv (every disagreeing pair) and
metric_disagreement_bins.csv (kind x lighting x size bin x luminance bin: counts and rates)."""
from __future__ import annotations
import argparse, csv
from collections import defaultdict
from pathlib import Path
from . import runner
from .experiments.metric_disagreement import collect


def bins_table(rows):
    g = defaultdict(list)
    for r in rows:
        for light in (r["light"], "ALL"):
            g[(r["kind"], light, r["size_bin"], r["lum_bin"])].append(r)
    out = []
    for (kind, light, sb, lb), v in sorted(g.items()):
        n = len(v)
        out.append(dict(kind=kind, light=light, size_bin_deg=sb, lum_bin_Y=lb, n=n,
                        n_disagree=sum(x["disagree"] for x in v), disagree_rate=sum(x["disagree"] for x in v) / n,
                        cam16_only=sum(x["cam16_only"] for x in v), de00_only=sum(x["de00_only"] for x in v),
                        conf_rate_de00=sum(x["conf_de00"] for x in v) / n, conf_rate_cam16=sum(x["conf_cam16"] for x in v) / n))
    return out


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m bench.metric_report")
    ap.add_argument("--config", default="metric_disagreement")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-seeds", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    cfg = runner.load_config(a.config)
    res = runner.run(cfg, a.seed, a.n_seeds, a.out)
    rows = [r for s in res["seeds"] for r in collect(cfg["params"], s)]
    out = Path(res["out"])
    write_csv(out / "metric_disagreement_pairs.csv", [r for r in rows if r["disagree"]])
    bt = bins_table(rows)
    write_csv(out / "metric_disagreement_bins.csv", bt)
    for r in bt:
        if r["light"] == "ALL" and r["kind"] == "emissive":
            print(f"{r['size_bin_deg']:>8} deg Y {r['lum_bin_Y']:>9}: n={r['n']:4d} disagree={r['disagree_rate']:.2f} cam16_only={r['cam16_only']} de00_only={r['de00_only']}")
    print("->", out)
    return 0


if __name__ == "__main__":
    main()
