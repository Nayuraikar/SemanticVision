import argparse, sys
from . import registry, runner


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m bench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True)
    r.add_argument("--seed", type=int, default=0, help="base seed; seeds = seed..seed+n_seeds-1")
    r.add_argument("--n-seeds", type=int, default=None)
    r.add_argument("--out", default=None)
    sub.add_parser("list")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        print("\n".join(registry.names()))
        return 0
    cfg = runner.load_config(a.config)
    res = runner.run(cfg, a.seed, a.n_seeds, a.out)
    for s in res["summary"]:
        print(f"{s['arm']:>14} {s['metric']:<22} {s['mean']:.4f} [{s['ci_lo']:.4f}, {s['ci_hi']:.4f}] n={s['n']}")
    for p in res["paired"]:
        print(f"{p['arm']} vs {p['baseline']} {p['metric']}: diff={p['mean_diff']:.4f} t_p={p['t_p']:.3g} w_p={p['wilcoxon_p']:.3g}")
    print("->", res["out"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
