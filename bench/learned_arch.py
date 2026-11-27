"""Encoder comparison (SPEC 5.6): python -m bench.learned_arch [--scale small] [--epochs 30] [--seed 0]
Trains MLP (per-object, no set context), DeepSets and a small GNN (adjacency message passing) with the same config/seed on
the same data, then scores the raw model output with core under the TRUE observer on held-out scenes (test_scene, test_both)
plus val. Paired tests (by sample) vs DeepSets, Holm-corrected. The GNN is kept only if it beats DeepSets on held-out scenes
(objective mean diff > 0 with Holm p < .05); the decision is written to learned_arch_summary.json.
Writes bench/results/learned_arch_{table,tests}.csv. Kept under 5 min by the epoch count."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from core.solver import SolverConfig
from core.solver.problem import Problem
from learning import data as D
from learning.model import NetConfig
from learning.train import TrainConfig, train, collate, model_args
from . import stats
from .ablation import holm, write_csv
from .learned_report import palette_of, score

OUT = Path(__file__).resolve().parent / "results"
ARCHS = ("mlp", "deepsets", "gnn")
EVAL = ("val", "test_scene", "test_both")
HELD_SCENES = ("test_scene", "test_both")


@torch.no_grad()
def predict(net, samples, bs=64):
    out = []
    for i in range(0, len(samples), bs):
        part = samples[i:i + bs]
        lab = net(*model_args(collate(part, with_loss=False)))[0].numpy()
        out += [lab[k, :len(s["inp"]["x"])] for k, s in enumerate(part)]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m bench.learned_arch")
    ap.add_argument("--scale", default="small")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    t0 = time.time()
    torch.set_num_threads(4)
    data = D.build(a.scale)
    cfg = SolverConfig(**D.SOLVER)
    res = {}
    for arch in ARCHS:
        tc = TrainConfig(epochs=a.epochs, seed=a.seed, net=NetConfig(arch=arch))
        net, info = train(data, tc, "cpu", verbose=False)
        net.eval()
        for split in EVAL:
            for s, lab in zip(data[split], predict(net, data[split])):
                pb = Problem(s["scene"], [s["obs"]], cfg)
                r = score(pb, s["scene"], s["obs"], palette_of(s["scene"], lab))
                res[(arch, split, s["i"])] = r
        print(f"{arch}: trained {info['seconds']:.0f}s (best epoch {info['best_epoch']}), {time.time() - t0:.0f}s total", flush=True)
    table, tests = [], []
    for split in EVAL + ("held_scenes",):
        sp = HELD_SCENES if split == "held_scenes" else (split,)
        keys = [(s2, s["i"]) for s2 in sp for s in data[s2]]
        for arch in ARCHS:
            row = dict(split=split, arch=arch, n=len(keys))
            for m in ("obj", "succ", "min_de", "screen_fail", "dev"):
                st_ = stats.summarize([res[(arch, k[0], k[1])][m] for k in keys])
                row.update({m: st_["mean"], f"{m}_lo": st_["ci_lo"], f"{m}_hi": st_["ci_hi"]})
            table.append(row)
            if arch != "deepsets":
                for m in ("obj", "succ"):
                    x = [res[(arch, k[0], k[1])][m] for k in keys]
                    y = [res[("deepsets", k[0], k[1])][m] for k in keys]
                    p = stats.paired(x, y)
                    s_ = stats.summarize(np.subtract(x, y))
                    tests.append(dict(split=split, a=arch, b="deepsets", metric=m, n=p["n"], mean_diff=p["mean_diff"],
                                      ci_lo=s_["ci_lo"], ci_hi=s_["ci_hi"], t_p=p["t_p"], wilcoxon_p=p["wilcoxon_p"], dz=p["dz"]))
    for t, h in zip(tests, holm([t["t_p"] for t in tests])):
        t["holm_p"] = h
    g = next(t for t in tests if t["split"] == "held_scenes" and t["a"] == "gnn" and t["metric"] == "obj")
    keep_gnn = bool(g["mean_diff"] > 0 and g["holm_p"] < 0.05)
    out = Path(a.out); out.mkdir(exist_ok=True)
    write_csv(out / "learned_arch_table.csv", table); write_csv(out / "learned_arch_tests.csv", tests)
    summ = dict(epochs=a.epochs, seed=a.seed, keep_gnn=keep_gnn, gnn_vs_deepsets_held_scenes=g, seconds=time.time() - t0,
                decision="GNN beats DeepSets on held-out scenes: report it" if keep_gnn else
                "GNN does not beat DeepSets on held-out scenes: DeepSets stays the deployed encoder; GNN not reported further")
    (out / "learned_arch_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    for r in table:
        print(f"{r['split']:11s} {r['arch']:8s} obj {r['obj']:.3f} [{r['obj_lo']:.3f},{r['obj_hi']:.3f}] succ {r['succ']:.3f}")
    for t in tests:
        print(f"{t['split']:11s} {t['a']} - deepsets {t['metric']}: {t['mean_diff']:+.3f} [{t['ci_lo']:+.3f},{t['ci_hi']:+.3f}] holm p {t['holm_p']:.3g}")
    print(summ["decision"], f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
