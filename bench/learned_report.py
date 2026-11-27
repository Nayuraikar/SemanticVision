"""Learned-solver report (SPEC 5.6): python -m bench.learned_report [--scale small] [--model core/allocator/learned.onnx]
Seeded: the dataset is learning.data.build(scale) (exact targets; DE reference on a subset), the model is the trained ONNX file.
Every score is computed with core (numpy) under the sample's TRUE observer; test_obs/test_both observers (Brettel/Vienot,
severity [.1,.5), threshold noise) and test_scene/test_both scenes (held-out lighting, larger N) were never used for training
or tuning. Metrics per sample: objective (core Problem, single true observer), relative optimality gap
(obj_exact - obj_model)/(obj_exact - obj_none), success (palette pairs with D >= tau), screen-space failing pairs, deviation,
constraint violations of the raw ONNX output, latency (features + ORT), and the deployed tier (verification + exact fallback)
with its fallback rate. Writes bench/results/learned_{rows,table,tests,gap}.csv + learned_summary.json."""
from __future__ import annotations
import argparse, json, resource, sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np
from scipy import stats as st
from core.allocator.onnx_tier import OnnxTier, DEFAULT_MODEL
from core.learned import embed_observer, scene_inputs
from core.solver import SolverConfig, screen_check
from core.solver.problem import Problem
from learning import data as D
from . import stats
from .ablation import holm, write_csv

OUT = Path(__file__).resolve().parent / "results"
SPLITS = ("train", "val", "test_obs", "test_scene", "test_both")
N_TRAIN_EVAL = 80          # seen training samples, for the train-vs-held-out generalisation gap


def palette_of(scene, lab_obj):
    first = np.unique(scene.groups, return_index=True)[1]
    return np.asarray(lab_obj, float)[first]


def violations(pb, X):
    """Raw (pre core-projection) palette X -> number of MOVED entries violating deviation/gamut/L constraints (an original
    colour already outside the bounds and left unchanged is not counted)."""
    r = pb._rgb(X)
    bad = (pb.deviation(X) > pb.delta + 0.05) | ((r < -1e-3) | (r > 1 + 1e-3)).any(-1) \
        | (X[:, 0] < pb.cfg.L_range[0] - 1e-3) | (X[:, 0] > pb.cfg.L_range[1] + 1e-3)
    return int((bad & (np.abs(X - pb.lab0).max(-1) > 1e-3)).sum())


def score(pb, scene, obs, P):
    X = pb.project(np.asarray(P, float)[None])
    Dm = pb.distance_tensor(X)[0, 0]
    tau = pb.tau[0]
    bad, _ = screen_check(scene, pb.to_rgb(X[0])[scene.groups], [obs])
    return dict(obj=float(pb.objective(X)[0]), min_de=float(Dm.min()) if len(Dm) else np.nan,
                succ=float((Dm >= tau).mean()) if len(Dm) else 1.0, screen_fail=len(bad),
                dev=float(pb.deviation(X[0]).mean()))


def proxy_observer(obs):
    """What deployment can simulate for verification: a Machado observer with the same embedding (kind, severity, k, s)."""
    return obs if obs.model == "machado" else replace(obs, id=obs.id + "~m", model="machado", split="train")


def evaluate(data, model_path, splits=SPLITS, verbose=True):
    tier, tier0 = OnnxTier(model_path), OnnxTier(model_path, repair_iters=0)     # with / without local repair
    cfg = SolverConfig(**D.SOLVER)
    rows = []
    for split in splits:
        samples = data[split][:N_TRAIN_EVAL] if split == "train" else data[split]
        for s in samples:
            scene, obs = s["scene"], s["obs"]
            pb = Problem(scene, [obs], cfg)
            t0 = time.perf_counter()
            inp = scene_inputs(scene, embed_observer(obs), **D.SOLVER)
            lab = tier.infer(inp)
            ms = (time.perf_counter() - t0) * 1e3
            Pm = palette_of(scene, lab)
            r = dict(split=split, i=s["i"], seed=s["seed"], n=s["n"], E=pb.E, lighting=s["lighting"], palette=s["palette"],
                     obs_model=obs.model, kind=obs.kind, severity=obs.severity, infer_ms=ms, exact_ms=s["exact_ms"],
                     viol=violations(pb, Pm))
            arms = dict(model=Pm, exact=palette_of(scene, s["target"]), none=pb.lab0)
            if s["de"] is not None:
                arms["de"] = palette_of(scene, s["de"]); r["de_ms"] = s["de_ms"]
            for a, P in arms.items():
                for k, v in score(pb, scene, obs, P).items():
                    r[f"{a}_{k}"] = v
            # deployed tier: verification on the proxy observer, [local repair,] exact fallback; scored under the TRUE observer
            pd = Problem(scene, [proxy_observer(obs)], cfg)
            for a, t in (("tier", tier), ("tier0", tier0)):
                t0 = time.perf_counter()
                t.z = embed_observer(obs)
                Pd = t.run(pd, pd.lab0.copy(), np.flatnonzero(pd.active))
                r[f"{a}_ms"] = (time.perf_counter() - t0) * 1e3
                r[f"{a}_fallback"] = int(t.last.get("fallback", False)); r[f"{a}_reason"] = t.last.get("reason", "")
                for k, v in score(pb, scene, obs, Pd).items():
                    r[f"{a}_{k}"] = v
            for a in ("model", "tier", "tier0", "de"):
                if f"{a}_obj" in r:
                    den = r["exact_obj"] - r["none_obj"]
                    r[f"{a}_gap"] = r["exact_obj"] - r[f"{a}_obj"]
                    r[f"{a}_relgap"] = r[f"{a}_gap"] / den if abs(den) > 1e-6 else 0.0
            rows.append(r)
        if verbose:
            print(f"{split}: {len(samples)} samples", flush=True)
    tier.z = None
    return rows, tier


def table(rows):
    out = []
    metrics = ("relgap", "gap", "succ", "min_de", "screen_fail", "dev")
    for split in dict.fromkeys(r["split"] for r in rows):
        R = [r for r in rows if r["split"] == split]
        row = dict(split=split, n=len(R), tier_clean=split in ("test_obs", "test_both"))   # tier verifies/repairs on a proxy;
        # on train-family splits the proxy IS the scoring observer, so tier_* there is not a held-out score (model_* always is)
        den = sum(r["exact_obj"] - r["none_obj"] for r in R)
        for a in ("model", "tier", "tier0", "exact", "none"):
            for m in metrics:
                k = f"{a}_{m}"
                if k in R[0]:
                    s = stats.summarize([r[k] for r in R])
                    row.update({k: s["mean"], f"{k}_lo": s["ci_lo"], f"{k}_hi": s["ci_hi"]})
            if f"{a}_gap" in R[0]:          # ratio of sums: robust to near-zero per-sample denominators
                row[f"{a}_relgap_agg"] = sum(r[f"{a}_gap"] for r in R) / den
        for a in ("tier", "tier0"):
            row[f"{a}_fallback_rate"] = float(np.mean([r[f"{a}_fallback"] for r in R]))
            row[f"{a}_ms"] = float(np.median([r[f"{a}_ms"] for r in R]))
            row[f"{a}_ms_mean"] = float(np.mean([r[f"{a}_ms"] for r in R]))
        row["repaired_rate"] = float(np.mean([r["tier_reason"] == "repaired" for r in R]))
        RD = [r for r in R if "de_obj" in r]
        if RD:
            row.update(n_de=len(RD), de_relgap=float(np.mean([r["de_relgap"] for r in RD])),
                       model_relgap_on_de=float(np.mean([r["model_relgap"] for r in RD])),
                       model_minus_de_obj=float(np.mean([r["model_obj"] - r["de_obj"] for r in RD])),
                       exact_minus_de_obj=float(np.mean([r["exact_obj"] - r["de_obj"] for r in RD])))
        row.update(viol_per_scene=float(np.mean([r["viol"] for r in R])),
                   infer_ms=float(np.median([r["infer_ms"] for r in R])), infer_ms_p95=float(np.percentile([r["infer_ms"] for r in R], 95)),
                   exact_ms=float(np.median([r["exact_ms"] for r in R])))
        out.append(row)
    return out


def tests(rows):
    """Paired (by sample) within split: model vs exact, model vs none, tier vs model; Holm over all."""
    T = []
    for split in dict.fromkeys(r["split"] for r in rows):
        R = [r for r in rows if r["split"] == split]
        for a, b, m in (("model", "exact", "obj"), ("model", "none", "obj"), ("model", "exact", "succ"),
                        ("model", "none", "succ"), ("tier", "model", "obj"), ("tier", "exact", "obj"),
                        ("tier0", "exact", "obj"), ("tier", "exact", "succ")):
            p = stats.paired([r[f"{a}_{m}"] for r in R], [r[f"{b}_{m}"] for r in R])
            d = np.array([r[f"{a}_{m}"] - r[f"{b}_{m}"] for r in R])
            s = stats.summarize(d)
            T.append(dict(split=split, a=a, b=b, metric=m, n=p["n"], mean_diff=p["mean_diff"], ci_lo=s["ci_lo"], ci_hi=s["ci_hi"],
                          t_p=p["t_p"], wilcoxon_p=p["wilcoxon_p"], dz=p["dz"]))
        RD = [r for r in R if "de_obj" in r]
        if len(RD) >= 2:
            for a in ("model", "exact"):
                p = stats.paired([r[f"{a}_obj"] for r in RD], [r["de_obj"] for r in RD])
                s = stats.summarize([r[f"{a}_obj"] - r["de_obj"] for r in RD])
                T.append(dict(split=split, a=a, b="de", metric="obj", n=p["n"], mean_diff=p["mean_diff"], ci_lo=s["ci_lo"],
                              ci_hi=s["ci_hi"], t_p=p["t_p"], wilcoxon_p=p["wilcoxon_p"], dz=p["dz"]))
    for t, h in zip(T, holm([t["t_p"] for t in T])):
        t["holm_p"] = h
    return T


def gen_gap(rows, ref="val"):
    """Generalisation gap: per-sample model optimality gap (obj units)/success on each split vs the iid `val` split (Welch t), and train vs val."""
    out = []
    base = {m: np.array([r[f"model_{m}"] for r in rows if r["split"] == ref]) for m in ("gap", "succ")}
    for split in dict.fromkeys(r["split"] for r in rows):
        if split == ref:
            continue
        for m in ("gap", "succ"):
            x = np.array([r[f"model_{m}"] for r in rows if r["split"] == split])
            t = st.ttest_ind(x, base[m], equal_var=False)
            se = np.sqrt(x.var(ddof=1) / len(x) + base[m].var(ddof=1) / len(base[m]))
            dlt = float(x.mean() - base[m].mean())
            out.append(dict(split=split, ref=ref, metric=f"model_{m}", mean=float(x.mean()), ref_mean=float(base[m].mean()),
                            diff=dlt, ci_lo=dlt - 1.96 * se, ci_hi=dlt + 1.96 * se, welch_p=float(t.pvalue)))
    return out


def footprint(model_path):
    import onnxruntime as ort
    r0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    OnnxTier(model_path)
    r1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1 if sys.platform == "darwin" else 1024                # ru_maxrss: bytes on macOS, KiB on Linux
    meta = Path(model_path).with_suffix(".metrics.json")
    n_params = json.loads(meta.read_text()).get("n_params") if meta.exists() else None
    return dict(onnx_bytes=Path(model_path).stat().st_size, n_params=n_params,
                session_peak_rss_delta_bytes=int((r1 - r0) * scale), ort=ort.__version__)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m bench.learned_report")
    ap.add_argument("--scale", default="small")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    t0 = time.time()
    data = D.build(a.scale)
    fp = footprint(a.model)
    rows, tier = evaluate(data, a.model)
    out = Path(a.out); out.mkdir(exist_ok=True)
    write_csv(out / "learned_rows.csv", rows_uniform(rows))
    tb, ts, gg = table(rows), tests(rows), gen_gap(rows)
    write_csv(out / "learned_table.csv", rows_uniform(tb)); write_csv(out / "learned_tests.csv", ts); write_csv(out / "learned_gap.csv", gg)
    summ = dict(scale=a.scale, model=a.model, footprint=fp, tier_stats=tier.stats(), seconds=time.time() - t0,
                table=tb, gen_gap=gg)
    (out / "learned_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    for r in tb:
        print(f"{r['split']:10s} n={r['n']:3d} gap {r['model_gap']:.2f} [{r['model_gap_lo']:.2f},{r['model_gap_hi']:.2f}] "
              f"rel {r['model_relgap_agg']:.3f} tier rel {r['tier_relgap_agg']:.3f} "
              f"succ m/t/x/0 {r['model_succ']:.3f}/{r['tier_succ']:.3f}/{r['exact_succ']:.3f}/{r['none_succ']:.3f} "
              f"fb {r['tier_fallback_rate']:.2f} (norepair {r['tier0_fallback_rate']:.2f}) viol {r['viol_per_scene']:.2f} "
              f"ms {r['infer_ms']:.1f}/{r['tier_ms']:.0f}/{r['exact_ms']:.0f}"
              + (f" DE n={r['n_de']} exact-de {r['exact_minus_de_obj']:+.2f} model-de {r['model_minus_de_obj']:+.2f}" if "n_de" in r else ""))
    print(json.dumps(fp), f"{time.time() - t0:.0f}s")


def rows_uniform(rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    return [{k: r.get(k, "") for k in keys} for r in rows]


if __name__ == "__main__":
    main()
