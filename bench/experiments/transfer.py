"""Held-out transfer (K5): the UNCHANGED live pipeline (core.server.Corrector = lighting-aware scene + allocator with
tier-2 ONNX + solver; Stabiliser(all_on), as /correct) on Three.js traffic streams vs instrument-panel dashboard
streams. Nothing here (or anywhere) is tuned on the dashboard.
Same observer draw per seed for both environments: kind, TRAIN observers (solver) and HELD-OUT observers (scoring).
Critical pairs and censuses: traffic_replay.critical(), unchanged. Anchor frame = collision (traffic) / escalation to
the red master warning (dashboard); window -pre/+post snapshots.
A seed/condition without a critical pair is CENSORED for the pair metrics (metric absent, never 0); without tier-2
calls it is censored for the fallback rate.
Latency stages per frame (ms, bench's own mapping, measured around unmodified core calls):
  detect = Corrector.analyse (scene, importance track, pair ratios)   decide = triggers + pair hysteresis
  allocate = allocator planning (Diagnostics.allocator_ms)            correct = rest of Corrector.run
  stabilise = rest of Stabiliser.step                                 total = whole Stabiliser.step
transfer.json keeps the recorded 40 ms budget (primary); transfer_b500.json raises it to 500 ms for BOTH
environments because at 40 ms the allocator has no budget left on these ~100-object scenes (secondary, chosen after
the primary result).
Run:    python -m bench run --config transfer          (or transfer_b500)
Report: python -m bench.experiments.transfer [raw.csv]"""
from __future__ import annotations
import csv
import hashlib
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy import stats as st

from bench import stats
from bench.ablation import holm
from bench.experiments import traffic_replay as TR
from bench.experiments.stability_ablation import Ctx, flicker_of
from bench.registry import experiment
from core.observers import heldout_observers, train_observers
from core.server import Corrector
from core.stabiliser import StabConfig, Stabiliser
from protocol.io import read_ndjson
from protocol.models import Snapshot

ROOT = Path(__file__).resolve().parents[2]
DASH_DIR = ROOT / "bench/logs/dashboard"
DASH_SRC = [ROOT / "adapters/threejs" / f for f in ("dashboardSim.ts", "snapshot.ts", "lighting.ts", "rng.ts")]
ENVS = {"traffic": ("day", "night", "fog", "rain"), "dashboard": ("normal", "dim", "glare")}
MAX_WORKERS = 4
STAGES = ("detect", "decide", "correct", "allocate", "stabilise", "total")
OUTCOMES = ("min_de00", "success", "infeasible_pairs", "tier2_fallback", "flicker_excess")
LATENCY = tuple(f"ms_{s}" for s in STAGES)
TIERS = ("skip", "reuse", "learned", "coarse", "fine")
CONTEXT = ("n_critical", "uncorrected_min_de00", "unresolved", "tier2_calls", "solves_per_frame", "n_candidates",
           "candidate_min_ratio") + tuple(f"tier_share_{t}" for t in TIERS)


# ---------------------------------------------------------------- streams
def dash_version():
    h = hashlib.sha1()
    for p in DASH_SRC:
        h.update(p.read_bytes())
    return h.hexdigest()[:8]


def dash_log_path(light, seed, steps, every):
    return DASH_DIR / f"alarm_{light}_s{seed}_n{steps}_e{every}_{dash_version()}.ndjson"


def ensure_dash_logs(lights, seeds, steps, every, procs=MAX_WORKERS):
    from concurrent.futures import ThreadPoolExecutor
    DASH_DIR.mkdir(parents=True, exist_ok=True)
    todo = [(l, s) for s in seeds for l in lights if not dash_log_path(l, s, steps, every).exists()]

    def rec(job):
        light, s = job
        p = dash_log_path(light, s, steps, every)
        tmp = p.with_suffix(".tmp")
        cmd = ["npx", "vite-node", "../adapters/threejs/record.ts", "--", "--scene", "dashboard", "--scenario", "alarm",
               "--env", light, "--seed", str(s), "--steps", str(steps), "--every", str(every), "--out", str(tmp)]
        r = subprocess.run(cmd, cwd=ROOT / "viewer", capture_output=True)
        if r.returncode:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"record failed for {p.name}: {r.stderr.decode()[-2000:]}")
        tmp.replace(p)
    with ThreadPoolExecutor(max(1, min(procs, MAX_WORKERS))) as ex:
        list(ex.map(rec, todo))


def escalation_window(snaps, pre, post):
    """Same slicing as traffic_replay.window, anchored on the red master warning."""
    fc = next(i for i, s in enumerate(snaps)
              if any(e.object_id == "master_warning" and e.data.get("to") == "red" for e in s.events))
    a, b = max(0, fc - pre), min(len(snaps), fc + post)
    return [s.model_copy(update={"frame": i}) for i, s in enumerate(snaps[a:b])], fc - a


def load_stream(env, cond, seed, params):
    steps, every = int(params.get("steps", 540)), int(params.get("every", 3))
    path = (TR.log_path if env == "traffic" else dash_log_path)(cond, seed, steps, every)
    cut = TR.window if env == "traffic" else escalation_window
    snaps, k0 = cut(list(read_ndjson(path, Snapshot)), int(params.get("pre", 10)), int(params.get("post", 40)))
    return snaps, k0, every / 60


def observers_for(seed, params):
    """One draw per seed, independent of the environment (same rule as traffic_replay)."""
    rng = np.random.default_rng([seed, 0x7AFF])
    kind = ("protan", "deutan")[int(rng.integers(2))]
    tr = [o for o in train_observers(60, seed) if o.kind == kind][:int(params.get("n_train", 3))]
    held = [o for o in heldout_observers(40, seed) if o.kind == kind][:int(params.get("n_heldout", 4))]
    return tr, held


# ---------------------------------------------------------------- censoring diagnostic
def candidates(ctx, k0, k1, n_obs):
    """Why is a stream censored? -> (n observer-independent candidate pairs: critical()'s filter without the observer
    test, min over candidates x held-out observers of max(dE00/tau at k0, k1); a pair is critical iff that is < 1)."""
    s0, s1 = ctx.snaps[k0], ctx.snaps[k1]
    haz = {o.id for o in s1.objects if "hazard" in o.task_tags}
    ids = {o.id for o in s0.objects} & {o.id for o in s1.objects}
    sc = ctx.scenes[k0]
    base0, base1 = ctx.lit(k0, {}), ctx.lit(k1, {})
    n, best = 0, np.inf
    for i, j in np.argwhere(np.triu(sc.adjacency, 1)):
        a, b = sc.ids[i], sc.ids[j]
        if not ({a, b} & haz) or not {a, b} <= ids:
            continue
        if sc.groups[i] == sc.groups[j] or float(TR.de00(base0[a], base0[b])) < TR.NORMAL_FLOOR:
            continue
        n += 1
        for oi in range(n_obs):
            r = [d / max(t, 1e-9) for d, t in (ctx.pair_D(k0, base0, (a, b), oi), ctx.pair_D(k1, base1, (a, b), oi))]
            best = min(best, max(r))
    return n, float(best)


# ---------------------------------------------------------------- pipeline
def _timed(fn, acc, key):
    def wrapped(*a, **k):
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            acc[key] += (time.perf_counter() - t) * 1e3
    return wrapped


def run_pipeline(snaps, dt, display, table):
    """The /correct path with a fresh Corrector + stabiliser. display: only used for reporting inside the pipeline;
    it is held-out, so the solver optimises for the TRAIN observers in `table` (core.server.solver_observers)."""
    cor = Corrector()
    cor.table = table
    t2 = cor.allocator.tiers[2]
    if type(t2).__name__ != "OnnxTier":
        raise RuntimeError("tier 2 is not the ONNX solver; refusing to report a stand-in run")
    stab = Stabiliser(StabConfig.all_on(default_dt=dt))
    acc = dict.fromkeys(("detect", "decide", "run", "allocate"), 0.0)
    stab.det.detect = _timed(stab.det.detect, acc, "decide")
    stab.pairs.update = _timed(stab.pairs.update, acc, "decide")
    infeasible, tiers = [], {}

    def solve_fn(s):
        before = cor.solves
        ov, an = _timed(cor.run, acc, "run")(s, display)
        if cor.solves > before:                         # real solve (not a signature-cache hit)
            for name, c in ov.diagnostics.tier_counts.items():
                tiers[name] = tiers.get(name, 0) + c
            acc["allocate"] += ov.diagnostics.allocator_ms
            infeasible.append(len(an.cue_pairs))        # object pairs still confusable for the solver's observers
        return ov, an

    analyse_fn = _timed(lambda s: cor.analyse(s, display), acc, "detect")
    shown, frames = [], []
    for s in snaps:
        for k in acc:
            acc[k] = 0.0
        t = time.perf_counter()
        r = stab.step(s, solve_fn, analyse_fn)
        total = (time.perf_counter() - t) * 1e3
        shown.append(r.displayed)
        frames.append(dict(detect=acc["detect"], decide=acc["decide"], allocate=acc["allocate"],
                           correct=acc["run"] - acc["allocate"],
                           stabilise=total - acc["detect"] - acc["decide"] - acc["run"], total=total))
    return dict(shown=shown, frames=frames, infeasible=infeasible, solves=stab.n_solves, tiers=tiers,
                tier2_calls=t2.calls, tier2_fallbacks=t2.fallbacks)


def score(snaps, k0, tr, held, dt):
    """-> {metric: value} for one stream; pair metrics only when critical pairs exist."""
    table = {o.id: o for o in tr} | {held[0].id: held[0]}
    res = run_pipeline(snaps, dt, held[0], table)
    ctx = Ctx(snaps, held[0], table, held)
    n = len(snaps)
    base = [ctx.lit(f, {}) for f in range(n)]
    lits = [ctx.lit(f, d) for f, d in enumerate(res["shown"])]
    crit = TR.critical(ctx, k0, n - 1, len(held))
    m = dict(n_critical=float(len(crit)), no_critical=float(not crit),
             flicker_excess=flicker_of(lits) - flicker_of(base), solves_per_frame=res["solves"] / n,
             tier2_calls=float(res["tier2_calls"]))
    n_cand, ratio = candidates(ctx, k0, n - 1, len(held))
    m["n_candidates"] = float(n_cand)
    if n_cand:
        m["candidate_min_ratio"] = ratio
    chosen = sum(res["tiers"].values())
    if chosen:
        m.update({f"tier_share_{t}": res["tiers"].get(t, 0) / chosen for t in TIERS})
    if res["infeasible"]:
        m["infeasible_pairs"] = float(np.mean(res["infeasible"]))
    if res["tier2_calls"]:
        m["tier2_fallback"] = res["tier2_fallbacks"] / res["tier2_calls"]
    for s in STAGES:
        m[f"ms_{s}"] = float(np.mean([f[s] for f in res["frames"]]))
    if crit:
        dmin, umin, succ, unres = [], [], [], []
        for pair, oi in crit:
            ser = [ctx.pair_D(f, lits[f], pair, oi) for f in range(k0, n)]
            ok = [d >= t for d, t in ser]
            dmin.append(min(d for d, _ in ser))
            umin.append(min(ctx.pair_D(f, base[f], pair, oi)[0] for f in range(k0, n)))
            succ.append(np.mean(ok))
            unres.append(not any(all(ok[k:k + 3]) for k in range(len(ok) - 2)))
        m.update(min_de00=float(np.mean(dmin)), uncorrected_min_de00=float(np.mean(umin)),
                 success=float(np.mean(succ)), unresolved=float(np.mean(unres)))
    return m


def run_condition(job):
    env, cond, seed, params = job
    tr, held = observers_for(seed, params)
    snaps, k0, dt = load_stream(env, cond, seed, params)
    if params.get("budget_ms"):                 # secondary run only; default = budget as recorded (40 ms)
        snaps = [s.model_copy(update={"budget_ms": float(params["budget_ms"])}) for s in snaps]
    return env, cond, score(snaps, k0, tr, held, dt)


# ---------------------------------------------------------------- experiment
_POOL = None


def _pool(workers):
    global _POOL
    if _POOL is None:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[k] = "1"
        _POOL = ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"))
    return _POOL


@experiment("transfer")
def run(params, seed):
    """-> {"pipeline": {"<env>_<condition>|<metric>": value}}; censored metrics are absent. (One runner arm on purpose:
    the runner's paired test assumes complete data; use report() for the censoring-aware tests.)"""
    envs = params.get("envs", ENVS)
    steps, every = int(params.get("steps", 540)), int(params.get("every", 3))
    workers = min(int(params.get("workers", MAX_WORKERS)), MAX_WORKERS)
    block = TR.prefetch_block(seed, int(params.get("prefetch", 1)))
    TR.ensure_logs(list(envs["traffic"]), block, steps, every, max(1, workers))
    ensure_dash_logs(list(envs["dashboard"]), block, steps, every, max(1, workers))
    jobs = [(env, c, seed, params) for env in envs for c in envs[env]]
    order = np.random.default_rng([seed, 0x0D3]).permutation(len(jobs))    # no fixed slot per environment (timing)
    jobs = [jobs[i] for i in order]
    done = list(_pool(workers).map(run_condition, jobs)) if workers else [run_condition(j) for j in jobs]
    out = {f"{env}_{cond}|{k}": v for env, cond, m in done for k, v in m.items()}
    return {"pipeline": dict(sorted(out.items()))}


# ---------------------------------------------------------------- report
def load_raw(path):
    """raw.csv -> {(condition, metric): {seed: value}}, sorted seeds."""
    data, seeds = {}, set()
    for r in csv.DictReader(open(path)):
        cond, metric = r["metric"].split("|")
        data.setdefault((cond, metric), {})[int(r["seed"])] = float(r["value"])
        seeds.add(int(r["seed"]))
    return data, sorted(seeds)


def group_values(data, seeds, conds, metric):
    """Per-seed mean over the conditions that HAVE the metric; seeds with none are censored (absent)."""
    out = {}
    for s in seeds:
        v = [data[(c, metric)][s] for c in conds if s in data.get((c, metric), {})]
        if v:
            out[s] = float(np.mean(v))
    return out


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    nan = float("nan")
    diff = float(a.mean() - b.mean()) if len(a) and len(b) else nan
    if len(a) < 2 or len(b) < 2:
        return dict(diff=diff, ci_lo=nan, ci_hi=nan, p=nan)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = float(np.sqrt(va + vb))
    if se == 0:
        return dict(diff=diff, ci_lo=diff, ci_hi=diff, p=1.0 if diff == 0 else 0.0)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    h = float(st.t.ppf(0.975, df)) * se
    return dict(diff=diff, ci_lo=diff - h, ci_hi=diff + h, p=float(st.ttest_ind(a, b, equal_var=False).pvalue))


def gap(dash: dict, traf: dict):
    """dashboard - traffic. Paired t over seeds present on both sides (same observer draw); Welch on all uncensored
    seeds when fewer than 2 are matched. welch_p is also kept as a sensitivity check when seeds are unmatched."""
    both = sorted(set(dash) & set(traf))
    w = welch(list(dash.values()), list(traf.values()))
    unmatched = len(set(dash) ^ set(traf))
    base = dict(n_dash=len(dash), n_traffic=len(traf), n_matched=len(both), welch_p=w["p"] if unmatched else float("nan"))
    if len(both) < 2:
        return dict(test="welch", gap=w["diff"], ci_lo=w["ci_lo"], ci_hi=w["ci_hi"], p=w["p"], **base)
    a, b = [dash[s] for s in both], [traf[s] for s in both]
    sm = stats.summarize(np.array(a) - np.array(b))
    return dict(test="paired", gap=sm["mean"], ci_lo=sm["ci_lo"], ci_hi=sm["ci_hi"], p=stats.paired(a, b)["t_p"], **base)


def report(raw, out_dir=None):
    data, seeds = load_raw(raw)
    conds = {env: [f"{env}_{c}" for c in cs] for env, cs in ENVS.items()}
    groups = {**{e: cs for e, cs in conds.items()}, **{c: [c] for cs in conds.values() for c in cs}}
    table = []
    for name, cs in groups.items():
        for met in OUTCOMES + LATENCY + CONTEXT:
            v = list(group_values(data, seeds, cs, met).values())
            row = dict(group=name, metric=met, n=len(v), censored_seeds=len(seeds) - len(v))
            sm = stats.summarize(v) if v else dict(mean=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"))
            table.append(dict(row, mean=sm["mean"], ci_lo=sm["ci_lo"], ci_hi=sm["ci_hi"]))
    tests = []
    for name in ["dashboard", *conds["dashboard"]]:
        for met in OUTCOMES + LATENCY:
            g = gap(group_values(data, seeds, groups[name], met), group_values(data, seeds, conds["traffic"], met))
            tests.append(dict(comparison=f"{name} - traffic", metric=met,
                              family="outcome" if met in OUTCOMES else "latency", **g))
    for fam in ("outcome", "latency"):                       # Holm within each family, over all comparisons
        part = [t for t in tests if t["family"] == fam]
        for t, q in zip(part, holm([t["p"] for t in part])):
            t["p_holm"] = float(q) if not np.isnan(t["p"]) else float("nan")
    censor = []
    for name, cs in groups.items():
        no_crit = sum(all(data.get((c, "no_critical"), {}).get(s) == 1.0 for c in cs) for s in seeds)
        cells = sum(data.get((c, "no_critical"), {}).get(s) == 1.0 for c in cs for s in seeds)
        no_t2 = sum(s not in data.get((c, "tier2_fallback"), {}) for c in cs for s in seeds)
        censor.append(dict(group=name, seeds=len(seeds), seeds_without_critical=int(no_crit),
                           condition_seeds=len(cs) * len(seeds), condition_seeds_without_critical=int(cells),
                           condition_seeds_without_tier2_calls=int(no_t2)))
    out = Path(out_dir) if out_dir else Path(raw).parent
    for suffix, rows in (("table", table), ("tests", tests), ("censoring", censor)):
        with open(out / f"{Path(raw).stem.removesuffix('_raw')}_{suffix}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return table, tests, censor


def print_report(table, tests, censor):
    val = {(r["group"], r["metric"]): r for r in table}
    names = list(dict.fromkeys(r["group"] for r in table))
    cols = OUTCOMES + CONTEXT + LATENCY
    print("group," + ",".join(cols))
    for g in names:
        print(g + "," + ",".join(f"{val[(g, m)]['mean']:.3g}(n{val[(g, m)]['n']})" for m in cols))
    print("\ncensoring")
    for c in censor:
        print(",".join(f"{k}={v}" for k, v in c.items()))
    print("\ngaps (dashboard - traffic)")
    for t in tests:
        print(f"{t['comparison']:28} {t['metric']:17} {t['test']:6} n={t['n_matched']}/{t['n_dash']}/{t['n_traffic']} "
              f"gap={t['gap']:.3g} [{t['ci_lo']:.3g},{t['ci_hi']:.3g}] p={t['p']:.3g} holm={t['p_holm']:.3g} welch={t['welch_p']:.3g}")


if __name__ == "__main__":
    import sys
    print_report(*report(sys.argv[1] if len(sys.argv) > 1 else ROOT / "bench/results/transfer_raw.csv"))
