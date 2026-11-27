"""Traffic replay (SPEC 6 + 5.9): stabiliser + solver on Three.js/Rapier-recorded traffic streams with a REAL physics
collision (red-light runner). Streams are recorded headless by adapters/threejs/record.ts (seeded, fixed timestep) into
bench/logs/traffic/ (re-recorded if missing). Solver + stabiliser pair decisions see TRAIN observers of the kind only;
everything is scored under HELD-OUT observers.
Critical pairs: adjacent pairs touching a hazard-tagged object (involved vehicles, their hazard lights, beacon, smoke,
nearby objects) that are confusable without correction at the collision frame AND at the window end.
Response latency = frames (x export dt) from the collision snapshot to the first 3-frame-sustained resolution."""
from __future__ import annotations
import subprocess
from pathlib import Path

import numpy as np
from bench.experiments.stability_ablation import Ctx, arm_cfg, flicker_of
from bench.registry import experiment
from core.observers import heldout_observers, train_observers
from core.stabiliser import Stabiliser, de00
from protocol.io import read_ndjson
from protocol.models import Snapshot

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "bench/logs/traffic"
DT = 1 / 60
NORMAL_FLOOR = 5.0      # dE00 (normal vision) below which a pair is not a CVD problem
ARMS = ("uncorrected", "none", "event_driven", "no_ema", "combined")


SCENE_SRC = [ROOT / "adapters/threejs" / f for f in ("trafficSim.ts", "environments.ts", "physics.ts", "snapshot.ts")]


def scene_version():
    """Short hash of the scene sources: cached streams are re-recorded whenever the scene changes."""
    import hashlib
    h = hashlib.sha1()
    for p in SCENE_SRC:
        h.update(p.read_bytes())
    return h.hexdigest()[:8]


def log_path(env, seed, steps, every):
    return LOG_DIR / f"collision_{env}_s{seed}_n{steps}_e{every}_{scene_version()}.ndjson"


def ensure_logs(envs, seeds, steps, every, procs=8):
    """Record missing streams (one headless Node process per (env, seed), at most `procs` at a time)."""
    from concurrent.futures import ThreadPoolExecutor
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    seeds = [seeds] if isinstance(seeds, int) else list(seeds)
    todo = [(env, s) for s in seeds for env in envs
            if not (log_path(env, s, steps, every).exists() and log_path(env, s, steps, every).stat().st_size > 0)]

    def rec(job):
        env, s = job
        p = log_path(env, s, steps, every)
        tmp = p.with_suffix(".tmp")
        cmd = ["npx", "vite-node", "../adapters/threejs/record.ts", "--", "--scenario", "collision", "--env", env,
               "--seed", str(s), "--steps", str(steps), "--every", str(every), "--out", str(tmp)]
        r = subprocess.run(cmd, cwd=ROOT / "viewer", capture_output=True)
        if r.returncode:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"record failed for {p.name}: {r.stderr.decode()[-2000:]}")
        tmp.replace(p)
    with ThreadPoolExecutor(max(1, procs)) as ex:
        list(ex.map(rec, todo))


def prefetch_block(seed, n):
    """Seeds recorded together with `seed` (aligned blocks of n, so consecutive runner seeds record once, in parallel)."""
    b = seed - seed % max(n, 1)
    return list(range(b, b + max(n, 1)))


def window(snaps, pre, post):
    fc = next(i for i, s in enumerate(snaps) if any(e.type == "collision" for e in s.events))
    a, b = max(0, fc - pre), min(len(snaps), fc + post)
    return [s.model_copy(update={"frame": i}) for i, s in enumerate(snaps[a:b])], fc - a


def critical(ctx, k0, k1, n_obs):
    """(pair, observer) confusable uncorrected at k0 and k1, adjacent, touching a hazard-tagged object."""
    s0, s1 = ctx.snaps[k0], ctx.snaps[k1]
    haz = {o.id for o in s1.objects if "hazard" in o.task_tags}
    ids0, ids1 = {o.id for o in s0.objects}, {o.id for o in s1.objects}
    sc = ctx.scenes[k0]
    base0, base1 = ctx.lit(k0, {}), ctx.lit(k1, {})
    out = []
    for i, j in np.argwhere(np.triu(sc.adjacency, 1)):
        a, b = sc.ids[i], sc.ids[j]
        if not ({a, b} & haz) or not {a, b} <= ids1 or not {a, b} <= ids0:
            continue
        # CVD-specific only: same palette group is one colour by design; < NORMAL_FLOOR dE00 is confusable for anyone
        if sc.groups[i] == sc.groups[j] or float(de00(base0[a], base0[b])) < NORMAL_FLOOR:
            continue
        for oi in range(n_obs):
            d0, t0 = ctx.pair_D(k0, base0, (a, b), oi)
            d1, t1 = ctx.pair_D(k1, base1, (a, b), oi)
            if d0 < t0 and d1 < t1:
                out.append(((a, b), oi))
    return out


def run_arm(ctx, name, **kw):
    if name == "uncorrected":
        return dict(lits=[ctx.lit(f, {}) for f in range(len(ctx.snaps))], trig=[[] for _ in ctx.snaps], solves=0)
    st = Stabiliser(arm_cfg(name, **kw))
    lits, trig = [], []
    for f, s in enumerate(ctx.snaps):
        r = st.step(s, ctx.solve_fn, ctx.analyse_fn)
        lits.append(ctx.lit(f, r.displayed))
        trig.append(list(r.triggers))
    return dict(lits=lits, trig=trig, solves=st.n_solves)


def _load(env, seed, params):
    steps, every = int(params.get("steps", 540)), int(params.get("every", 3))
    snaps = list(read_ndjson(log_path(env, seed, steps, every), Snapshot))
    snaps, k0 = window(snaps, int(params.get("pre", 10)), int(params.get("post", 40)))
    rng = np.random.default_rng([seed, 0x7AFF])
    kind = ("protan", "deutan")[int(rng.integers(2))]
    tr = [o for o in train_observers(60, seed) if o.kind == kind][:int(params.get("n_train", 3))]
    return snaps, k0, every * DT, kind, tr


def census_env(env, seed, params):
    """Scene-design check under TRAIN observers only: CVD-confusable adjacent hazard pairs (unique pairs)."""
    snaps, k0, _, kind, tr = _load(env, seed, params)
    ctx = Ctx(snaps, tr[0], {o.id: o for o in tr}, tr, params.get("solver"))
    crit = critical(ctx, k0, len(snaps) - 1, len(tr))
    return sorted({p for p, _ in crit}), kind


@experiment("traffic_census")
def run_census(params, seed):
    envs = params.get("envs", ["day", "night", "fog", "rain"])
    pre = prefetch_block(seed, int(params.get("prefetch", 1)))
    ensure_logs(envs, pre, int(params.get("steps", 540)), int(params.get("every", 3)), int(params.get("record_procs", 8)))
    out = {}
    for env in envs:
        pairs, _ = census_env(env, seed, params)
        out[env] = {"train_pairs": float(len(pairs))}
    return out


def run_env(env, seed, params):
    snaps, k0, dt, kind, tr = _load(env, seed, params)
    held = [o for o in heldout_observers(40, seed) if o.kind == kind][:int(params.get("n_heldout", 4))]
    ctx = Ctx(snaps, held[0], {o.id: o for o in tr} | {held[0].id: held[0]}, held, params.get("solver"))
    ctx.precompute(int(params.get("workers", 0)))
    n = len(snaps)
    crit = critical(ctx, k0, n - 1, len(held))
    kw = dict(compute_frames=int(params.get("compute_frames", 1)), default_dt=dt, **params.get("stab", {}))
    runs = {a: run_arm(ctx, a, **(kw if a != "uncorrected" else {})) for a in params.get("arms", ARMS)}
    fl0 = flicker_of(runs["uncorrected"]["lits"])
    ms = dt * 1e3
    out = {}
    for a, r in runs.items():
        res, cens, succ, dmin = [], 0, [], []
        for pair, oi in crit:
            ser = [ctx.pair_D(f, r["lits"][f], pair, oi) for f in range(k0, n)]
            ok = [d >= t for d, t in ser]
            first = next((k for k in range(len(ok)) if all(ok[k:k + 3])), None)
            if first is None:
                cens += 1                       # right-censored at the window end, never counted as a latency
            else:
                res.append(first)
            succ.append(np.mean(ok))
            dmin.append(min(d for d, _ in ser))
        m = dict(solves_per_frame=r["solves"] / n, flicker_excess=flicker_of(r["lits"]) - fl0,
                 n_critical=float(len(crit)), no_critical=float(not crit))
        trig = next((f - k0 for f in range(k0, n) if "collision" in r["trig"][f]), None)
        if a != "uncorrected":
            m["trigger_missed"] = float(trig is None)
            if trig is not None:                # trigger fires on snapshot k0+trig; overrides show compute_frames later
                disp = trig + kw["compute_frames"]
                m.update(collision_trigger_frames=float(trig), collision_trigger_ms=trig * ms,
                         display_frames=float(disp), display_ms=disp * ms)
        if crit:
            win = n - k0
            m.update(unresolved=cens / len(crit), n_censored_pairs=float(cens), crit_success=float(np.mean(succ)),
                     crit_min_de00=float(np.mean(dmin)),
                     latency_lb_frames=float(np.mean(res + [win] * cens)), latency_lb_ms=float(np.mean(res + [win] * cens)) * ms)
            if res:                             # resolved pairs only
                m.update(latency_frames=float(np.mean(res)), latency_ms=float(np.mean(res)) * ms)
        out[a] = m
    return out


PER_ENV = ("latency_frames", "latency_ms", "latency_lb_ms", "unresolved", "crit_success", "n_critical", "no_critical")


@experiment("traffic_replay")
def run(params, seed):
    """Per-arm means over environments of the metrics each environment has (latency only where critical pairs exist
    and were resolved; censored pairs -> unresolved / latency_lb_*). no_critical_envs = environments without any
    critical pair (latency censored for them, never zero)."""
    envs = params.get("envs", ["day", "night", "fog", "rain"])
    steps, every = int(params.get("steps", 540)), int(params.get("every", 3))
    pre = prefetch_block(seed, int(params.get("prefetch", 1)))
    ensure_logs(envs, pre, steps, every, int(params.get("record_procs", 8)))
    per = {env: run_env(env, seed, params) for env in envs}
    out = {}
    for a in per[envs[0]]:
        keys = {k for env in envs for k in per[env][a]}
        out[a] = {"no_critical_envs": float(sum(per[env][a]["no_critical"] for env in envs))}
        for k in sorted(keys):
            vals = [per[env][a][k] for env in envs if k in per[env][a]]
            out[a][k] = float(np.mean(vals))
            if k in PER_ENV:
                for env in envs:
                    if k in per[env][a]:
                        out[a][f"{k}_{env}"] = per[env][a][k]
    return out


def censoring_report(raw_csv):
    """Explicit censoring table from a traffic_replay raw CSV: seeds without critical pairs, per env and overall."""
    import csv
    rows = list(csv.DictReader(open(raw_csv)))
    seeds = sorted({int(r["seed"]) for r in rows})
    arm = rows[0]["arm"]
    val = {(int(r["seed"]), r["metric"]): float(r["value"]) for r in rows if r["arm"] == arm}
    envs = sorted({m[len("no_critical_"):] for (_, m) in val if m.startswith("no_critical_") and m != "no_critical_envs"})
    lines = [f"seeds={len(seeds)}"]
    for env in envs:
        lines.append(f"{env}: no_critical_seeds={int(sum(val[(s, f'no_critical_{env}')] for s in seeds))}")
    lines.append(f"all_envs_no_critical_seeds={sum(val[(s, 'no_critical_envs')] == len(envs) for s in seeds)}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    print(censoring_report(sys.argv[1] if len(sys.argv) > 1 else ROOT / "bench/results/traffic_replay_raw.csv"))
