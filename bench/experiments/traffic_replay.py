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


def log_path(env, seed, steps, every):
    return LOG_DIR / f"collision_{env}_s{seed}_n{steps}_e{every}.ndjson"


def ensure_logs(envs, seed, steps, every):
    """Record missing streams (one headless Node process per env, in parallel)."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    procs = []
    for env in envs:
        p = log_path(env, seed, steps, every)
        if p.exists() and p.stat().st_size > 0:
            continue
        cmd = ["npx", "vite-node", "../adapters/threejs/record.ts", "--", "--scenario", "collision", "--env", env,
               "--seed", str(seed), "--steps", str(steps), "--every", str(every), "--out", str(p)]
        procs.append((p, subprocess.Popen(cmd, cwd=ROOT / "viewer", stdout=subprocess.PIPE, stderr=subprocess.PIPE)))
    for p, pr in procs:
        out, err = pr.communicate()
        if pr.returncode:
            p.unlink(missing_ok=True)
            raise RuntimeError(f"record failed for {p.name}: {err.decode()[-2000:]}")


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


def run_env(env, seed, params):
    steps, every = int(params.get("steps", 540)), int(params.get("every", 3))
    snaps = list(read_ndjson(log_path(env, seed, steps, every), Snapshot))
    snaps, k0 = window(snaps, int(params.get("pre", 10)), int(params.get("post", 40)))
    dt = every * DT
    rng = np.random.default_rng([seed, 0x7AFF])
    kind = ("protan", "deutan")[int(rng.integers(2))]
    tr = [o for o in train_observers(60, seed) if o.kind == kind][:int(params.get("n_train", 3))]
    held = [o for o in heldout_observers(40, seed) if o.kind == kind][:int(params.get("n_heldout", 4))]
    ctx = Ctx(snaps, held[0], {o.id: o for o in tr} | {held[0].id: held[0]}, held, params.get("solver"))
    ctx.precompute(int(params.get("workers", 0)))
    n = len(snaps)
    crit = critical(ctx, k0, n - 1, len(held))
    kw = dict(compute_frames=int(params.get("compute_frames", 1)), default_dt=dt, **params.get("stab", {}))
    runs = {a: run_arm(ctx, a, **(kw if a != "uncorrected" else {})) for a in params.get("arms", ARMS)}
    fl0 = flicker_of(runs["uncorrected"]["lits"])
    out = {}
    for a, r in runs.items():
        lat, unres, succ, dmin = [], [], [], []
        for pair, oi in crit:
            ser = [ctx.pair_D(f, r["lits"][f], pair, oi) for f in range(k0, n)]
            ok = [d >= t for d, t in ser]
            first = next((k for k in range(len(ok)) if all(ok[k:k + 3])), None)
            lat.append(len(ok) if first is None else first)
            unres.append(first is None)
            succ.append(np.mean(ok))
            dmin.append(min(d for d, _ in ser))
        trig = next((f - k0 for f in range(k0, n) if "collision" in r["trig"][f]), n - k0)
        m = dict(collision_trigger_frames=float(trig), solves_per_frame=r["solves"] / n,
                 flicker_excess=flicker_of(r["lits"]) - fl0, n_critical=float(len(crit)))
        if crit:
            m.update(latency_frames=float(np.mean(lat)), latency_ms=float(np.mean(lat)) * dt * 1e3,
                     unresolved=float(np.mean(unres)), crit_success=float(np.mean(succ)), crit_min_de00=float(np.mean(dmin)))
        out[a] = m
    return out


@experiment("traffic_replay")
def run(params, seed):
    envs = params.get("envs", ["day", "night", "fog", "rain"])
    ensure_logs(envs, seed, int(params.get("steps", 540)), int(params.get("every", 3)))
    per = {env: run_env(env, seed, params) for env in envs}
    out = {}
    for a in per[envs[0]]:
        keys = {k for env in envs for k in per[env][a]}
        out[a] = {}
        for k in sorted(keys):
            vals = [per[env][a][k] for env in envs if k in per[env][a]]
            out[a][k] = float(np.mean(vals))
            if k in ("latency_ms", "crit_success", "n_critical"):
                for env in envs:
                    if k in per[env][a]:
                        out[a][f"{k}_{env}"] = per[env][a][k]
    return out
