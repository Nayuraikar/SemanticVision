"""Calibration parameter recovery vs trial count (SPEC 5.3). Truth = HELD-OUT simulated observers (Brettel/Vienot,
severity in held-out range, jittered k/s_L/s_C, jittered bg term and lapse) while the estimator models Machado mixtures
with fixed lapse -> model mismatch by construction. Arms: eig@n (max-EIG selection), random@n (random stimuli),
eig@stop (stop rule). Scores: predictive error on held-out random probe stimuli never used for selection
(mean |P_hat(different) - P_true|), logk/rho error, dominant-type accuracy, 90% CI coverage of logk."""
from __future__ import annotations
import numpy as np
from bench.registry import experiment
from calibration import Calibrator, Session, CalibConfig, SimObserver
from calibration import model as M
from calibration import stimuli as S
from core.observers import heldout_observers


def truth_from(o, rng) -> SimObserver:
    return SimObserver(o.model, o.kind, o.severity, o.k, o.s_L, o.s_C, bg_gain=0.25 * float(np.exp(rng.normal(0, 0.25))),
                       lapse=0.05)


def probes(n, seed):
    r = np.random.default_rng([seed, 0x9B0])
    return r.integers(0, S.pool().n_stim, n)


def p_true(truth: SimObserver, stims):
    pl = S.pool()
    out = []
    for s in stims:
        p, size, bg = S.decode(s)
        out.append(M.p_diff(truth.ratio(pl.c1[p], pl.c2[p], size, bg), truth.gamma, truth.lapse))
    return np.array(out)


def score(cal: Calibrator, truth: SimObserver, stims, ptrue) -> dict:
    w = cal.weights()
    phat = w @ M.p_diff(cal.ratio(stims))
    c = truth.canonical()
    st = cal.stats()
    e = [st["e_protan"]["mean"], st["e_deutan"]["mean"], st["e_tritan"]["mean"]]
    lk = cal.parts.logk
    return dict(pred_err=float(np.mean(np.abs(phat - ptrue))), logk_err=abs(float(w @ lk) - c["logk"]),
                rho_err=abs(float(w @ cal.parts.rho) - c["rho"]), kind_acc=float(S.KINDS[int(np.argmax(e))] == truth.kind),
                width=cal.width(), cover_logk=float(np.quantile(lk[w > 1e-9], 0.05) <= c["logk"] <= np.quantile(lk[w > 1e-9], 0.95)))


@experiment("calibration_recovery")
def run(params, seed):
    n_obs, max_n = int(params.get("n_obs", 4)), int(params.get("max_trials", 40))
    marks = [m for m in params.get("marks", [0, 5, 10, 20, 30, 40]) if m <= max_n]
    cfg = CalibConfig(max_trials=max_n, min_trials=int(params.get("min_trials", 12)),
                      target_width=float(params.get("target_width", 0.4)), n_particles=int(params.get("n_particles", 1000)),
                      mode="mixed")
    acc: dict[str, list[dict]] = {}
    for i, o in enumerate(heldout_observers(n_obs, seed)):
        rng = np.random.default_rng([seed, i, 0xA11])
        truth = truth_from(o, rng)
        pr = probes(int(params.get("n_probes", 120)), seed * 1000 + i)
        pt = p_true(truth, pr)
        cs = Calibrator(seed * 100 + i, cfg.n_particles)                 # shares nothing with eig session below
        # --- eig arm via Session (selection + stop rule exactly as served) ---
        s = Session(CalibConfig(**{**cfg.to_dict(), "seed": seed * 100 + i, "min_trials": max_n, "max_trials": max_n}))
        stop_at, traj = None, {}
        if 0 in marks:
            traj[0] = score(s.cal, truth, pr, pt)
        while (st := s.next()) is not None:
            s.answer(st["trial_id"], truth.respond(st["mode"], st["c1"], st["c2"], st["size_deg"], st["bg_lum"], rng))
            if s.n in marks:
                traj[s.n] = score(s.cal, truth, pr, pt)
            if stop_at is None and s.n >= cfg.min_trials and s.cal.width() <= cfg.target_width:
                stop_at = s.n
                traj["stop"] = {**score(s.cal, truth, pr, pt), "n_trials": float(s.n)}
        if "stop" not in traj:
            traj["stop"] = {**score(s.cal, truth, pr, pt), "n_trials": float(max_n)}
        for m, v in traj.items():
            acc.setdefault(f"eig@{m}", []).append(v)
        # --- random-stimulus baseline ---
        rr = np.random.default_rng([seed, i, 0xBAD])
        if 0 in marks:
            acc.setdefault("random@0", []).append(score(cs, truth, pr, pt))
        pl = S.pool()
        for t in range(max_n):
            stim = int(rr.integers(pl.n_stim))
            while stim in cs.shown:
                stim = int(rr.integers(pl.n_stim))
            mode = ("same_different", "slider")[t % 2]
            p, size, bg = S.decode(stim)
            cs.update(stim, mode, truth.respond(mode, pl.c1[p], pl.c2[p], size, bg, rng))
            if cs.n in marks:
                acc.setdefault(f"random@{cs.n}", []).append(score(cs, truth, pr, pt))
    return {arm: {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} for arm, rows in acc.items()}
