"""K2: point-estimate (posterior mean) vs posterior-robust (CVaR / worst-case over M posterior samples) correction
under controlled PROFILE error (SPEC 5.5, 11.2). The solver gets a profile = equal-weight posterior samples in the
Calibrator.samples() format, wrong in a controlled way; every number is scored under the TRUE HELD-OUT simulated
observer (Brettel/Vienot, held-out severity, jittered k/s_L/s_C) on its critical pairs (confusable + adjacent in the
original scene). Arms are '<level>|<method>' (+ 'ref|none'). Metrics per seed (mean over scenes x true observers):
min_de00, success_rate, deviation (max dE00 from original), worst_deg (max over cells of the loss in min dE00 vs no
correction), min_cell (worst cell's min dE00)."""
from __future__ import annotations
import numpy as np
from bench.registry import experiment
from adapters.synthetic import SynthConfig, generate
from core.observers import heldout_observers, KINDS
from core.profiles import samples_to_observers, posterior_mean_observer
from core.solver import Scene, SolverConfig, correct, none, naturalness, pair_scores

WRONG = {"protan": "deutan", "deutan": "protan", "tritan": "protan"}
# level -> (P(true kind), severity shift (+abs or xscale), width multiplier)
LEVELS = {
    "correct":        dict(p_true=1.0, sev=("add", 0.0), width=1.0),
    "wide_correct":   dict(p_true=1.0, sev=("add", 0.0), width=3.0),
    "sev_over":       dict(p_true=1.0, sev=("add", 0.4), width=1.0),
    "sev_under":      dict(p_true=1.0, sev=("mul", 0.2), width=1.0),
    "narrow_wrong":   dict(p_true=1.0, sev=("add", 0.4), width=0.25),
    "wide_wrong":     dict(p_true=1.0, sev=("add", 0.4), width=5.0),
    "type_ambiguous": dict(p_true=0.4, sev=("add", 0.0), width=1.0),
    "wrong_type":     dict(p_true=0.0, sev=("add", 0.0), width=1.0),
}
METHODS = ("point", "cvar", "worst")
SD_SEV, SD_LOGK, SD_RHO = 0.08, 0.2, 0.2


def make_posterior(o, level: str, rng, n: int = 64) -> dict:
    """Synthetic posterior (Calibrator.samples format) centred per `level` relative to the true observer `o`."""
    spec = LEVELS[level]
    w = spec["width"]
    kind = np.where(rng.random(n) < spec["p_true"], o.kind, WRONG[o.kind])
    mode, v = spec["sev"]
    c = o.severity + v if mode == "add" else o.severity * v
    sev = np.clip(c + rng.normal(0, SD_SEV * w, n), 0.02, 1.0)
    logk = np.log(o.k / np.sqrt(o.s_L * o.s_C)) + rng.normal(0, SD_LOGK * w, n)
    rho = np.log(o.s_L / o.s_C) + rng.normal(0, SD_RHO * w, n)
    e = {kd: np.where(kind == kd, sev, 0.0) for kd in KINDS}
    return dict(e_protan=e["protan"].tolist(), e_deutan=e["deutan"].tolist(), e_tritan=e["tritan"].tolist(),
                severity=sev.tolist(), k=np.exp(logk).tolist(), s_L=np.exp(rho / 2).tolist(),
                s_C=np.exp(-rho / 2).tolist(), logk=logk.tolist(), rho=rho.tolist())


def solve_with(scene, samples, method, m, alpha, scfg, seed):
    if method == "point":
        return correct(scene, [posterior_mean_observer(samples)], scfg, seed)
    obs = samples_to_observers(samples, m)
    return correct(scene, obs, SolverConfig(**{**scfg.__dict__, "risk": method, "alpha": alpha}), seed)


def score(scene, o, colors):
    g = scene.graph(o)
    ed = g.edges[scene.groups[g.edges[:, 0]] != scene.groups[g.edges[:, 1]]] if len(g.edges) else g.edges
    if not len(ed):
        return None
    s = pair_scores(colors, o, ed, g.tau, scene.illum_xyz)
    return dict(min_de00=s["min"], success_rate=s["success"], deviation=naturalness(scene.colors, colors, scene.protected)["max_de00"])


@experiment("profile_error")
def run(params, seed):
    cfg = SynthConfig.from_dict(params.get("scene", {}))
    snaps = list(generate(cfg, seed))[:: max(1, int(params.get("stride", 20)))][: int(params.get("n_scenes", 1))]
    scfg = SolverConfig(**params.get("solver", {}))
    m, alpha = int(params.get("m", 8)), float(params.get("alpha", 0.25))
    levels = params.get("levels", list(LEVELS))
    methods = params.get("methods", list(METHODS))
    truth = heldout_observers(int(params.get("n_true", 4)), seed)
    sc_kw = dict(adj_margin=float(params.get("adj_margin", 0.3)), cluster_tol=float(params.get("cluster_tol", 8.0)))
    cells: dict[str, list[dict]] = {}
    for si, snap in enumerate(snaps):
        scene = Scene.from_snapshot(snap, **sc_kw)
        for oi, o in enumerate(truth):
            ref = score(scene, o, none(scene.colors))
            if ref is None:
                continue
            cells.setdefault("ref|none", []).append(dict(ref, ref_min=ref["min_de00"]))
            for li, lv in enumerate(levels):
                post = make_posterior(o, lv, np.random.default_rng([seed, si, oi, li, 0xFE]), int(params.get("n_samples", 64)))
                for me in methods:
                    sol = solve_with(scene, post, me, m, alpha, scfg, seed)
                    cells.setdefault(f"{lv}|{me}", []).append(dict(score(scene, o, sol.colors), ref_min=ref["min_de00"]))
    out = {}
    for arm, rows in cells.items():
        mn = np.array([r["min_de00"] for r in rows])
        rf = np.array([r["ref_min"] for r in rows])
        out[arm] = dict(min_de00=float(mn.mean()), success_rate=float(np.mean([r["success_rate"] for r in rows])),
                        deviation=float(np.mean([r["deviation"] for r in rows])),
                        worst_deg=float(np.max(np.maximum(rf - mn, 0))), min_cell=float(mn.min()))
    return out
