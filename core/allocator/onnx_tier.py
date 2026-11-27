"""Tier 2 = learned solver via ONNX Runtime (SPEC 5.6 deployment). One forward pass predicts the whole palette conditioned on
z(theta) (the solver's observers' mean embedding, or a fixed calibrated-profile embedding); only the requested groups are
taken. Every call is verified (finite, constraints, objective not below the input palette, screen-space check from
core/solver on the solver's observers for pairs incident to the changed groups). A screen failure first gets a cheap local
repair (a few coarse coordinate-descent steps on the failing groups, warm-started at the prediction); if verification still
fails the exact solver (coarse tier,
warm-started from the prediction) runs instead. Fallbacks are counted (fallback_rate) and logged."""
from __future__ import annotations
import logging, os, time
from dataclasses import replace
from collections import Counter
from pathlib import Path
import numpy as np
from core.learned import scene_inputs, embed_many
from core.perception.color import ciede2000
from core.solver.solver import screen_check, coordinate_descent
from .tiers import Tier, CoarseExact

DEFAULT_MODEL = Path(__file__).with_name("learned.onnx")
log = logging.getLogger("sv.learned")


class OnnxTier(Tier):
    id, name = 2, "learned"

    def __init__(self, path=DEFAULT_MODEL, fallback: Tier | None = None, z=None, threads: int = 1, verify: bool = True,
                 repair_iters: int = 4):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = so.inter_op_num_threads = threads
        self.path = Path(path)
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.names = [i.name for i in self.sess.get_inputs()]
        self.fallback = fallback or CoarseExact()
        self.z = None if z is None else np.asarray(z, np.float32)
        self.verify_on = verify
        self.repair_iters = repair_iters                       # 0 = verification failure goes straight to the exact solver
        self.calls = self.fallbacks = 0
        self.reasons: Counter = Counter()
        self.last: dict = {}

    @property
    def fallback_rate(self):
        return self.fallbacks / max(self.calls, 1)

    def stats(self):
        return dict(calls=self.calls, fallbacks=self.fallbacks, fallback_rate=self.fallback_rate, reasons=dict(self.reasons))

    def infer(self, inp: dict) -> np.ndarray:
        n = len(inp["x"])
        f = dict(x=inp["x"][None], z=inp["z"][None], labg=inp["labg"][None], delta=inp["delta"][None], S=inp["S"][None],
                 A=inp["A"][None], mask=np.ones((1, n), np.float32))
        return self.sess.run(["lab"], {k: f[k] for k in self.names})[0][0]

    def predict(self, pb) -> np.ndarray:
        """-> palette Lab (E,3) predicted for every group (first object of each group; all members are identical)."""
        z = self.z if self.z is not None else embed_many(pb.obs)
        lab = self.infer(scene_inputs(pb.scene, z, delta=pb.cfg.delta, delta_mem=pb.cfg.delta_mem))
        first = np.unique(pb.scene.groups, return_index=True)[1]
        return lab[first].astype(float)

    def check(self, pb, P, Q, groups):
        """-> (ok, reason, groups touching a screen-space failure). Constraints apply to moved entries (an original colour
        already outside the L*/gamut bounds that the model leaves unchanged is not a model error)."""
        g = np.asarray(groups, int)
        if not np.isfinite(Q).all():
            return False, "nonfinite", g
        mv = g[np.abs(Q[g] - pb.lab0[g]).max(-1) > 1e-3]
        r, L = pb._rgb(Q[mv]), Q[mv, 0]
        if (ciede2000(Q[mv], pb.lab0[mv]) > pb.delta[mv] + 0.05).any() or ((r < -1e-3) | (r > 1 + 1e-3)).any() \
                or (L < pb.cfg.L_range[0] - 1e-3).any() or (L > pb.cfg.L_range[1] + 1e-3).any():
            return False, "constraint", g
        f = pb.objective(np.stack([P, Q]))
        if f[1] < f[0] - 1e-6:
            return False, "objective", g
        bad, _ = screen_check(pb.scene, pb.to_rgb(Q)[pb.scene.groups], pb.obs)
        gs = set(g.tolist())
        hit = sorted({x for a, b in bad for x in (a, b) if a in gs or b in gs} & gs)
        if hit:
            return False, "screen", np.array(hit, int)
        return True, "ok", g[:0]

    def repair(self, pb, Q, hit, deadline):
        """Cheap local polish (no grid, coarse steps) of the groups touching a screen failure, warm-started at Q."""
        cfg = replace(pb.cfg, max_iter=self.repair_iters, step_min=0.08)
        return coordinate_descent(pb, P0=Q, deadline=deadline, act=hit, cfg=cfg, init_grid=False)[0]

    def run(self, pb, P, groups, deadline=None, cache=None, keys=None):
        groups = np.asarray(groups, int)
        if not len(groups):
            return P
        t0 = time.perf_counter()
        Q = P.copy()
        try:
            Q[groups] = self.predict(pb)[groups]
        except Exception as e:                                   # model/runtime failure -> exact solver
            log.warning("learned tier inference failed: %s", e)
            Q[:] = np.nan
        if not self.verify_on:
            ok, why, hit = bool(np.isfinite(Q).all()), "unverified", groups[:0]
        else:
            ok, why, hit = self.check(pb, P, Q, groups)
            if why == "screen" and self.repair_iters > 0:
                Q = self.repair(pb, Q, hit, deadline)
                ok, why2, hit = self.check(pb, P, Q, groups)
                why = "repaired" if ok else why2
        self.calls += 1
        self.reasons[why] += 1
        self.last = dict(groups=len(groups), fallback=not ok, reason=why, ms=(time.perf_counter() - t0) * 1e3)
        if ok:
            return Q
        self.fallbacks += 1
        log.info("learned tier fallback (%s) on %d groups; rate %.3f", why, len(groups), self.fallback_rate)
        return self.fallback.run(pb, Q if why == "screen" else P, groups, deadline, cache, keys)   # warm start only if Q is sane


def load_learned_tier(path=DEFAULT_MODEL, **kw):
    """OnnxTier if the model file and onnxruntime are available and SV_LEARNED != 'standin', else None."""
    if os.environ.get("SV_LEARNED", "").lower() == "standin" or not Path(path).exists():
        return None
    try:
        return OnnxTier(path, **kw)
    except Exception as e:                                       # pragma: no cover - missing runtime
        log.warning("learned tier unavailable (%s); using stand-in", e)
        return None
