"""Particle-filter posterior over observer parameters with max-expected-information-gain stimulus selection
(QUEST+ style: for each candidate stimulus, EIG = H(E[p(resp)]) - E[H(p(resp))] over the posterior).
Deterministic given (seed, history): every random draw is keyed on (seed, trial count)."""
from __future__ import annotations
import numpy as np
from . import model as M
from . import stimuli as S

MODES = ("same_different", "slider")
MAP_DELTAS = np.geomspace(0.02, 1.0, 24)


def _H(p, axis=-1):
    return -(p * np.log2(np.clip(p, 1e-12, 1))).sum(axis)


def wquantile(x, w, q):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    c /= c[-1]
    return np.interp(q, c, x[o])


class Calibrator:
    def __init__(self, seed: int = 0, n_particles: int = 1200, prior: M.Prior | None = None, n_cand: int = 800):
        self.seed, self.N, self.n_cand = int(seed), int(n_particles), int(n_cand)
        self.pool = S.pool()
        self.prior = prior or M.Prior()
        self.parts = M.sample_prior(self.N, np.random.default_rng([self.seed, 0xCA1]), self.prior)
        self.logw = np.zeros(self.N)
        self.history: list[tuple[int, str, object]] = []   # (stim, mode, response)
        self.shown: set[int] = set()
        self.resamples = 0
        self._refresh()

    @property
    def n(self) -> int:
        return len(self.history)

    # ---- model tables ----
    def _refresh(self):
        P, pl = self.parts, self.pool
        Mx = M.mixture_matrices(P.w, P.sev)
        de, cos2 = M.pair_stats(np.einsum("nij,pj->npi", Mx, pl.c1), np.einsum("nij,pj->npi", Mx, pl.c2))
        self.deff = de * M.gain(cos2, P.s_L[:, None], P.s_C[:, None])      # (N,P)
        self.tau = M.tau_combos(np.exp(P.logk), P.bg)                       # (N,C)

    def weights(self) -> np.ndarray:
        w = np.exp(self.logw - self.logw.max())
        return w / w.sum()

    def ratio(self, stim) -> np.ndarray:
        stim = np.asarray(stim, int)
        return self.deff[:, stim // S.N_COMBO] / self.tau[:, stim % S.N_COMBO]       # (N,len(stim))

    # ---- selection ----
    def eig(self, stim, mode: str) -> np.ndarray:
        w = self.weights()
        r = self.ratio(stim)
        if mode == "same_different":
            p = M.p_diff(r)
            pb = w @ p
            return _H(np.stack([pb, 1 - pb], -1)) - w @ _H(np.stack([p, 1 - p], -1))
        p = M.slider_probs(r)                                   # (N,K,5)
        return _H(np.einsum("n,nkc->kc", w, p)) - np.einsum("n,nk->k", w, _H(p))

    def select(self, mode: str) -> tuple[int, float]:
        rng = np.random.default_rng([self.seed, self.n, 0x5E1])
        pool = np.setdiff1d(np.arange(self.pool.n_stim), np.fromiter(self.shown, int, len(self.shown)))
        cand = rng.choice(pool, min(self.n_cand, len(pool)), replace=False)
        g = self.eig(cand, mode)
        i = int(np.argmax(g))
        return int(cand[i]), float(g[i])

    # ---- update ----
    def log_lik(self, stim: int, mode: str, response) -> np.ndarray:
        r = self.ratio([stim])[:, 0]
        if mode == "same_different":
            p = M.p_diff(r)
            return np.log(np.clip(p if response == "different" else 1 - p, 1e-12, 1))
        j = int(np.argmin(np.abs(M.RATINGS - float(response))))
        return np.log(np.clip(M.slider_probs(r)[:, j], 1e-12, 1))

    def update(self, stim: int, mode: str, response):
        if mode not in MODES:
            raise ValueError(mode)
        if mode == "same_different" and response not in ("same", "different"):
            raise ValueError("response must be 'same' or 'different'")
        if mode == "slider":
            response = M.quantise_slider(float(response))
        self.logw = self.logw + self.log_lik(stim, mode, response)
        self.logw -= self.logw.max()
        self.history.append((int(stim), mode, response))
        self.shown.add(int(stim))
        w = self.weights()
        if 1.0 / np.sum(w ** 2) < 0.5 * self.N:
            self._resample(w)

    def _resample(self, w):
        rng = np.random.default_rng([self.seed, self.n, 0x7E5])
        u = (rng.random() + np.arange(self.N)) / self.N
        idx = np.minimum(np.searchsorted(np.cumsum(w), u), self.N - 1)
        z = M.to_z(self.parts.take(np.arange(self.N)))
        mean = w @ z
        cov = np.cov(z.T, aweights=w) + 1e-6 * np.eye(z.shape[1])
        a = 0.98
        h = np.sqrt(1 - a * a)
        L = np.linalg.cholesky(cov)
        zn = a * z[idx] + (1 - a) * mean + h * rng.standard_normal(z.shape) @ L.T
        self.parts = M.from_z(zn)
        self.logw = np.zeros(self.N)
        self.resamples += 1
        self._refresh()

    # ---- outputs ----
    def samples(self, n: int = 500) -> dict[str, list[float]]:
        """Equal-weight posterior samples (systematic resample, deterministic)."""
        w = self.weights()
        rng = np.random.default_rng([self.seed, self.n, 0x5A3])
        idx = np.minimum(np.searchsorted(np.cumsum(w), (rng.random() + np.arange(n)) / n), self.N - 1)
        rng.shuffle(idx)
        P = self.parts.take(idx)
        t = P.table()
        t.update(w_protan=P.w[:, 0], w_deutan=P.w[:, 1], w_tritan=P.w[:, 2], logk=P.logk, rho=P.rho)
        return {k: [float(x) for x in v] for k, v in t.items()}

    def stats(self) -> dict:
        w = self.weights()
        t = self.parts.table()
        out = {}
        for key, v in t.items():
            out[key] = dict(mean=float(w @ v), lo=float(wquantile(v, w, 0.05)), hi=float(wquantile(v, w, 0.95)))
        out["k"]["mean"] = float(np.exp(w @ self.parts.logk))             # geometric mean
        return out

    def width(self) -> float:
        """Max normalised 90% width over the correction-relevant parameters: red-green effective severity
        e_p+e_d (protan/deutan share confusion axes; their split is reported via type_probs, not gated on),
        tritan effective severity e_t, and log k (range = prior range). e_i = s*w_i."""
        w = self.weights()
        e = self.parts.e
        ws = [wquantile(v, w, 0.95) - wquantile(v, w, 0.05) for v in (e[:, 0] + e[:, 1], e[:, 2])]
        ws.append((wquantile(self.parts.logk, w, 0.95) - wquantile(self.parts.logk, w, 0.05)) / M.PRIOR_LOGK_RANGE)
        return float(max(ws))

    def type_probs(self) -> dict[str, float]:
        w = self.weights()
        e = self.parts.e
        top = e.argmax(1)
        normal = e.max(1) < 0.1
        out = {"normal": float(w[normal].sum())}
        for i, kd in enumerate(S.KINDS):
            out[kd] = float(w[(top == i) & ~normal].sum())
        return out

    def discrimination_map(self, n: int = 300) -> list[dict]:
        """Per (axis, lightness): smallest modulation delta (relative cone contrast / luminance ratio) the observer can
        tell apart at 2 deg on a 0.2 background, with 90% credible interval. None = not resolvable up to delta=1."""
        w = self.weights()
        rng = np.random.default_rng([self.seed, self.n, 0x3AB])
        idx = rng.choice(self.N, n, p=w)
        P = self.parts.take(idx)
        Mx = M.mixture_matrices(P.w, P.sev)
        tau = M.tau_combos(np.exp(P.logk), P.bg)[:, S.REF_COMBO]
        out = []
        for a, axis in enumerate(S.AXES):
            for l, y in enumerate(S.LEVELS_Y):
                b = S.base_colour(y, S.TINTS["gray"])
                c1, c2 = S.make_pair(axis, np.broadcast_to(b, (len(MAP_DELTAS), 3)), MAP_DELTAS)
                de, cos2 = M.pair_stats(np.einsum("nij,pj->npi", Mx, c1), np.einsum("nij,pj->npi", Mx, c2))
                r = de * M.gain(cos2, P.s_L[:, None], P.s_C[:, None]) / tau[:, None]      # (n,G)
                th = np.full(n, np.inf)
                for i in range(n):
                    hit = np.flatnonzero(r[i] >= 1)
                    if len(hit):
                        j = hit[0]
                        if j == 0:
                            th[i] = MAP_DELTAS[0]
                        else:
                            f = (1 - r[i, j - 1]) / max(r[i, j] - r[i, j - 1], 1e-12)
                            th[i] = float(np.exp(np.log(MAP_DELTAS[j - 1]) + f * (np.log(MAP_DELTAS[j]) - np.log(MAP_DELTAS[j - 1]))))
                fin = np.isfinite(th)
                q = lambda p: None if not np.isfinite(np.quantile(th, p, method="nearest")) else float(np.quantile(th, p, method="nearest"))
                out.append(dict(axis=axis, Y=y, delta50=q(0.5), lo=q(0.05), hi=q(0.95), resolvable=float(fin.mean())))
        return out

    def summary(self, full: bool = False) -> dict:
        st = self.stats()
        e = [st["e_protan"]["mean"], st["e_deutan"]["mean"], st["e_tritan"]["mean"]]
        s = dict(n_trials=self.n, width=self.width(), params=st, type_probs=self.type_probs(),
                 dominant=S.KINDS[int(np.argmax(e))], effective_severity=e)
        if full:
            s["discrimination_map"] = self.discrimination_map()
        return s
