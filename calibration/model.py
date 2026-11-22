"""Observer parameterisation and trial (response) models.

theta = (w[3] type mixture over protan/deutan/tritan on the simplex, severity s, log k, rho, bg_gain)
  perceived colour = (sum_i w_i M_i(s)) rgb               (Machado matrices, linear sRGB)
  s_L = exp(rho/2), s_C = exp(-rho/2)                      (s_L*s_C = 1: overall scale lives in k only -> identifiable)
  effective difference d = dE00(perceived pair) * g,  g = sqrt(s_L^2 cos^2 + s_C^2 sin^2)   (phi: lightness vs chroma
  direction of the perceived Lab difference);  r = d / tau,  tau = threshold.tau(size, bg, k, bg_gain)
  same/different: P(diff) = gamma + (1-gamma-lapse) (1 - exp(-r^beta))
  slider (5 levels 0,.25,.5,.75,1): latent N(r/(r+1), sigma), censored/quantised."""
from __future__ import annotations
from dataclasses import dataclass, field, replace
from math import erf
import numpy as np
from core.perception import color as C
from core.perception.cvd import machado_matrix, simulate
from core.perception.threshold import ThresholdParams, tau as tau_fn
from .stimuli import KINDS, SIZES, BGS, N_COMBO

GAMMA, LAPSE, BETA, SLIDER_SIGMA = 0.05, 0.03, 2.0, 0.2
RATINGS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
EDGES = np.array([0.125, 0.375, 0.625, 0.875])
LOGK_SD, RHO_SD, LOG_CLIP = 0.5, 0.5, 1.2
BG_RANGE = (0.05, 0.6)
PRIOR_LOGK_RANGE = 2 * LOG_CLIP
PARAMS = ("e_protan", "e_deutan", "e_tritan", "severity", "k", "s_L", "s_C", "bg_gain")

_ERF = np.vectorize(erf)


def _Phi(x):
    return 0.5 * (1 + _ERF(np.asarray(x) / np.sqrt(2)))


try:  # fast path
    from scipy.special import ndtr as _Phi  # noqa: F811
except Exception:
    pass


# ---------- self-report prior ----------
RG_COLOURS = {"red", "green", "brown", "orange", "pink", "purple"}
TRITAN_COLOURS = {"yellow", "turquoise"}


@dataclass(frozen=True)
class SelfReport:
    """Optional. Only shapes the prior (the posterior can overrule it)."""
    type_guess: str | None = None                  # protan | deutan | tritan | unsure | None
    struggle_colors: tuple[str, ...] = ()

    @staticmethod
    def from_dict(d: dict | None) -> "SelfReport":
        d = d or {}
        return SelfReport(d.get("type_guess"), tuple(d.get("struggle_colors") or ()))


@dataclass(frozen=True)
class Prior:
    alpha: tuple = (0.6, 0.6, 0.6)                 # Dirichlet over type mixture
    sev_ab: tuple = (1.5, 1.5)                     # Beta over severity

    @staticmethod
    def from_self_report(sr: SelfReport | None) -> "Prior":
        if sr is None:
            return Prior()
        a = np.array([0.6, 0.6, 0.6])
        sev = np.array([1.5, 1.5])
        if sr.type_guess in KINDS:
            a[KINDS.index(sr.type_guess)] += 2.0
            sev = np.array([2.0, 1.5])
        cs = {c.lower() for c in sr.struggle_colors}
        n_rg, n_t = len(cs & RG_COLOURS), len(cs & TRITAN_COLOURS)
        a[0] += 0.4 * min(n_rg, 3); a[1] += 0.4 * min(n_rg, 3); a[2] += 0.8 * min(n_t, 2)
        if cs or sr.type_guess in KINDS:
            sev = sev + np.array([0.5, 0.0])
        return Prior(tuple(a), tuple(sev))


# ---------- particles ----------
@dataclass
class Particles:
    w: np.ndarray       # (N,3) simplex
    sev: np.ndarray     # (N,)
    logk: np.ndarray
    rho: np.ndarray
    bg: np.ndarray      # bg_gain

    def __len__(self):
        return len(self.sev)

    def take(self, idx) -> "Particles":
        return Particles(self.w[idx], self.sev[idx], self.logk[idx], self.rho[idx], self.bg[idx])

    @property
    def e(self):
        return self.w * self.sev[:, None]          # effective per-type severity (what correction depends on)

    @property
    def s_L(self):
        return np.exp(self.rho / 2)

    @property
    def s_C(self):
        return np.exp(-self.rho / 2)

    def table(self) -> dict[str, np.ndarray]:
        e = self.e
        return dict(e_protan=e[:, 0], e_deutan=e[:, 1], e_tritan=e[:, 2], severity=self.sev, k=np.exp(self.logk),
                    s_L=self.s_L, s_C=self.s_C, bg_gain=self.bg)


def _trunc_normal(rng, n, sd):
    out = rng.normal(0, sd, n)
    bad = np.abs(out) > LOG_CLIP
    while bad.any():
        out[bad] = rng.normal(0, sd, int(bad.sum()))
        bad = np.abs(out) > LOG_CLIP
    return out


def sample_prior(n: int, rng, prior: Prior | None = None) -> Particles:
    prior = prior or Prior()
    w = rng.dirichlet(prior.alpha, n)
    w = np.clip(w, 1e-3, None)
    w /= w.sum(1, keepdims=True)
    return Particles(w, rng.beta(*prior.sev_ab, n).clip(1e-3, 1.0), _trunc_normal(rng, n, LOGK_SD),
                     _trunc_normal(rng, n, RHO_SD), rng.uniform(*BG_RANGE, n))


def to_z(p: Particles) -> np.ndarray:
    lw = np.log(p.w[:, :2] / p.w[:, 2:3])
    s = np.clip(p.sev, 1e-3, 1 - 1e-3)
    b = (p.bg - BG_RANGE[0]) / (BG_RANGE[1] - BG_RANGE[0])
    b = np.clip(b, 1e-3, 1 - 1e-3)
    return np.column_stack([np.clip(lw, -7, 7), np.log(s / (1 - s)), p.logk, p.rho, np.log(b / (1 - b))])


def from_z(z: np.ndarray) -> Particles:
    ex = np.exp(np.column_stack([z[:, 0], z[:, 1], np.zeros(len(z))]) - np.max(
        np.column_stack([z[:, 0], z[:, 1], np.zeros(len(z))]), axis=1, keepdims=True))
    w = ex / ex.sum(1, keepdims=True)
    sev = 1 / (1 + np.exp(-z[:, 2]))
    b = 1 / (1 + np.exp(-z[:, 5]))
    return Particles(w, sev, np.clip(z[:, 3], -LOG_CLIP, LOG_CLIP), np.clip(z[:, 4], -LOG_CLIP, LOG_CLIP),
                     BG_RANGE[0] + b * (BG_RANGE[1] - BG_RANGE[0]))


# ---------- perception ----------
_SG = np.linspace(0, 1, 21)
_TAB = np.array([[machado_matrix(k, s) for s in _SG] for k in KINDS])      # (3,21,3,3)


def mixture_matrices(w: np.ndarray, sev: np.ndarray) -> np.ndarray:
    """(N,3),(N,) -> (N,3,3). Linear interpolation of the published tables is exact on this grid."""
    x = np.clip(np.asarray(sev, float), 0, 1) * 20
    i = np.minimum(np.floor(x + 1e-9).astype(int), 19)
    f = (x - i)[:, None, None]
    M = np.zeros((len(x), 3, 3))
    for a in range(3):
        M += w[:, a, None, None] * ((1 - f) * _TAB[a, i] + f * _TAB[a, i + 1])
    return M


def pair_stats(p1: np.ndarray, p2: np.ndarray):
    """Perceived linear-RGB pairs (...,3) -> (dE00, cos^2 of the lightness share of the Lab difference)."""
    l1, l2 = C.linear_to_lab(np.clip(p1, 0, 1)), C.linear_to_lab(np.clip(p2, 0, 1))
    de = C.ciede2000(l1, l2)
    dl2 = (l1[..., 0] - l2[..., 0]) ** 2
    dab2 = ((l1[..., 1:] - l2[..., 1:]) ** 2).sum(-1)
    return de, dl2 / np.maximum(dl2 + dab2, 1e-12)


def gain(cos2, s_L, s_C):
    return np.sqrt(s_L ** 2 * cos2 + s_C ** 2 * (1 - cos2))


_SZ = np.repeat(np.array(SIZES), len(BGS))
_BG = np.tile(np.array(BGS), len(SIZES))


def tau_combos(k, bg_gain) -> np.ndarray:
    """k,bg_gain (N,) -> (N, N_COMBO): the same tau(.) the correction graph uses."""
    return tau_fn(_SZ[None, :], _BG[None, :], 1.0, 1.0, np.asarray(k)[:, None],
                  ThresholdParams(bg_gain=np.asarray(bg_gain)[:, None]))


def p_diff(r, gamma=GAMMA, lapse=LAPSE):
    return gamma + (1 - gamma - lapse) * (1 - np.exp(-np.power(np.maximum(r, 0), BETA)))


def slider_probs(r, sigma=SLIDER_SIGMA):
    """r (...) -> (..., 5) probabilities of the quantised slider levels."""
    mu = (np.maximum(r, 0) / (np.maximum(r, 0) + 1))[..., None]
    cdf = _Phi((EDGES - mu) / sigma)
    lo = np.concatenate([np.zeros_like(cdf[..., :1]), cdf], -1)
    hi = np.concatenate([cdf, np.ones_like(cdf[..., :1])], -1)
    return (1 - LAPSE) * (hi - lo) + LAPSE / 5


def quantise_slider(v: float) -> float:
    return float(RATINGS[int(np.clip(round(float(v) * 4), 0, 4))])


# ---------- simulated truth (bench / tests): generated by a model family the estimator does not use ----------
@dataclass(frozen=True)
class SimObserver:
    model: str = "machado"
    kind: str = "deutan"
    severity: float = 1.0
    k: float = 1.0
    s_L: float = 1.0
    s_C: float = 1.0
    bg_gain: float = 0.25
    gamma: float = GAMMA
    lapse: float = LAPSE
    sigma: float = SLIDER_SIGMA

    def canonical(self) -> dict:
        """Identifiable summary: (k, rho) with s_L*s_C = 1."""
        c = float(np.sqrt(self.s_L * self.s_C))
        return dict(logk=float(np.log(self.k / c)), rho=float(np.log(self.s_L / self.s_C)))

    def ratio(self, c1, c2, size_deg, bg_lum) -> float:
        p1 = simulate(np.clip(c1, 0, 1), self.model, self.kind, self.severity) if self.severity > 0 else np.asarray(c1)
        p2 = simulate(np.clip(c2, 0, 1), self.model, self.kind, self.severity) if self.severity > 0 else np.asarray(c2)
        de, cos2 = pair_stats(p1, p2)
        t = tau_fn(size_deg, bg_lum, 1.0, 1.0, self.k, ThresholdParams(bg_gain=self.bg_gain))
        return float(de * gain(cos2, self.s_L, self.s_C) / t)

    def respond(self, mode: str, c1, c2, size_deg, bg_lum, rng):
        r = self.ratio(c1, c2, size_deg, bg_lum)
        if mode == "same_different":
            return "different" if rng.random() < p_diff(r, self.gamma, self.lapse) else "same"
        if rng.random() < self.lapse:
            return float(RATINGS[rng.integers(5)])
        return quantise_slider(float(np.clip(rng.normal(r / (r + 1), self.sigma), 0, 1)))
