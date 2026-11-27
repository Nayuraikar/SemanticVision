"""Observer-conditioned learned solver (SPEC 5.6): per-object encoder (MLP | DeepSets | small GNN) with FiLM conditioning
on z(theta), per-object DeltaLab residual, shared across a palette group via S, then a differentiable projection onto the
constraint set (deviation <= delta (delta_mem for memory colours), L* range, sRGB gamut). Variable object count N.
Inputs (batch-first, N padded with mask=0 when batching): x (B,N,F), z (B,Z), labg (B,N,3), delta (B,N), S (B,N,N),
A (B,N,N), mask (B,N)."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import torch
from torch import nn
from core.learned import F_DIM, Z_DIM
from .torch_percept import ciede2000, lab_to_linear

L_RANGE = (8.0, 95.0)
GAMUT_TOL = 1e-4


def _feasible(X, lab0, delta, tol=0.0):
    r = lab_to_linear(X)
    ok = ((r >= -GAMUT_TOL) & (r <= 1 + GAMUT_TOL)).all(-1)
    ok = ok & (X[..., 0] >= L_RANGE[0] - 1e-6) & (X[..., 0] <= L_RANGE[1] + 1e-6)
    return ok & (ciede2000(X, lab0) <= delta - tol)


def project(X, lab0, delta, iters: int = 14):
    """(…,N,3) Lab -> feasible Lab. 1) chroma bisection into the sRGB gamut at fixed L (core Problem._gamut), 2) clamp L,
    3) bisection on t along lab0 -> X keeping the largest tested t with ALL constraints satisfied. Since t=0 (= lab0) is the
    lower end and only feasible t are kept, the output is feasible whenever lab0 is (a group's own colour). Bisection
    factors are detached (straight-through on X)."""
    L, a, b = X.unbind(-1)
    lo = torch.zeros_like(L); hi = torch.ones_like(L)
    inside = lambda Y: ((lambda r: ((r >= -GAMUT_TOL) & (r <= 1 + GAMUT_TOL)).all(-1))(lab_to_linear(Y)))
    ok1 = inside(X)
    with torch.no_grad():
        for _ in range(iters):
            m = 0.5 * (lo + hi)
            ok = inside(torch.stack([L, a * m, b * m], -1))
            lo = torch.where(ok, m, lo); hi = torch.where(ok, hi, m)
        c = torch.where(ok1, torch.ones_like(lo), lo)
    X = torch.stack([torch.clamp(L, L_RANGE[0], L_RANGE[1]), a * c, b * c], -1)
    d = X - lab0
    okf = _feasible(X, lab0, delta)
    with torch.no_grad():
        lo = torch.zeros_like(L); hi = torch.ones_like(L)
        for _ in range(iters):
            m = 0.5 * (lo + hi)
            ok = _feasible(lab0 + m[..., None] * d, lab0, delta)
            lo = torch.where(ok, m, lo); hi = torch.where(ok, hi, m)
        t = torch.where(okf, torch.ones_like(lo), lo)
    return lab0 + t[..., None] * d


@dataclass
class NetConfig:
    arch: str = "deepsets"       # mlp | deepsets | gnn
    hidden: int = 96
    layers: int = 3
    res_scale: float = 1.25      # max |residual| per axis, in units of the entry's delta


class FiLMBlock(nn.Module):
    def __init__(self, h, arch):
        super().__init__()
        self.arch = arch
        self.lin = nn.Linear(h, h)
        self.ctx = nn.Linear(2 * h, h) if arch == "deepsets" else nn.Linear(h, h) if arch == "gnn" else None
        self.norm = nn.LayerNorm(h)

    def forward(self, h, gamma, beta, A, mask):
        u = self.lin(h)
        m = mask[..., None]
        if self.arch == "deepsets":
            cnt = torch.clamp(m.sum(1, keepdim=True), min=1.0)
            mean = (h * m).sum(1, keepdim=True) / cnt
            mx = torch.where(m > 0, h, torch.full_like(h, -1e4)).max(1, keepdim=True)[0]
            u = u + self.ctx(torch.cat([mean, mx], -1))
        elif self.arch == "gnn":
            deg = torch.clamp(A.sum(-1, keepdim=True), min=1.0)
            u = u + self.ctx((A @ h) / deg)
        u = self.norm(u) * (1 + gamma[:, None]) + beta[:, None]
        return (h + torch.relu(u)) * m


class LearnedSolver(nn.Module):
    def __init__(self, cfg: NetConfig = NetConfig(), f_dim: int = F_DIM, z_dim: int = Z_DIM):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden
        self.enc = nn.Sequential(nn.Linear(f_dim + z_dim, h), nn.ReLU(), nn.Linear(h, h))
        self.film = nn.Sequential(nn.Linear(z_dim, h), nn.ReLU(), nn.Linear(h, 2 * h * cfg.layers))
        self.blocks = nn.ModuleList(FiLMBlock(h, cfg.arch) for _ in range(cfg.layers))
        self.head = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 3))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def raw(self, x, z, labg, delta, S, A, mask):
        """-> unprojected Lab (B,N,3) = labg + S @ residual."""
        B, N, _ = x.shape
        h = self.enc(torch.cat([x, z[:, None].expand(B, N, z.shape[-1])], -1)) * mask[..., None]
        fl = self.film(z).view(B, self.cfg.layers, 2, -1)
        for i, blk in enumerate(self.blocks):
            h = blk(h, fl[:, i, 0], fl[:, i, 1], A, mask)
        r = torch.tanh(self.head(h)) * (self.cfg.res_scale * delta[..., None])
        return labg + S @ (r * mask[..., None])

    def forward(self, x, z, labg, delta, S, A, mask):
        raw = self.raw(x, z, labg, delta, S, A, mask)
        return project(raw, labg, delta), raw


class Exportable(nn.Module):
    """ONNX wrapper: projected Lab only."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, z, labg, delta, S, A, mask):
        return self.net(x, z, labg, delta, S, A, mask)[0]


def config_dict(cfg: NetConfig):
    return asdict(cfg)
