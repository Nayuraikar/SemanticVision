"""Contextual bandit for the learned benefit per (object features, tier): one LinUCB arm per tier >= 1 (tier 0 benefit = 0).
estimate(X) is optimistic (UCB) so rarely-tried tiers get explored; update(x, tier, reward) is a rank-1 ridge update."""
from __future__ import annotations
import numpy as np


class LinUCB:
    def __init__(self, d, n_tiers=5, alpha=0.5, lam=1.0, decay=1.0, seed=0):
        self.d, self.T, self.alpha, self.decay = d, n_tiers, alpha, decay
        self.Ainv = np.stack([np.eye(d) / lam for _ in range(n_tiers)])
        self.bvec = np.zeros((n_tiers, d))
        self.n = np.zeros(n_tiers, int)

    @property
    def theta(self):
        return np.einsum("tij,tj->ti", self.Ainv, self.bvec)

    def estimate(self, X, explore=True):
        """X (n,d) -> (n,T) predicted benefit (>=0); column 0 = 0."""
        X = np.asarray(X, float)
        mu = X @ self.theta.T
        if explore and self.alpha > 0:
            mu = mu + self.alpha * np.sqrt(np.maximum(np.einsum("ni,tij,nj->nt", X, self.Ainv, X), 0))
        mu = np.maximum(mu, 0.0)
        mu[:, 0] = 0.0
        return mu

    def update(self, x, tier, reward):
        if tier <= 0:
            return
        x = np.asarray(x, float)
        A = self.Ainv[tier]
        Ax = A @ x
        self.Ainv[tier] = A - np.outer(Ax, Ax) / (1.0 + x @ Ax)
        self.bvec[tier] += reward * x
        self.n[tier] += 1

    def update_batch(self, X, tiers, rewards):
        for x, t, r in zip(X, tiers, rewards):
            self.update(x, int(t), float(r))
        self.alpha *= self.decay
