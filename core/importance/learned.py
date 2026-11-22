"""Learned importance: small MLP trained on task-observer labels. Kept only if it beats the rules on held-out scenes
(decided by bench/importance_report.py; weights saved to mlp.npz only then). Runtime forward is numpy (no torch)."""
from __future__ import annotations
from pathlib import Path
import numpy as np

MODEL_PATH = Path(__file__).with_name("mlp.npz")


class NumpyMLP:
    def __init__(self, mean, std, layers, cols=None):
        self.mean, self.std, self.layers = mean, std, layers          # layers: [(W,b),...]
        self.cols = cols

    def __call__(self, X):
        X = np.asarray(X, float)
        if self.cols is not None:
            X = X[:, self.cols]
        h = (X - self.mean) / self.std
        for i, (W, b) in enumerate(self.layers):
            h = h @ W + b
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0)
        return np.clip(h[:, 0], 0.0, 1.0)

    def save(self, path=MODEL_PATH):
        d = dict(mean=self.mean, std=self.std, n=len(self.layers))
        for i, (W, b) in enumerate(self.layers):
            d[f"W{i}"], d[f"b{i}"] = W, b
        if self.cols is not None:
            d["cols"] = np.asarray(self.cols)
        np.savez(path, **d)

    @classmethod
    def load(cls, path=MODEL_PATH):
        z = np.load(path)
        return cls(z["mean"], z["std"], [(z[f"W{i}"], z[f"b{i}"]) for i in range(int(z["n"]))],
                   z["cols"] if "cols" in z else None)


def train(X, y, seed=0, cols=None, hidden=32, epochs=400, lr=3e-3, wd=1e-4, val_frac=0.15) -> NumpyMLP:
    import torch
    torch.set_num_threads(1)
    X = np.asarray(X, float)
    X = X if cols is None else X[:, cols]
    rng = np.random.default_rng([seed, 0x31D])
    idx = rng.permutation(len(X))
    nv = max(1, int(val_frac * len(X)))
    va, tr = idx[:nv], idx[nv:]
    mean, std = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xt = torch.tensor((X - mean) / std, dtype=torch.float32)
    yt = torch.tensor(np.asarray(y, float), dtype=torch.float32)[:, None]
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden),
                              torch.nn.ReLU(), torch.nn.Linear(hidden, 1))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    best, state = np.inf, None
    for ep in range(epochs):
        opt.zero_grad()
        loss = ((net(Xt[tr]) - yt[tr]) ** 2).mean()
        loss.backward(); opt.step()
        if ep % 5 == 4:
            with torch.no_grad():
                v = float(((net(Xt[va]) - yt[va]) ** 2).mean())
            if v < best:
                best, state = v, [p.detach().numpy().copy() for p in net.parameters()]
    layers = [(state[i].T.astype(float), state[i + 1].astype(float)) for i in range(0, len(state), 2)]
    return NumpyMLP(mean, std, layers, None if cols is None else np.asarray(cols))
