"""ONNX export with a dynamic object count N, and ONNX Runtime vs PyTorch verification."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from core.learned import F_DIM, Z_DIM
from .model import Exportable

INPUTS = ("x", "z", "labg", "delta", "S", "A", "mask")
TOL_LAB = 0.05      # max |Lab| difference accepted (float32 + bisection boundary flips)


def dummy(n=7, f=F_DIM, zd=Z_DIM, seed=0):
    g = torch.Generator().manual_seed(seed)
    lab = torch.stack([torch.rand(n, generator=g) * 60 + 20, torch.randn(n, generator=g) * 10,
                       torch.randn(n, generator=g) * 10], -1)
    S = torch.eye(n)
    return (torch.randn(1, n, f, generator=g) * 0.3, torch.rand(1, zd, generator=g), lab[None], torch.full((1, n), 15.0),
            S[None], (torch.rand(1, n, n, generator=g) > 0.5).float(), torch.ones(1, n))


def export_onnx(net, path, opset=17):
    path = Path(path)
    net = net.cpu().eval()
    dyn = {"x": {1: "N"}, "labg": {1: "N"}, "delta": {1: "N"}, "S": {1: "N", 2: "N"}, "A": {1: "N", 2: "N"},
           "mask": {1: "N"}, "lab": {1: "N"}}
    with torch.no_grad():
        torch.onnx.export(Exportable(net), dummy(), str(path), input_names=list(INPUTS), output_names=["lab"],
                          dynamic_axes=dyn, opset_version=opset, dynamo=False)
    return path


def feeds(inp):
    """core.learned.scene_inputs dict -> ONNX feeds (batch 1)."""
    n = len(inp["x"])
    return dict(x=inp["x"][None], z=inp["z"][None], labg=inp["labg"][None], delta=inp["delta"][None], S=inp["S"][None],
                A=inp["A"][None], mask=np.ones((1, n), np.float32))


def verify_onnx(net, path, samples, tol=TOL_LAB):
    """Run torch and ORT on each sample (different N). -> dict(max_abs, mean_abs, ok, sizes)."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]          # unused inputs (A for mlp/deepsets) are pruned at export
    net = net.cpu().eval()
    mx, mean, sizes = 0.0, [], []
    for s in samples:
        f = feeds(s["inp"] if "inp" in s else s)
        with torch.no_grad():
            ref = net(*[torch.as_tensor(f[k]) for k in INPUTS])[0].numpy()
        got = sess.run(["lab"], {k: f[k] for k in names})[0]
        d = np.abs(ref - got)
        mx = max(mx, float(d.max())); mean.append(float(d.mean())); sizes.append(int(f["x"].shape[1]))
    return dict(max_abs=mx, mean_abs=float(np.mean(mean)), ok=bool(mx <= tol), sizes=sizes)
