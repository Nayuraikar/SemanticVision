"""Training (SPEC 5.6). Loss = separation (soft-min capped dE00 under the sample's TRUE observer and its own tau, core Problem
convention) + naturalness + temporal (output jitter under input perturbation) + constraint penalty (pre-projection violation)
+ supervised (vs exact-solver output). Local M1: MPS if available, else CPU.
python -m learning.train [--scale small] [--arch deepsets] [--epochs 40] [--device auto|cpu|mps] [--name sv_learned]
Writes learning/artifacts/<name>.{pt,onnx,metrics.json}. Model selection uses the `val` split only."""
from __future__ import annotations
import argparse, json, math, time
from dataclasses import dataclass, asdict, field
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as Fn
from core.learned.features import PAD_LAB
from .model import LearnedSolver, NetConfig
from .torch_percept import ciede2000, observer_lab, lab_to_linear

ART = Path(__file__).resolve().parent / "artifacts"


@dataclass
class TrainConfig:
    epochs: int = 40
    batch: int = 32
    lr: float = 2e-3
    wd: float = 1e-5
    seed: int = 0
    w_sep: float = 1.0
    w_nat: float = 0.02          # = core SolverConfig.lam (mean dE00 deviation)
    w_temp: float = 0.05
    w_cons: float = 0.1
    w_sup: float = 1.0
    cap: float = 2.0             # core SolverConfig.cap
    beta: float = 1.0            # core SolverConfig.beta
    jitter: float = 0.02         # temporal: feature noise std
    net: NetConfig = field(default_factory=NetConfig)


def pick_device(name="auto", hidden=96):
    """auto: MPS for wide nets (hidden >= 256), else CPU -- measured on M1: the default net (hidden 96, batch 32) runs
    ~1.4 s/epoch on CPU vs ~5.5 s/epoch on MPS (kernel-launch bound). mps falls back to CPU when unavailable."""
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() and hidden >= 256 else "cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    return torch.device(name)


def collate(samples, device="cpu", with_loss=True):
    B = len(samples)
    N = max(len(s["inp"]["x"]) for s in samples)
    F = samples[0]["inp"]["x"].shape[1]
    P = max([len(s["pairs"]) for s in samples] + [1]) if with_loss else 1
    t = dict(x=np.zeros((B, N, F)), z=np.stack([s["inp"]["z"] for s in samples]), labg=np.tile(PAD_LAB, (B, N, 1)),
             delta=np.ones((B, N)), S=np.tile(np.eye(N), (B, 1, 1)), A=np.zeros((B, N, N)), mask=np.zeros((B, N)),
             target=np.tile(PAD_LAB, (B, N, 1)), pairs=np.zeros((B, P, 2), np.int64), ptau=np.ones((B, P)),
             pw=np.zeros((B, P)), M_cat=np.tile(np.eye(3), (B, 1, 1)), M_cvd=np.tile(np.eye(3), (B, 1, 1)))
    for b, s in enumerate(samples):
        n, d = len(s["inp"]["x"]), s["inp"]
        t["x"][b, :n] = d["x"]; t["labg"][b, :n] = d["labg"]; t["delta"][b, :n] = d["delta"]
        t["S"][b, :n, :n] = d["S"]; t["A"][b, :n, :n] = d["A"]; t["mask"][b, :n] = 1
        if with_loss:
            t["target"][b, :n] = s["target"]
            p = len(s["pairs"])
            t["pairs"][b, :p] = s["pairs"]; t["ptau"][b, :p] = s["ptau"]; t["pw"][b, :p] = s["pw"]
            t["M_cat"][b] = s["M_cat"]
            if s.get("M_cvd") is not None:
                t["M_cvd"][b] = s["M_cvd"]
    out = {k: torch.as_tensor(v, dtype=torch.long if k == "pairs" else torch.float32, device=device) for k, v in t.items()}
    return out


def model_args(b):
    return b["x"], b["z"], b["labg"], b["delta"], b["S"], b["A"], b["mask"]


def softmin_sep(lab, b, cap, beta):
    """Per-sample soft-min of capped perceived dE00 over loss pairs (B,), plus per-pair D (B,P)."""
    pl = observer_lab(lab, b["M_cat"], b["M_cvd"])
    bi = torch.arange(lab.shape[0], device=lab.device)[:, None]
    D = ciede2000(pl[bi, b["pairs"][..., 0]], pl[bi, b["pairs"][..., 1]])
    V = torch.minimum(D, cap * b["ptau"])
    lw = torch.where(b["pw"] > 0, torch.log(torch.clamp(b["pw"], min=1e-12)), torch.full_like(b["pw"], -1e4))
    s = -(torch.logsumexp(-beta * V + lw, -1) - torch.logsumexp(lw, -1)) / beta
    return s, D


def masked_mean(v, m):
    return (v * m).sum() / torch.clamp(m.sum(), min=1.0)


def losses(net, b, tc: TrainConfig, train=True):
    out, raw = net(*model_args(b))
    m = b["mask"]
    has = (b["pw"].sum(-1) > 0).float()
    sep, D = softmin_sep(out, b, tc.cap, tc.beta)
    dev = ciede2000(out, b["labg"])
    L = dict(sep=-(sep * has).sum() / torch.clamp(has.sum(), min=1.0), nat=masked_mean(dev, m))
    L["sup"] = masked_mean(Fn.smooth_l1_loss(out / 10, b["target"] / 10, reduction="none").sum(-1), m)
    rgb = lab_to_linear(raw)
    L["cons"] = masked_mean(torch.relu(ciede2000(raw, b["labg"]) - b["delta"]) / b["delta"]
                            + (torch.relu(rgb - 1) + torch.relu(-rgb)).sum(-1), m)
    if train and tc.w_temp > 0:
        x2 = b["x"] + tc.jitter * torch.randn_like(b["x"]) * m[..., None]
        out2, _ = net(x2, *model_args(b)[1:])
        L["temp"] = masked_mean(ciede2000(out2, out), m)
    else:
        L["temp"] = torch.zeros((), device=out.device)
    L["total"] = (tc.w_sep * L["sep"] + tc.w_nat * L["nat"] + tc.w_temp * L["temp"] + tc.w_cons * L["cons"]
                  + tc.w_sup * L["sup"])
    return L, out, sep, D


@torch.no_grad()
def evaluate(net, samples, tc: TrainConfig, device="cpu", bs=64):
    """Torch-side metrics on samples with a matrix observer (train family): mean soft-min sep (model vs exact target),
    success (pair D >= tau share), mean deviation. Held-out-family metrics are computed by bench with core (numpy)."""
    net.eval()
    agg = {k: [] for k in ("sep", "sep_exact", "succ", "succ_exact", "dev", "dev_exact")}
    for i in range(0, len(samples), bs):
        b = collate(samples[i:i + bs], device)
        L, out, sep, D = losses(net, b, tc, train=False)
        sx, Dx = softmin_sep(b["target"], b, tc.cap, tc.beta)
        pm = (b["pw"] > 0).float()
        for k, s, d, o in (("", sep, D, out), ("_exact", sx, Dx, b["target"])):
            agg["sep" + k] += s.cpu().tolist()
            agg["succ" + k] += (((d >= b["ptau"]).float() * pm).sum(-1) / torch.clamp(pm.sum(-1), min=1)).cpu().tolist()
            agg["dev" + k] += ((ciede2000(o, b["labg"]) * b["mask"]).sum(-1) / b["mask"].sum(-1)).cpu().tolist()
    r = {k: float(np.mean(v)) for k, v in agg.items()}
    r["gap"] = r["sep_exact"] - r["sep"]
    r["score"] = r["sep"] - tc.w_nat * r["dev"]
    return r


def seed_all(seed):
    torch.manual_seed(seed); np.random.seed(seed)


def train(data, tc: TrainConfig = TrainConfig(), device="cpu", verbose=True, time_limit_s=None):
    """-> (net on cpu, history). Keeps the checkpoint with the best val score (val split only)."""
    seed_all(tc.seed)
    device = torch.device(device)
    net = LearnedSolver(tc.net).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=tc.lr, weight_decay=tc.wd)
    tr, va = data["train"], data["val"]
    steps = tc.epochs * math.ceil(len(tr) / tc.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tc.lr, total_steps=max(steps, 1), pct_start=0.1)
    rng = np.random.default_rng(tc.seed)
    hist, best, t0 = [], (-np.inf, None, -1), time.time()
    for ep in range(tc.epochs):
        net.train()
        perm = rng.permutation(len(tr))
        acc = {}
        for i in range(0, len(tr), tc.batch):
            b = collate([tr[j] for j in perm[i:i + tc.batch]], device)
            L, *_ = losses(net, b, tc)
            opt.zero_grad(); L["total"].backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step(); sched.step()
            for k, v in L.items():
                acc.setdefault(k, []).append(float(v.detach()))
        ev = evaluate(net, va, tc, device)
        row = dict(epoch=ep, t=time.time() - t0, **{f"train_{k}": float(np.mean(v)) for k, v in acc.items()},
                   **{f"val_{k}": v for k, v in ev.items()})
        hist.append(row)
        if ev["score"] > best[0]:
            best = (ev["score"], {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, ep)
        if verbose:
            print(f"ep {ep:3d} {row['t']:6.0f}s loss {row['train_total']:.3f} sep {row['train_sep']:.2f} "
                  f"val sep {ev['sep']:.2f}/{ev['sep_exact']:.2f} succ {ev['succ']:.3f}/{ev['succ_exact']:.3f}", flush=True)
        if time_limit_s and time.time() - t0 > time_limit_s:
            break
    net = net.cpu()
    net.load_state_dict(best[1])
    return net, dict(history=hist, best_epoch=best[2], best_score=best[0], seconds=time.time() - t0)


def save(net, tc: TrainConfig, info: dict, name="sv_learned", out=ART, onnx=True):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=net.state_dict(), net=asdict(tc.net), train=asdict(tc), info=info), out / f"{name}.pt")
    meta = dict(name=name, train=asdict(tc), best_epoch=info.get("best_epoch"), seconds=info.get("seconds"),
                n_params=sum(p.numel() for p in net.parameters()))
    if onnx:
        from .export import export_onnx, verify_onnx
        path = export_onnx(net, out / f"{name}.onnx")
        meta["onnx"] = dict(path=str(path.name), bytes=path.stat().st_size, verify=info.get("verify_samples") and
                            verify_onnx(net, path, info["verify_samples"]))
    meta["final"] = info.get("final", {})
    (out / f"{name}.metrics.json").write_text(json.dumps(meta, indent=1, default=float))
    return meta


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = LearnedSolver(NetConfig(**ck["net"]))
    net.load_state_dict(ck["state"]); net.eval()
    return net, ck


def main(argv=None):
    from . import data as D
    ap = argparse.ArgumentParser(prog="python -m learning.train")
    ap.add_argument("--scale", default="small")
    ap.add_argument("--arch", default="deepsets", choices=["mlp", "deepsets", "gnn"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--name", default="sv_learned")
    ap.add_argument("--out", default=str(ART))
    ap.add_argument("--procs", type=int, default=None)
    ap.add_argument("--time-limit", type=float, default=None)
    ap.add_argument("--deploy", action="store_true", help="copy the ONNX model to core/allocator/learned.onnx (tier 2)")
    a = ap.parse_args(argv)
    data = D.build(a.scale, procs=a.procs)
    tc = TrainConfig(epochs=a.epochs, batch=a.batch, seed=a.seed, net=NetConfig(arch=a.arch, hidden=a.hidden))
    dev = pick_device(a.device, a.hidden)
    print("device", dev, "| train", len(data["train"]), "val", len(data["val"]), flush=True)
    net, info = train(data, tc, dev, time_limit_s=a.time_limit)
    info["final"] = {s: evaluate(net, data[s][:200], tc) for s in ("train", "val")}   # held-out: bench.learned_report only
    info["verify_samples"] = data["val"][:12]
    meta = save(net, tc, info, a.name, a.out)
    print(json.dumps({k: meta[k] for k in ("n_params", "best_epoch", "seconds")} | dict(onnx=meta.get("onnx")), default=str))
    if a.deploy:
        import shutil
        from core.allocator.onnx_tier import DEFAULT_MODEL
        shutil.copy(Path(a.out) / f"{a.name}.onnx", DEFAULT_MODEL)
        shutil.copy(Path(a.out) / f"{a.name}.metrics.json", DEFAULT_MODEL.with_suffix(".metrics.json"))
        print("deployed ->", DEFAULT_MODEL)


if __name__ == "__main__":
    main()
