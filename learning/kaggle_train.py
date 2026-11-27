"""Kaggle-ready single-file entry point for the learned solver (SPEC 5.6). Upload the repository as a Kaggle dataset, then:
    !python /kaggle/input/semanticvision/learning/kaggle_train.py --repo /kaggle/input/semanticvision --out /kaggle/working
Builds the large seeded dataset (exact solver, DE reference on a subset; all CPU cores), trains each requested encoder (CUDA if
available), keeps the best val checkpoint, exports ONNX (dynamic N), verifies ONNX vs PyTorch, and writes per arch
<out>/sv_learned_<arch>.{pt,onnx,metrics.json} plus <out>/kaggle_metrics.json (val metrics only; held-out splits are scored by
`python -m bench.learned_report --model <onnx>` back in the repo, never during training).
A read-only repo is fine: the dataset cache goes to <out>/cache. --data <pkl> reuses a dataset built elsewhere
(python -m learning.data --scale kaggle)."""
from __future__ import annotations
import argparse, json, os, pickle, sys, time
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kaggle_train.py")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--out", default="/kaggle/working" if Path("/kaggle/working").exists() else "kaggle_out")
    ap.add_argument("--scale", default="kaggle")
    ap.add_argument("--data", default=None, help="prebuilt dataset pickle (skips generation)")
    ap.add_argument("--archs", default="deepsets,gnn,mlp")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--device", default="auto", help="auto = cuda > mps > cpu")
    ap.add_argument("--time-limit", type=float, default=None, help="seconds per arch")
    a = ap.parse_args(argv)
    sys.path.insert(0, a.repo)
    import torch
    from learning import data as D
    from learning.model import NetConfig
    from learning.train import TrainConfig, train, evaluate, save

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.data:
        data = pickle.loads(Path(a.data).read_bytes())
    else:
        D.CACHE = out / "cache"
        data = D.build(a.scale, procs=a.procs)
    print({k: len(v) for k, v in data.items()}, f"data {time.time() - t0:.0f}s", flush=True)
    dev = a.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    report = dict(scale=a.scale, device=dev, data_seconds=time.time() - t0, archs={})
    for arch in a.archs.split(","):
        tc = TrainConfig(epochs=a.epochs, batch=a.batch, lr=a.lr, seed=a.seed,
                         net=NetConfig(arch=arch, hidden=a.hidden, layers=a.layers))
        net, info = train(data, tc, dev, time_limit_s=a.time_limit)
        info["final"] = {"val": evaluate(net, data["val"], tc)}
        info["verify_samples"] = data["val"][:16]
        meta = save(net, tc, info, f"sv_learned_{arch}", out)
        report["archs"][arch] = dict(val=info["final"]["val"], best_epoch=info["best_epoch"], seconds=info["seconds"],
                                     n_params=meta["n_params"], onnx=meta.get("onnx"))
        print(arch, json.dumps(report["archs"][arch], default=float), flush=True)
    report["seconds"] = time.time() - t0
    (out / "kaggle_metrics.json").write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
