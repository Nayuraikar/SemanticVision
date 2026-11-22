"""Config-driven multi-seed runner. Config JSON:
{"name": str, "experiment": str, "params": {...}, "n_seeds": int, "baseline": arm (optional)}"""
from __future__ import annotations
import csv, json
from pathlib import Path
from . import registry, stats

ROOT = Path(__file__).resolve().parent


def load_config(path) -> dict:
    p = Path(path)
    if not p.exists() and (ROOT / "configs" / p).exists():
        p = ROOT / "configs" / p
    if not p.exists() and (ROOT / "configs" / (str(path) + ".json")).exists():
        p = ROOT / "configs" / (str(path) + ".json")
    cfg = json.loads(p.read_text())
    cfg.setdefault("name", p.stem)
    cfg.setdefault("params", {})
    cfg.setdefault("n_seeds", 1)
    if "experiment" not in cfg:
        raise ValueError("config needs 'experiment'")
    return cfg


def run(cfg: dict, seed: int = 0, n_seeds: int | None = None, out_dir=None) -> dict:
    fn = registry.get(cfg["experiment"])
    n = int(n_seeds or cfg["n_seeds"])
    seeds = [seed + i for i in range(n)]
    rows = []
    for s in seeds:
        res = fn(cfg["params"], s)
        for arm, mets in res.items():
            for m, v in mets.items():
                rows.append(dict(seed=s, arm=arm, metric=m, value=float(v)))
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    metrics = list(dict.fromkeys(r["metric"] for r in rows))
    get = lambda a, m: [r["value"] for r in sorted(rows, key=lambda r: r["seed"]) if r["arm"] == a and r["metric"] == m]
    summary = [dict(arm=a, metric=m, **stats.summarize(get(a, m))) for a in arms for m in metrics if get(a, m)]
    base = cfg.get("baseline", arms[0] if arms else None)
    paired = []
    for a in arms:
        if a == base:
            continue
        for m in metrics:
            x, y = get(a, m), get(base, m)
            if x and len(x) == len(y):
                paired.append(dict(arm=a, baseline=base, metric=m, **stats.paired(x, y)))
    out = Path(out_dir) if out_dir else ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)
    for suffix, data in (("raw", rows), ("summary", summary), ("paired", paired)):
        if not data:
            continue
        with open(out / f"{cfg['name']}_{suffix}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    return dict(rows=rows, summary=summary, paired=paired, seeds=seeds, out=str(out))
