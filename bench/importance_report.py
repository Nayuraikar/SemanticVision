"""python -m bench.importance_report: run importance_eval (held-out scenes), keep the MLP ONLY if it beats the rules on both
metrics (paired t and Wilcoxon p < .05, positive diff). Keep -> train on pooled train scenes, save core/importance/mlp.npz;
otherwise delete any stale model and report 'keep rules'. Writes bench/results/importance_decision.json."""
from __future__ import annotations
import json
from pathlib import Path
from . import runner
from .experiments.importance_eval import scenes, stack, TRAIN
from core.importance import train_mlp, MODEL_PATH


def main(n_seeds=12):
    res = runner.run(runner.load_config("importance_eval"), seed=0, n_seeds=n_seeds)
    rows = {(p["arm"], p["metric"]): p for p in res["paired"]}
    ok = all(rows[("mlp", m)]["mean_diff"] > 0 and rows[("mlp", m)]["t_p"] < .05 and rows[("mlp", m)]["wilcoxon_p"] < .05
             for m in ("spearman", "top20"))
    dec = dict(keep_mlp=bool(ok), n_seeds=n_seeds, paired={f"{a}/{m}": rows[(a, m)] for a, m in rows if a == "mlp"})
    if ok:
        X, y = stack(scenes(TRAIN, 40, 424242))
        train_mlp(X, y, 0).save()
    elif MODEL_PATH.exists():
        MODEL_PATH.unlink()
    Path(res["out"], "importance_decision.json").write_text(json.dumps(dec, indent=1))
    print(json.dumps({"keep_mlp": ok, "paired": dec["paired"]}, indent=1))
    return dec


if __name__ == "__main__":
    main()
