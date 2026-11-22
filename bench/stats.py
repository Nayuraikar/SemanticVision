"""Aggregation: mean, std, t-based CI; paired tests (scipy)."""
from __future__ import annotations
import numpy as np
from scipy import stats as st


def summarize(x, conf=0.95):
    x = np.asarray(x, float)
    n = len(x)
    m = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    h = float(st.t.ppf(0.5 + conf / 2, n - 1) * sd / np.sqrt(n)) if n > 1 else float("nan")
    return dict(n=n, mean=m, std=sd, ci_lo=m - h if n > 1 else float("nan"), ci_hi=m + h if n > 1 else float("nan"))


def paired(a, b):
    """a, b aligned by seed. Returns mean diff (a-b), paired t p, Wilcoxon p, Cohen's d_z."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    out = dict(n=len(d), mean_diff=float(d.mean()), t_p=float("nan"), wilcoxon_p=float("nan"), dz=float("nan"))
    if len(d) < 2:
        return out
    sd = d.std(ddof=1)
    if sd > 0:
        out["t_p"] = float(st.ttest_rel(a, b).pvalue)
        out["dz"] = float(d.mean() / sd)
    else:
        out["t_p"] = 0.0 if d.mean() != 0 else 1.0
    if len(d) >= 5 and np.any(d != 0):
        out["wilcoxon_p"] = float(st.wilcoxon(a, b).pvalue)
    return out
