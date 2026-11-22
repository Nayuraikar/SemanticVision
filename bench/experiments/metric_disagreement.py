"""CIEDE2000 vs CAM16-UCS on emissive small-field cases (SPEC 5.1). Pairs of adjacent objects in held-out rendered scenes
(rendered colour, CAT16 to the rendered white, held-out observers' CVD). Each metric uses its own tau (dE00 tau; CAM16-UCS tau
= tau * calibrated scale). A pair 'disagrees' when exactly one metric calls it confusable. Reflective large-field pairs are the control."""
from __future__ import annotations
import numpy as np
from bench.registry import experiment
from bench.k3 import (geometry, truth_colors, truth_illum, world_seed, cam16_tau_scale, confusable_pairs)
from adapters.synthetic import WorldConfig, generate_world, LIGHT_PRESETS
from core.observers import heldout_observers

EMISSIVE_CLS = ("light", "indicator")
SIZE_BINS = (0.0, 0.5, 1.0, 2.0, 1e9)       # deg, smaller field of the pair
LUM_BINS = (0.0, 0.05, 0.2, 0.5, 1e9)       # relative luminance Y, pair mean (display, clipped)
YL = np.array([0.2126, 0.7152, 0.0722])


def _bin(v, edges):
    i = int(np.searchsorted(edges, v, side="right") - 1)
    return f"{edges[i]:g}-{edges[i + 1]:g}" if edges[i + 1] < 1e8 else f">={edges[i]:g}"


def collect(params, seed):
    """-> list of per-(pair, observer) dicts."""
    scale = cam16_tau_scale()
    lights = params.get("lights", list(LIGHT_PRESETS))
    obs = heldout_observers(int(params.get("n_obs", 6)), seed)
    rows = []
    for light in lights:
        for k in range(int(params.get("n_worlds", 4))):
            cfg = WorldConfig.from_dict({**params.get("world", {}), "lighting": light})
            snap, truth, _ = generate_world(cfg, world_seed(seed, k))
            if len(snap.objects) < 2:
                continue
            geo, tc, ti = geometry(snap, float(params.get("adj_margin", 0.3))), truth_colors(snap, truth), truth_illum(truth)
            em = np.array([o.cls in EMISSIVE_CLS and max(o.emissive_rgb) > 0 for o in snap.objects])
            lum = tc @ YL
            for o in obs:
                pr, d0, t0, c0 = confusable_pairs(tc, geo, o, truth["bg_lum"], ti)
                _, d1, t1, c1 = confusable_pairs(tc, geo, o, truth["bg_lum"], ti, space="cam16", tau_scale=scale)
                for q, (i, j) in enumerate(pr):
                    kind = "emissive" if em[i] and em[j] else ("reflective" if not em[i] and not em[j] else "mixed")
                    sz, ly = float(min(geo["size"][i], geo["size"][j])), float(0.5 * (lum[i] + lum[j]))
                    rows.append(dict(seed=seed, light=light, observer=o.id, kind=kind, a=snap.objects[i].id, b=snap.objects[j].id,
                                     size_deg=sz, lum=ly, size_bin=_bin(sz, SIZE_BINS), lum_bin=_bin(ly, LUM_BINS),
                                     d_de00=float(d0[q]), tau_de00=float(t0[q]), d_cam16=float(d1[q]), tau_cam16=float(t1[q]),
                                     conf_de00=bool(c0[q]), conf_cam16=bool(c1[q]),
                                     disagree=bool(c0[q] != c1[q]), cam16_only=bool(c1[q] and not c0[q]),
                                     de00_only=bool(c0[q] and not c1[q])))
    return rows


@experiment("metric_disagreement")
def run(params, seed):
    rows = collect(params, seed)
    out = {}
    for kind in ("emissive", "reflective", "mixed"):
        r = [x for x in rows if x["kind"] == kind]
        if not r:
            out[kind] = dict(n_pairs=0.0, disagree_rate=0.0, cam16_only_rate=0.0, de00_only_rate=0.0)
            continue
        m = lambda key: float(np.mean([x[key] for x in r]))
        out[kind] = dict(n_pairs=float(len(r)), disagree_rate=m("disagree"), cam16_only_rate=m("cam16_only"),
                         de00_only_rate=m("de00_only"))
    return out
