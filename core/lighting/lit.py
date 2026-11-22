"""Lit colour: reflected (albedo * irradiance) and emissive added separately, shadow classification."""
from __future__ import annotations
import numpy as np
from .field import LightField

LIT_T, SHADOW_T = 0.95, 0.05


def shadow_state(vis):
    """shadow_vis -> 'lit' | 'penumbra' | 'shadow'; unknown (None) counts as lit."""
    if vis is None or vis >= LIT_T:
        return "lit"
    return "shadow" if vis <= SHADOW_T else "penumbra"


def lit_components(albedo, emissive, E, exposure=1.0):
    """-> (reflected, emitted) linear HDR. Emission never scales with irradiance (only with camera exposure)."""
    return np.asarray(albedo, float) * E * exposure, np.asarray(emissive, float) * exposure


def lit_color(albedo, emissive, E, exposure=1.0):
    r, e = lit_components(albedo, emissive, E, exposure)
    return np.clip(r + e, 0.0, 1.0)


# ablation rungs (K3): which terms the predictor uses
RUNGS = ("albedo", "sh2", "sh2_shadow", "sh2_cat16", "sh2_full")


def rung_colors(field: LightField, objs, rung: str):
    """Predicted display colours (N,3) for SceneObjects under an ablation rung. cat16 is applied later (perception
    pipeline, illum_xyz), so 'sh2_cat16' returns the same colours as 'sh2' and 'sh2_full' (= shadow + CAT16) the same as 'sh2_shadow'; use rung_illum() for the white point."""
    out = []
    for o in objs:
        alb, emi = np.array(o.albedo_rgb, float), np.array(o.emissive_rgb, float)
        if rung == "albedo":
            out.append(np.clip(alb + emi, 0, 1)); continue
        n = None if o.normal is None else np.array(o.normal, float)
        if rung in ("sh2_shadow", "sh2_full"):
            p = None if o.position is None else np.array(o.position, float)
            E = field.irradiance(n, p, 1.0 if o.shadow_vis is None else o.shadow_vis)
        else:
            E = field.irradiance(n)
        out.append(lit_color(alb, emi, E, field.exposure))
    return np.array(out).reshape(-1, 3)


def rung_illum(field: LightField, rung: str):
    return field.white_xyz() if rung in ("sh2_cat16", "sh2_full") else None


def object_lit(field: LightField, o, shadow=True):
    """Canonical lit colour of a SceneObject: (display, reflected, emitted, E)."""
    n = None if o.normal is None else np.array(o.normal, float)
    p = None if o.position is None else np.array(o.position, float)
    E = field.irradiance(n, p, o.shadow_vis) if shadow and (o.shadow_vis is not None or (p is not None and field.dom is not None and field.dom.position is not None)) else field.irradiance(n)
    r, e = lit_components(o.albedo_rgb, o.emissive_rgb, E, field.exposure)
    return np.clip(r + e, 0, 1), r, e, E
