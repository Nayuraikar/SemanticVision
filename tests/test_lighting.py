import numpy as np
from pathlib import Path
from adapters.synthetic import WorldConfig, LitWorld, generate_world, LIGHT_PRESETS
from core.lighting import (LightField, sh, shadow_state, lit_components, lit_color, object_lit, rung_colors, rung_illum,
                           RUNGS, white_xyz_from_rgb)
from core.perception import color as C, distance as Dst
from protocol import Snapshot, SnapshotBuilder, read_ndjson

FIX = Path(__file__).parent / "fixtures" / "toy_scene.ndjson"


def ad_field(amb=(0.2, 0.2, 0.2), dom=(0.8, 0.8, 0.8), d=(0.3, 1, 0.2), pos=None, exp=1.0):
    b = SnapshotBuilder("o", 8).ambient_dominant(amb, dom, d, exposure=exp, dominant_pos=pos)
    b.add_object("a", "car", (0.5, 0.5, 0.5))
    return LightField.from_lighting(b.build().lighting)


def test_sh_uniform_and_band_limited_exact():
    n = np.random.default_rng(0).normal(size=(20, 3))
    E = sh.irradiance(sh.uniform_coeffs([0.3, 0.2, 0.1]), n)
    assert np.allclose(E, [0.3, 0.2, 0.1], atol=1e-6)
    # gradient sky is band<=1: SH irradiance == numeric cosine integral
    env = lambda d: np.stack([0.5 + 0.4 * d[:, 1], 0.2 + 0 * d[:, 1], 0.3 - 0.1 * d[:, 1]], 1)
    c = sh.project(env, 4096)
    dirs = sh.fibonacci_sphere(20000)
    for nn in n[:5]:
        nn = nn / np.linalg.norm(nn)
        w = np.maximum(dirs @ nn, 0)[:, None]
        num = (env(dirs) * w).sum(0) * 4 * np.pi / len(dirs) / np.pi
        assert np.allclose(sh.irradiance(c, nn), num, atol=2e-3)


def test_dominant_l2_peak_and_shadow_removes_dominant_only():
    f = ad_field()
    n = np.array([0.3, 1, 0.2]); n /= np.linalg.norm(n)
    lit, shade = f.irradiance(n, None, 1.0), f.irradiance(n, None, 0.0)
    assert np.allclose(lit, f.irradiance(n))                       # vis=1 == plain SH
    assert np.allclose(lit[0] - shade[0], 0.8, atol=0.06)          # SH-L2 truncation of cos+ at the peak
    assert shade[0] > 0.15                                         # ambient remains in shadow
    mid = f.irradiance(n, None, 0.5)
    assert shade[0] < mid[0] < lit[0]


def test_point_light_position_and_falloff():
    f = ad_field(pos=(0.0, 10.0, 0.0))
    n = np.array([0.0, 1.0, 0.0])
    near, far = f.irradiance(n, (0, 8, 0), 1.0), f.irradiance(n, (0, 0, 0), 1.0)
    assert near[0] > far[0] > f.irradiance(n, None, 0.0)[0]        # 1/d^2, ambient floor
    side = f.irradiance(np.array([1.0, 0, 0]), (-10, 10, 0), 1.0)  # light is to the +x side of this point
    assert side[0] > f.irradiance(np.array([1.0, 0, 0]), (10, 10, 0), 1.0)[0]


def test_emissive_separate_never_scaled_by_irradiance():
    e = np.array([4.0, 0.2, 0.1])
    r1, m1 = lit_components([0.1, 0.1, 0.1], e, np.array([0.5, 0.5, 0.5]), 1.0)
    r2, m2 = lit_components([0.1, 0.1, 0.1], e, np.array([2.0, 2.0, 2.0]), 1.0)
    assert np.allclose(m1, m2) and np.allclose(m1, e) and np.allclose(r2, 4 * r1)
    assert lit_color([0.1] * 3, e, np.array([0.5] * 3))[0] == 1.0
    assert np.allclose(lit_components([0.1] * 3, e, np.array([1.0] * 3), 2.0)[1], 2 * e)   # exposure only


def test_shadow_classification():
    assert [shadow_state(v) for v in (None, 1.0, 0.96, 0.5, 0.05, 0.0)] == ["lit", "lit", "lit", "penumbra", "shadow", "shadow"]


def test_cat16_moves_illuminant_white_towards_neutral():
    f = LightField.from_lighting(generate_world(WorldConfig(lighting="tinted"), 0)[0].lighting)
    xyz = f.white_xyz()
    assert abs(xyz[1] - 1) < 1e-9
    white = np.clip(f.white_rgb() / f.white_rgb().max(), 0, 1)
    raw = C.linear_to_xyz(white)
    ad = C.linear_to_xyz(np.clip(Dst.perceived(white, illum_xyz=xyz), 0, 1.5))
    xy = lambda v: v[:2] / v.sum()
    d65 = xy(C.D65_XYZ)
    assert np.linalg.norm(xy(ad) - d65) < 0.5 * np.linalg.norm(xy(raw) - d65)


def test_rungs_defined_and_illum_only_for_cat():
    snap, _, _ = generate_world(WorldConfig(lighting="day"), 1)
    f = LightField.from_lighting(snap.lighting)
    assert RUNGS == ("albedo", "sh2", "sh2_shadow", "sh2_cat16", "sh2_full")
    for r in RUNGS:
        assert rung_colors(f, snap.objects, r).shape == (len(snap.objects), 3)
    assert [rung_illum(f, r) is not None for r in RUNGS] == [False, False, False, True, True]
    assert np.allclose(rung_colors(f, snap.objects, "sh2"), rung_colors(f, snap.objects, "sh2_cat16"))


def test_toy_fixture_lit_color_matches_core():
    """TS exporter (adapters/threejs) vs core/lighting: same lit_color from the exported lighting/normal/position/shadow_vis."""
    snaps = list(read_ndjson(FIX, Snapshot))
    assert snaps[0].lighting.kind == "sh2" and snaps[0].lighting.dominant is not None
    for s in snaps[::6]:
        f = LightField.from_lighting(s.lighting)
        for o in s.objects:
            assert o.lit_color is not None and o.normal and o.position and o.shadow_vis is not None
            got = object_lit(f, o)[0]
            assert np.allclose(got, o.lit_color, atol=2e-3), (o.id, got, o.lit_color)


# ---- synthetic world + software renderer ------------------------------------------------------------
def test_world_deterministic_and_valid():
    a, ta, _ = generate_world(WorldConfig(lighting="dusk"), 4)
    b, tb, _ = generate_world(WorldConfig(lighting="dusk"), 4)
    assert a.model_dump_json() == b.model_dump_json()
    assert all(np.array_equal(ta["lit"][k], tb["lit"][k]) for k in ta["lit"])
    c, _, _ = generate_world(WorldConfig(lighting="dusk"), 5)
    assert a.model_dump_json() != c.model_dump_json()
    assert isinstance(a, Snapshot) and a.lighting.kind == "sh2"
    assert all(o.normal and o.position and o.shadow_vis is not None for o in a.objects)


def test_world_has_shadows_emissive_small_fields_in_every_light():
    for light in LIGHT_PRESETS:
        vis, small_em = [], 0
        for seed in range(6):
            s, t, _ = generate_world(WorldConfig(lighting=light, palette="mixed"), seed)
            vis += [o.shadow_vis for o in s.objects]
            small_em += sum(1 for o in s.objects if max(o.emissive_rgb) > 0 and np.sqrt(o.screen_area) * 60 < 2.0)
        assert min(vis) < 0.5 and small_em >= 3, (light, min(vis), small_em)


def test_renderer_truth_vs_sh2_and_shadow_ladder():
    err = {r: [] for r in ("albedo", "sh2", "sh2_shadow")}
    for light in ("day", "tinted", "dusk"):
        for seed in range(4):
            s, t, _ = generate_world(WorldConfig(lighting=light, palette="surfaces", n_objects=14), seed)
            f = LightField.from_lighting(s.lighting)
            gt = np.array([t["lit"][o.id] for o in s.objects])
            for r in err:
                err[r].append(np.abs(rung_colors(f, s.objects, r) - gt).mean())
    m = {r: float(np.mean(v)) for r, v in err.items()}
    assert m["sh2"] < m["albedo"] and m["sh2_shadow"] <= m["sh2"] + 1e-3 and m["sh2_shadow"] < 0.06, m


def test_renderer_emissive_independent_of_irradiance():
    w = LitWorld(WorldConfig(lighting="day", palette="signals", n_objects=8), 3)
    r1 = w.render()
    w.spec = LIGHT_PRESETS["fog"].__class__(**{**LIGHT_PRESETS["day"].__dict__, "sky": (0, 0, 0), "ground": (0, 0, 0),
                                                "dom_rgb": (0, 0, 0)})
    r2 = w.render()
    lamps = [i for i, o in enumerate(w.objs) if o["cls"] == "light"]
    for i in lamps:
        m = r1.inst == i
        if m.any():                                     # emission survives with zero irradiance
            assert (r2.rgb[m] @ np.array([0.2126, 0.7152, 0.0722])).mean() > 0.5 * (w.emis[i] @ np.array([0.2126, 0.7152, 0.0722]))
