import numpy as np, pytest, colour
from core.perception import color as C, cvd, distance as Dm, threshold as T, graph as G

rng = np.random.default_rng(42)
RGB = rng.uniform(0, 1, (500, 3))
XYZ = C.linear_to_xyz(RGB)
WXY = colour.XYZ_to_xy(C.D65_XYZ)

# ---- published reference data ----
MACH1 = {  # Machado 2009 Table, severity 1.0
    "protan": [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
    "deutan": [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]],
    "tritan": [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]],
}
VIENOT_RGB = {  # Vienot 1999 dichromat matrices on linear sRGB
    "protan": [[0.11238, 0.88762, 0.0], [0.11238, 0.88762, 0.0], [0.00401, -0.00401, 1.0]],
    "deutan": [[0.29275, 0.70725, 0.0], [0.29275, 0.70725, 0.0], [-0.02234, 0.02234, 1.0]],
}
SHARMA = [  # Sharma, Wu, Dalal 2005 test pairs (subset), dE00 to 4 dp
    ((50, 2.6772, -79.7751), (50, 0, -82.7485), 2.0425),
    ((50, 3.1571, -77.2803), (50, 0, -82.7485), 2.8615),
    ((50, 2.8361, -74.0200), (50, 0, -82.7485), 3.4412),
    ((50, -1.3802, -84.2814), (50, 0, -82.7485), 1.0000),
    ((50, 2.5, 0), (50, 0, -2.5), 4.3065),
    ((50, 2.5, 0), (73, 25, -18), 27.1492),
    ((50, 2.5, 0), (61, -5, 29), 22.8977),
    ((50, 2.5, 0), (56, -27, -3), 31.9030),
    ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
    ((2.0776, 0.0795, -1.1350), (0.9033, -0.0636, -0.5514), 0.9082),
]


def test_srgb_eotf_vs_colour():
    c = rng.uniform(0, 1, (200, 3))
    assert np.abs(C.srgb_to_linear(c) - colour.models.eotf_sRGB(c)).max() < 1e-9
    assert np.abs(C.linear_to_srgb(RGB) - colour.models.eotf_inverse_sRGB(RGB)).max() < 1e-9
    assert np.abs(C.linear_to_srgb(C.srgb_to_linear(c)) - c).max() < 1e-12


def test_rgb_xyz_lms():
    ref = colour.RGB_to_XYZ(RGB, colour.RGB_COLOURSPACES["sRGB"], illuminant=WXY, chromatic_adaptation_transform=None)
    assert np.abs(XYZ - ref).max() < 3e-4
    assert np.abs(C.xyz_to_linear(XYZ) - RGB).max() < 1e-12
    assert np.abs(C.lms_to_linear(C.linear_to_lms(RGB)) - RGB).max() < 1e-12
    hpe = XYZ @ np.array(colour.appearance.ciecam02.MATRIX_XYZ_TO_HPE).T
    assert np.abs(C.linear_to_lms(RGB) - hpe).max() < 3e-4  # same HPE matrix, rounding in sRGB matrix
    assert np.abs(C.M_XYZ2LMS @ np.ones(3) - 1).max() < 1e-4  # equal-energy normalised HPE


def test_cat16_vs_colour():
    src = colour.xy_to_XYZ(colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["A"])
    ref = colour.adaptation.chromatic_adaptation_VonKries(
        XYZ, src, C.D65_XYZ, transform="CAT16")
    assert np.abs(C.cat16_adapt(XYZ, src, C.D65_XYZ, 1.0) - ref).max() < 1e-9
    assert np.abs(C.cat16_adapt(C.D65_XYZ, C.D65_XYZ) - C.D65_XYZ).max() < 1e-12
    # white maps to white at any D only when D=1; D=0 is identity
    assert np.abs(C.cat16_adapt(XYZ, src, C.D65_XYZ, 0.0) - XYZ).max() < 1e-12
    assert np.abs(C.cat16_adapt(src, src, C.D65_XYZ, 1.0) - C.D65_XYZ).max() < 1e-9
    # degree of adaptation vs colour's CIECAM16 formula
    from colour.appearance.cam16 import degree_of_adaptation
    assert abs(C.cat16_degree(64 / np.pi * 0.2, 1.0) - degree_of_adaptation(1.0, 64 / np.pi * 0.2)) < 1e-12


def test_lab_vs_colour():
    lab = C.xyz_to_lab(XYZ)
    ref = colour.XYZ_to_Lab(XYZ, WXY)
    assert np.abs(lab - ref).max() < 1e-4  # colour uses 216/24389 constants
    assert np.abs(C.lab_to_xyz(lab) - XYZ).max() < 1e-12
    dark = XYZ * 1e-3
    assert np.abs(C.xyz_to_lab(dark) - colour.XYZ_to_Lab(dark, WXY)).max() < 1e-3


@pytest.mark.parametrize("a,b,d", SHARMA)
def test_ciede2000_sharma(a, b, d):
    assert abs(float(C.ciede2000(a, b)) - d) < 5e-4
    assert abs(float(C.ciede2000(b, a)) - d) < 5e-4


def test_ciede2000_vs_colour():
    l1 = np.stack([rng.uniform(0, 100, 2000), rng.uniform(-100, 100, 2000), rng.uniform(-100, 100, 2000)], -1)
    l2 = np.stack([rng.uniform(0, 100, 2000), rng.uniform(-100, 100, 2000), rng.uniform(-100, 100, 2000)], -1)
    assert np.abs(C.ciede2000(l1, l2) - colour.delta_E(l1, l2, method="CIE 2000")).max() < 1e-9
    assert np.allclose(C.ciede2000(l1[:5], l1[:5]), 0)
    assert np.abs(C.ciede2000(l1[:, None][:3], l2[None, :3]).shape == (3, 3))
    # gray axis (zero chroma) edge cases
    g = np.array([[50, 0, 0.0]]); h = np.array([[60, 0, 0.0]])
    assert abs(C.ciede2000(g, h)[0] - colour.delta_E(g, h)[0]) < 1e-9


def test_cam16ucs_vs_colour():
    xyz = rng.uniform(0.02, 0.95, (300, 3)) * np.array([.95, 1, 1.08])
    for LA, Yb, sur in [(64 / np.pi * 0.2, 0.2, "average"), (20.0, 0.1, "dim"), (100.0, 0.3, "dark")]:
        s = colour.XYZ_to_CAM16(xyz * 100, C.D65_XYZ * 100, LA, Yb * 100 if False else Yb * 100 * 1.0 if False else Yb,
                                colour.VIEWING_CONDITIONS_CAM16[sur.capitalize()])
        ref = colour.JMh_CAM16_to_CAM16UCS(np.stack([s.J, s.M, s.h], -1))
        mine = C.xyz_to_cam16ucs(xyz, C.D65_XYZ, LA, Yb, sur)
        assert np.abs(ref - mine).max() < 1e-4
    # distance matches colour's delta_E CAM16-UCS
    a, b = xyz[:100], xyz[100:200]
    ua, ub = C.xyz_to_cam16ucs(a), C.xyz_to_cam16ucs(b)
    ref = colour.delta_E(colour.XYZ_to_CAM16UCS(a, ) if False else
                         colour.JMh_CAM16_to_CAM16UCS(np.stack([(s := colour.XYZ_to_CAM16(a * 100, C.D65_XYZ * 100, 64 / np.pi * 0.2, 0.2)).J, s.M, s.h], -1)),
                         colour.JMh_CAM16_to_CAM16UCS(np.stack([(s := colour.XYZ_to_CAM16(b * 100, C.D65_XYZ * 100, 64 / np.pi * 0.2, 0.2)).J, s.M, s.h], -1)),
                         method="CAM16-UCS")
    assert np.abs(C.cam16ucs_distance(ua, ub) - ref).max() < 1e-4


@pytest.mark.parametrize("kind", ["protan", "deutan", "tritan"])
def test_machado(kind):
    name = {"protan": "Protanomaly", "deutan": "Deuteranomaly", "tritan": "Tritanomaly"}[kind]
    assert np.abs(cvd.machado_matrix(kind, 1.0) - np.array(MACH1[kind])).max() < 1e-6  # published
    assert np.abs(cvd.machado_matrix(kind, 0.0) - np.eye(3)).max() < 1e-12
    tab = colour.blindness.CVD_MATRICES_MACHADO2010[name]
    for s in np.round(np.linspace(0, 1, 11), 1):  # tabulated severities: exact vs colour
        ref = colour.blindness.matrix_cvd_Machado2009(name, float(s))
        assert np.abs(cvd.machado_matrix(kind, float(s)) - ref).max() < 1e-6
    for s in np.linspace(0.01, 0.99, 23):  # in-between: linear interpolation of colour's published table
        lo = np.floor(s * 10 + 1e-9) / 10
        lo = min(round(lo, 1), 0.9)
        f = (s - lo) / 0.1
        ref = (1 - f) * np.asarray(tab[lo]) + f * np.asarray(tab[round(lo + 0.1, 1)])
        assert np.abs(cvd.machado_matrix(kind, float(s)) - ref).max() < 1e-6
    ref = colour.algebra.vecmul(colour.blindness.matrix_cvd_Machado2009(name, 0.3), RGB)
    assert np.abs(cvd.machado(RGB, kind, 0.3, clip=False) - ref).max() < 1e-9
    # white preserved (rows sum to ~1)
    assert np.abs(cvd.machado(np.ones(3), kind, 0.6, clip=False) - 1).max() < 2e-3
    with pytest.raises(ValueError):
        cvd.machado_matrix(kind, 1.2)


@pytest.mark.parametrize("kind", ["protan", "deutan"])
def test_vienot(kind):
    assert np.abs(cvd.vienot_matrix(kind) - np.array(VIENOT_RGB[kind])).max() < 5e-4  # published
    out = cvd.vienot(RGB, kind, clip=False)
    assert np.abs(cvd.vienot(out, kind, clip=False) - out).max() < 1e-9  # idempotent projection
    assert np.abs(cvd.vienot(np.full(3, .5), kind, clip=False) - .5).max() < 2e-3
    # agrees with colour's Machado severity-1 to within a model-difference tolerance
    ref = colour.algebra.vecmul(colour.blindness.matrix_cvd_Machado2009(
        {"protan": "Protanomaly", "deutan": "Deuteranomaly"}[kind], 1.0), RGB)
    assert np.abs(cvd.vienot(RGB, kind, clip=False) - ref).mean() < 0.06


@pytest.mark.parametrize("kind", ["protan", "deutan", "tritan"])
def test_brettel(kind):
    out = cvd.brettel(RGB, kind, clip=False)
    assert np.abs(cvd.brettel(out, kind, clip=False) - out).max() < 1e-6  # idempotent
    g = np.linspace(0, 1, 11)[:, None] * np.ones(3)
    assert np.abs(cvd.brettel(g, kind, clip=False) - g).max() < 1e-6  # neutrals invariant
    ax = {"protan": 0, "deutan": 1, "tritan": 2}[kind]
    # confusion lines are parallel to the missing cone axis: other cone responses unchanged
    l0, l1 = C.linear_to_lms(RGB), C.linear_to_lms(out)
    oth = [i for i in range(3) if i != ax]
    assert np.abs(l0[:, oth] - l1[:, oth]).max() < 1e-9
    # close to Machado sev 1 and (prot/deut) to Vienot
    ref = colour.algebra.vecmul(colour.blindness.matrix_cvd_Machado2009(
        {"protan": "Protanomaly", "deutan": "Deuteranomaly", "tritan": "Tritanomaly"}[kind], 1.0), RGB)
    assert np.abs(out - ref).mean() < 0.08
    if kind != "tritan":
        assert np.abs(out - cvd.vienot(RGB, kind, clip=False)).mean() < 0.06


def test_brettel_half_planes_differ():
    P1, P2, _, _ = cvd.brettel_matrices("deutan")
    assert np.abs(P1 - P2).max() > 1e-3
    # the two planes meet along the white axis
    W = C.M_RGB2LMS @ np.ones(3)
    assert np.abs(P1 @ W - W).max() < 1e-9 and np.abs(P2 @ W - W).max() < 1e-9


def test_simulate_dispatch_and_severity():
    for m in ("machado", "vienot", "brettel"):
        assert np.abs(cvd.simulate(RGB, m, "deutan", 0.0) - RGB).max() < 1e-12
        full = cvd.simulate(RGB, m, "deutan", 1.0)
        half = cvd.simulate(RGB, m, "deutan", 0.5)
        assert np.abs(RGB - half).mean() < np.abs(RGB - full).mean()


def test_pair_distance():
    red, green = np.array([0.8, 0.05, 0.02]), np.array([0.05, 0.3, 0.02])
    n = Dm.pair_distance(red, green)
    d = Dm.pair_distance(red, green, model="machado", kind="deutan", severity=1.0)
    assert n > 30 and d < n / 2
    assert Dm.pair_distance(red, red, model="brettel", kind="protan") == pytest.approx(0, abs=1e-9)
    ref = colour.delta_E(colour.XYZ_to_Lab(C.linear_to_xyz(red), WXY), colour.XYZ_to_Lab(C.linear_to_xyz(green), WXY), method="CIE 2000")
    assert abs(n - ref) < 1e-4
    assert Dm.pair_distance(red, green, space="cam16") > Dm.pair_distance(red, green, model="vienot", kind="deutan", space="cam16")
    M = Dm.pair_matrix(RGB[:20], model="machado", kind="protan")
    assert M.shape == (20, 20) and np.allclose(M, M.T, atol=1e-9) and np.allclose(np.diag(M), 0, atol=1e-9)
    assert abs(M[2, 5] - Dm.pair_distance(RGB[2], RGB[5], model="machado", kind="protan")) < 1e-9
    # CAT16 under illuminant A: neutral under that illuminant stays neutral
    A = colour.xy_to_XYZ(colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["A"])
    lit_white = C.xyz_to_linear(A / A[1] * 0.5)
    p = Dm.perceived(lit_white, illum_xyz=A, L_A=1e5)
    assert np.ptp(p) < 1e-6


def test_tau_monotonic():
    s = np.array([0.2, 0.5, 1, 2, 5, 20])
    t = T.tau(s, 0.2)
    assert np.all(np.diff(t) <= 0) and t[-1] > 0
    assert T.tau(2, 0.2) < T.tau(2, 0.01) and T.tau(2, 0.2) < T.tau(2, 1.0)
    assert T.tau(2, 0.2, s_L=2, s_C=2) < T.tau(2, 0.2) < T.tau(2, 0.2, s_L=0.5, s_C=0.5)
    assert T.tau(2, 0.2, k=2) == pytest.approx(2 * T.tau(2, 0.2))
    assert T.tau(1e-9, 0.2) <= T.ThresholdParams().size_cap * 2.3 * 1.0001 + 1e-9
    assert T.tau(100, 0.2) == pytest.approx(2.3 * (1 + (2 / 100) ** 0.5))
    assert T.tau(np.ones((3, 4)), np.ones((3, 4)) * .2).shape == (3, 4)


def test_graph():
    cols = np.array([[0.8, 0.05, 0.02], [0.35, 0.3, 0.02], [0.02, 0.02, 0.8], [0.8, 0.05, 0.03]])
    bbox = np.array([[0, 0, 1, 1], [1.1, 0, 2, 1], [50, 50, 51, 51], [0, 1.05, 1, 2]])
    g = G.build_graph(cols, ids=list("abcd"), size_deg=[3, 3, 3, 0.5], importance=[1, 1, 1, 1], bbox=bbox,
                      adj_margin=0.2, model="machado", kind="deutan", severity=1.0)
    assert g.n == 4 and g.D.shape == (4, 4) and g.adjacency[0, 1] and g.adjacency[0, 3] and not g.adjacency[0, 2]
    assert np.array_equal(g.adjacency, g.adjacency.T)
    assert g.weight[0, 2] == 0 and (0, 2) not in map(tuple, g.edges)  # far pair consumes no budget
    assert (0, 1) in map(tuple, g.edges)  # red/green adjacent confusable under deutan
    assert g.tau[0, 3] > g.tau[0, 1]  # small member raises tau
    assert g.size_weight[0, 3] < g.size_weight[0, 1]
    assert all(g.edge_margin > 0) and len(g.edge_weight) == len(g.edges)
    assert list(g.neighbours(0)) == [1, 3] or set(g.neighbours(0)) <= {1, 3}
    # normal vision: red/green not confusable
    g0 = G.build_graph(cols[:2], bbox=bbox[:2], adj_margin=0.2)
    assert len(g0.edges) == 0
    # far_weight>0 keeps far pairs
    g2 = G.build_graph(cols, bbox=bbox, adj_margin=0.2, far_weight=0.1, model="machado", kind="deutan")
    assert g2.weight[0, 2] > 0 or not g2.confusable[0, 2]
    assert np.allclose(G.bbox_adjacency(bbox, 0.2), g.adjacency)
