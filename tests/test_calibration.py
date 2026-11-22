import numpy as np
import pytest
from fastapi.testclient import TestClient

from adapters.synthetic.generator import SynthConfig, generate
from calibration import CalibConfig, Calibrator, Prior, SelfReport, Session, SimObserver
from calibration import model as M
from calibration import stimuli as S
from core.perception.cvd import machado_matrix
from core.profiles import ProfileStore, profile_observer
from core.server import create_app
from protocol.models import Overrides

N_FAST = 400


def _run(sess: Session, truth: SimObserver, seed=0, n=None):
    rng = np.random.default_rng(seed)
    while (st := sess.next()) is not None and (n is None or sess.n < n):
        sess.answer(st["trial_id"], truth.respond(st["mode"], st["c1"], st["c2"], st["size_deg"], st["bg_lum"], rng))
    return sess


# ---- stimuli / model ----
def test_pool_covers_axes_in_gamut():
    p = S.pool()
    assert set(p.axis.tolist()) == set(range(len(S.AXES)))
    assert set(p.level.tolist()) == set(range(len(S.LEVELS_Y)))
    for c in (p.c1, p.c2):
        assert c.min() >= -1e-9 and c.max() <= 1 + 1e-9
    assert np.all(np.abs(p.c1 - p.c2).max(1) > 1e-4)
    assert S.decode(p.n_stim - 1)[0] == p.n_pairs - 1


def test_mixture_matrices_match_published_tables():
    for i, k in enumerate(S.KINDS):
        for s in (0.0, 0.33, 1.0):
            w = np.eye(3)[i][None]
            assert np.allclose(M.mixture_matrices(w, np.array([s]))[0], machado_matrix(k, s), atol=1e-9)
    assert np.allclose(M.mixture_matrices(np.full((1, 3), 1 / 3), np.array([0.0]))[0], np.eye(3))


def test_response_models():
    r = np.linspace(0, 6, 50)
    p = M.p_diff(r)
    assert np.all(np.diff(p) >= 0) and p[0] == pytest.approx(M.GAMMA) and p[-1] <= 1 - M.LAPSE + 1e-9
    sp = M.slider_probs(r)
    assert sp.shape == (50, 5) and np.allclose(sp.sum(-1), 1) and np.all(sp >= 0)
    assert sp[0].argmax() == 0 and sp[-1].argmax() >= 3     # r->0 rates "0"; r=6 -> mu=.857 -> upper levels
    assert M.quantise_slider(0.62) == 0.5 and M.quantise_slider(1.0) == 1.0 and M.quantise_slider(-3) == 0.0


def test_threshold_and_gain_consistent_with_graph_tau():
    from core.perception.threshold import tau
    t = M.tau_combos(np.array([1.3]), np.array([0.25]))[0]
    assert t[S.REF_COMBO] == pytest.approx(float(tau(2.0, 0.2, 1, 1, 1.3)))
    assert M.gain(np.array([1.0]), 2.0, 0.5)[0] == pytest.approx(2.0) and M.gain(np.array([0.0]), 2.0, 0.5)[0] == pytest.approx(0.5)


def test_self_report_only_shifts_prior():
    base = Prior()
    p = Prior.from_self_report(SelfReport("tritan", ("yellow",)))
    assert p.alpha[2] > base.alpha[2] + 1 and p.alpha[0] == base.alpha[0]
    assert Prior.from_self_report(SelfReport("unsure", ())).alpha == base.alpha
    rg = Prior.from_self_report(SelfReport(None, ("red", "green", "brown")))
    assert rg.alpha[0] > base.alpha[0] and rg.alpha[1] > base.alpha[1] and rg.alpha[2] == base.alpha[2]
    rng = np.random.default_rng(0)
    m_base = M.sample_prior(4000, rng, base).w.mean(0)
    m_t = M.sample_prior(4000, rng, p).w.mean(0)
    assert m_t[2] > m_base[2] + 0.2
    # posterior can overrule: full support (no particle has zero weight on any type)
    assert M.sample_prior(500, rng, p).w.min() > 0


def test_z_roundtrip():
    P = M.sample_prior(300, np.random.default_rng(1))
    Q = M.from_z(M.to_z(P))
    assert np.allclose(P.w, Q.w, atol=2e-3) and np.allclose(P.sev, Q.sev, atol=2e-3) and np.allclose(P.bg, Q.bg, atol=1e-6)


# ---- Bayesian machinery ----
def test_eig_nonneg_and_selection_beats_average():
    c = Calibrator(0, N_FAST)
    cand = np.random.default_rng(0).choice(c.pool.n_stim, 300, replace=False)
    for mode in ("same_different", "slider"):
        g = c.eig(cand, mode)
        assert np.all(g >= -1e-9) and g.max() > 0
        stim, best = c.select(mode)
        assert best >= np.median(g) and best >= g.mean()
        assert c.eig([stim], mode)[0] == pytest.approx(best)


def test_update_is_deterministic_and_replayable():
    truth = SimObserver("machado", "deutan", 0.8, 1.2)
    s = _run(Session(CalibConfig(seed=3, n_particles=N_FAST, max_trials=14, min_trials=14)), truth, n=14)
    hist = list(s.cal.history)
    r = Session.replay(s.cfg, None, hist)
    assert np.allclose(r.cal.logw, s.cal.logw) and np.allclose(r.cal.parts.w, s.cal.parts.w)
    assert r.summary() == s.summary()
    nxt = s.next()
    assert r.next() == nxt                      # selection deterministic given history
    assert len(set(h[0] for h in hist)) == len(hist)   # never repeats a stimulus


def test_parameter_recovery_in_model():
    truth = SimObserver("machado", "deutan", 0.9, 1.0)
    s = _run(Session(CalibConfig(seed=1, n_particles=800, max_trials=32, min_trials=32)), truth, seed=1, n=32)
    sm = s.summary()
    w0 = Calibrator(1, 800).width()
    assert sm["width"] < w0 - 0.15
    e = sm["effective_severity"]
    assert e[0] + e[1] > 0.5 and e[2] < 0.25               # red-green, not tritan
    assert abs(np.log(sm["params"]["k"]["mean"]) - truth.canonical()["logk"]) < 0.5
    assert sm["type_probs"]["deutan"] + sm["type_probs"]["protan"] > 0.7


def test_normal_vision_and_tritan():
    s = _run(Session(CalibConfig(seed=2, n_particles=800, max_trials=32, min_trials=32)), SimObserver(severity=0.0), seed=2, n=32)
    assert sum(s.summary()["effective_severity"]) < 0.45     # residual mass: mild severity trades off with k at 32 trials
    t = _run(Session(CalibConfig(seed=4, n_particles=800, max_trials=32, min_trials=32)), SimObserver("machado", "tritan", 0.9), seed=4, n=32)
    assert t.summary()["dominant"] == "tritan"


def test_stop_rule():
    truth = SimObserver("machado", "protan", 0.7)
    a = _run(Session(CalibConfig(seed=0, n_particles=N_FAST, min_trials=6, max_trials=20, target_width=5.0)), truth)
    assert a.n == 6 and a.progress()["done"] and a.next() is None
    b = _run(Session(CalibConfig(seed=0, n_particles=N_FAST, min_trials=4, max_trials=9, target_width=1e-6)), truth)
    assert b.n == 9
    with pytest.raises(ValueError):
        CalibConfig.from_dict({"mode": "nope"})
    with pytest.raises(ValueError):
        CalibConfig.from_dict({"min_trials": 10, "max_trials": 5})


def test_session_validation_and_pending_idempotent():
    s = Session(CalibConfig(seed=0, n_particles=N_FAST))
    st = s.next()
    assert s.next() == st and "stim" not in st and st["mode"] == "same_different"
    with pytest.raises(KeyError):
        s.answer(99, "same")
    with pytest.raises(ValueError):
        s.answer(st["trial_id"], 0.5)
    s.answer(st["trial_id"], "different")
    st2 = s.next()
    assert st2["mode"] == "slider" and st2["trial_id"] == 1
    with pytest.raises(ValueError):
        s.answer(1, 1.5)
    s.answer(1, 0.6)
    assert s.cal.history[-1][2] in M.RATINGS


def test_samples_and_map():
    truth = SimObserver("machado", "deutan", 1.0)
    s = _run(Session(CalibConfig(seed=5, n_particles=N_FAST, max_trials=16, min_trials=16)), truth, n=16)
    smp = s.cal.samples(200)
    assert len(smp["severity"]) == 200 and set(M.PARAMS) <= set(smp)
    w = np.array([smp["w_protan"], smp["w_deutan"], smp["w_tritan"]])
    assert np.allclose(w.sum(0), 1)
    mp = s.cal.discrimination_map(80)
    assert len(mp) == len(S.AXES) * len(S.LEVELS_Y)
    cell = next(c for c in mp if c["axis"] == "lightness")
    assert cell["delta50"] is not None and cell["lo"] <= cell["delta50"] <= cell["hi"]


# ---- storage ----
def test_profile_store_roundtrip(tmp_path):
    st = ProfileStore(tmp_path / "p.sqlite")
    sess = Session(CalibConfig(seed=0, n_particles=N_FAST))
    pid = st.create(sess.cfg.to_dict(), {"type_guess": None, "struggle_colors": []}, sess.summary())
    p = st.get(pid)
    assert p["status"] == "calibrating" and p["n_trials"] == 0 and p["samples"] is None
    st.add_trial(pid, 0, 7, "same_different", "different", sess.summary())
    st.add_trial(pid, 1, 9, "slider", 0.25, sess.summary())
    assert st.trials(pid) == [(7, "same_different", "different"), (9, "slider", 0.25)] and st.get(pid)["n_trials"] == 2
    st.finish(pid, sess.summary(), sess.cal.samples(10))
    assert st.get(pid)["status"] == "complete" and len(st.get(pid)["samples"]["severity"]) == 10
    assert ProfileStore(tmp_path / "p.sqlite").get(pid)["id"] == pid      # persisted
    assert st.get("nope") is None
    st.delete(pid)
    assert st.get(pid) is None and st.trials(pid) == []


def test_profile_observer_conversion():
    sess = _run(Session(CalibConfig(seed=0, n_particles=N_FAST, max_trials=10, min_trials=10)), SimObserver("machado", "tritan", 1.0), n=10)
    o = profile_observer("abc", sess.summary())
    assert o.id == "profile:abc" and o.split == "train" and o.model == "machado" and 0 <= o.severity <= 1 and o.k > 0
    p = sess.summary()["params"]
    S_ = 0.5 * (p["s_L"]["mean"] + p["s_C"]["mean"])
    assert o.k == pytest.approx(p["k"]["mean"] * S_)


# ---- server ----
@pytest.fixture()
def client(tmp_path):
    return TestClient(create_app(gt_dir=tmp_path / "gt", profiles_db=tmp_path / "p.sqlite"))


def _calibrate(client, pid, truth, seed=0, max_steps=40):
    rng = np.random.default_rng(seed)
    for _ in range(max_steps):
        r = client.get(f"/profiles/{pid}/next")
        assert r.status_code == 200, r.text
        j = r.json()
        if j["done"]:
            return j
        st = j["stimulus"]
        assert set(st) >= {"trial_id", "mode", "c1", "c2", "size_deg", "bg_lum"} and all(0 <= x <= 1 for x in st["c1"] + st["c2"])
        a = client.post(f"/profiles/{pid}/answer", json={"trial_id": st["trial_id"], "response": truth.respond(
            st["mode"], st["c1"], st["c2"], st["size_deg"], st["bg_lum"], rng)})
        assert a.status_code == 200, a.text
    raise AssertionError("did not finish")


def test_profile_endpoints_flow(client, tmp_path):
    r = client.post("/profiles", json={"self_report": {"type_guess": "deutan", "struggle_colors": ["red", "green"]},
                                       "config": {"seed": 1, "min_trials": 6, "max_trials": 10, "n_particles": N_FAST}})
    assert r.status_code == 201
    pid = r.json()["id"]
    assert r.json()["status"] == "calibrating" and "not a medical diagnosis" in r.json()["note"]
    assert client.get("/profiles/zzz").status_code == 404 and client.get("/profiles/zzz/next").status_code == 404
    st = client.get(f"/profiles/{pid}/next").json()["stimulus"]
    assert client.get(f"/profiles/{pid}/next").json()["stimulus"] == st                         # idempotent
    assert client.post(f"/profiles/{pid}/answer", json={"trial_id": 5, "response": "same"}).status_code == 409
    assert client.post(f"/profiles/{pid}/answer", json={"trial_id": 0, "response": 0.5}).status_code == 422
    assert client.post(f"/profiles/{pid}/answer", json={"trial_id": 0, "response": "maybe"}).status_code == 422
    assert client.post("/profiles", json={"config": {"mode": "x"}}).status_code == 422
    assert client.post("/profiles", json={"self_report": {"type_guess": "bogus"}}).status_code == 422
    truth = SimObserver("machado", "deutan", 0.9)
    rng = np.random.default_rng(0)
    client.post(f"/profiles/{pid}/answer", json={"trial_id": 0, "response": truth.respond(st["mode"], st["c1"], st["c2"], st["size_deg"], st["bg_lum"], rng)})
    # restart: a fresh app on the same DB replays the trial log and continues identically
    app2 = TestClient(create_app(gt_dir=tmp_path / "gt", profiles_db=tmp_path / "p.sqlite"))
    g = app2.get(f"/profiles/{pid}").json()
    assert g["n_trials"] == 1 and g["progress"]["n"] == 1
    assert app2.get(f"/profiles/{pid}/next").json()["stimulus"]["trial_id"] == 1
    j = _calibrate(app2, pid, truth, seed=1)
    assert j["done"] and j["progress"]["n"] >= 6
    f = app2.post(f"/profiles/{pid}/finish")
    assert f.status_code == 200 and f.json()["status"] == "complete" and f.json()["summary"]["discrimination_map"]
    full = app2.get(f"/profiles/{pid}?samples=true").json()
    assert len(full["samples"]["severity"]) == 500 and full["n_samples"] == 500
    assert app2.get(f"/profiles/{pid}/next").status_code == 409
    assert app2.post(f"/profiles/{pid}/answer", json={"trial_id": 99, "response": "same"}).status_code == 409


def test_correct_with_profile_observer(client):
    pid = client.post("/profiles", json={"config": {"seed": 2, "min_trials": 4, "max_trials": 6, "n_particles": N_FAST}}).json()["id"]
    _calibrate(client, pid, SimObserver("machado", "deutan", 1.0), seed=2)
    snap = list(generate(SynthConfig(n_objects=6, n_frames=1), seed=0))[0]

    def post(oid):
        s = snap.model_copy(update={"observer_id": oid})
        return client.post("/correct", content=s.model_dump_json(by_alias=True), headers={"content-type": "application/json"})

    r = post(f"profile:{pid}")
    assert r.status_code == 200, r.text
    ov = Overrides.model_validate(r.json())
    assert ov.frame == snap.frame and ov.diagnostics.solver_tier in ("exact", "allocated")
    assert post(f"profile:{pid}").json()["diagnostics"]["solver_tier"] == "cached"
    assert post("profile:missing").status_code == 404


def test_bench_experiment_smoke():
    from bench.experiments.calibration_recovery import run
    out = run({"n_obs": 1, "max_trials": 8, "min_trials": 4, "marks": [0, 8], "n_particles": 300, "n_probes": 20}, seed=0)
    assert {"eig@0", "eig@8", "eig@stop", "random@0", "random@8"} <= set(out)
    assert all(np.isfinite(v) for a in out.values() for v in a.values())
    assert out["eig@0"]["pred_err"] == pytest.approx(out["random@0"]["pred_err"])      # same prior
