"""K4 stabiliser: slew bound, trigger detection, double buffer, replay determinism, server sessions."""
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from core.server import create_app
from core.stabiliser import (Analysis, CueHysteresis, DoubleBuffer, PairHysteresis, Palette, PaletteBuilder, StabConfig,
                             Stabiliser, TriggerDetector, de00, flicker, replay, slew_limit)
from protocol import SnapshotBuilder
from protocol.models import Cue, Diagnostics, Overrides, PaletteOverride

DT = 1 / 30


def snap(f, objs, expo=1.0, events=(), obs="o"):
    """objs: id -> dict(alb, emis, state, tags, bbox)."""
    b = SnapshotBuilder(obs, 50.0, frame=f, t=round(f * DT, 9))
    b.ambient_dominant((0.3, 0.3, 0.3), (0.9, 0.9, 0.9), (0, 1, 0), exposure=expo)
    for i, o in objs.items():
        b.add_object(i, "wall", o.get("alb", (0.5, 0.5, 0.5)), bbox=o.get("bbox", (0.1, 0.1, 0.1, 0.1)), screen_area=0.01,
                     depth=5.0, state=o.get("state", "default"), emissive_rgb=o.get("emis", (0, 0, 0)),
                     task_tags=o.get("tags", []))
    for t, oid, data in events:
        b.add_event(t, oid, **data)
    return b.build()


def fake_solver(shift, ratios=None, log=None):
    """solve_fn: moves every albedo by `shift(frame, i)` (clipped) and emissive by the same; deterministic."""
    def solve_fn(s):
        if log is not None:
            log.append(s.frame)
        po = [PaletteOverride(id=o.id, albedo=tuple(np.clip(np.array(o.albedo_rgb) + shift(s.frame, k), 0, 1)),
                              emissive=tuple(np.maximum(np.array(o.emissive_rgb) + 2 * shift(s.frame, k), 0)))
              for k, o in enumerate(s.objects)]
        ids = [o.id for o in s.objects]
        an = Analysis({tuple(sorted(p)): r for p, r in (ratios or {}).items()}, {i: k for k, i in enumerate(ids)})
        return Overrides(frame=s.frame, palette_overrides=po, diagnostics=Diagnostics(solver_tier="exact")), an
    return solve_fn


# ---------------------------------------------------------------- slew
def test_slew_limit_bound_random():
    rng = np.random.default_rng(0)
    for _ in range(20):
        prev, tgt = rng.uniform(0, 1, (30, 3)), rng.uniform(0, 4, (30, 3))      # HDR targets too
        lim = rng.uniform(0.2, 6, 30)
        out, clamped = slew_limit(prev, tgt, lim)
        d = de00(prev, out)
        assert np.all(d <= lim + 1e-9)
        far = de00(prev, tgt) > lim
        assert np.array_equal(clamped, far)
        assert np.allclose(out[~far], tgt[~far])
        assert np.all(d[far] > 0.5 * lim[far])       # clamped rows still make progress (not frozen)


def test_stabiliser_slew_bound_every_frame_albedo_and_emissive():
    rng = np.random.default_rng(1)
    cfg = StabConfig(slew=True, r_max=30.0, default_dt=DT)
    st = Stabiliser(cfg)
    objs = {"a": dict(alb=(0.4, 0.5, 0.6), emis=(0.0, 0.0, 0.0)), "b": dict(alb=(0.2, 0.2, 0.2), emis=(3.0, 1.0, 0.1))}
    jumps = {f: rng.uniform(-0.6, 0.6, (2, 3)) for f in range(60)}
    solve_fn = fake_solver(lambda f, k: jumps[f][k])
    prev, clamped_total = None, 0
    for f in range(60):
        r = st.step(snap(f, objs), solve_fn)
        clamped_total += r.slew_clamped
        if prev is not None:
            for i in objs:
                assert de00(r.displayed[i][0], prev[i][0]) <= cfg.r_max * DT + 1e-6
                assert de00(r.displayed[i][1], prev[i][1]) <= cfg.r_max * DT + 1e-6
        prev = r.displayed
    assert clamped_total > 0 and st.buffer.rejected == 0


def test_slew_does_not_delay_true_scene_changes():
    """An annunciator turning green must show immediately: slew limits the correction offset, not the scene."""
    st = Stabiliser(StabConfig(slew=True, ema=True, r_max=10.0, default_dt=DT))
    off = fake_solver(lambda f, k: np.zeros(3))
    for f in range(3):
        st.step(snap(f, {"l": dict(emis=(3, 0, 0))}), off)
    r = st.step(snap(3, {"l": dict(emis=(0, 3, 0), state="green")}), off)
    assert np.allclose(r.displayed["l"][1], (0, 3, 0), atol=1e-6)


def test_flicker_metric():
    s = np.zeros((4, 1, 3))
    s[1], s[2], s[3] = 0.2, 0.2, 0.5
    f = flicker(s)
    assert f.shape == (1,) and f[0] == pytest.approx(float((de00(s[0], s[1]) + de00(s[2], s[3])).sum()))


# ---------------------------------------------------------------- hysteresis
def test_pair_hysteresis_band_and_hold():
    h = PairHysteresis(enter=1.0, exit=1.3, dwell=0, hold=2)
    k = ("a", "b")
    seq = [1.2, 0.9, 1.1, 1.25, 1.35, 1.1, 0.95]
    act = [k in h.update({k: r}, f)[0] for f, r in enumerate(seq)]
    assert act == [False, True, True, True, False, False, True]        # no chatter inside (1.0, 1.3)
    assert k in h.update({}, 7)[0] and k in h.update({}, 8)[0]          # vanished pair keeps state for `hold` frames
    h.update({}, 11)
    assert k not in h.state
    with pytest.raises(ValueError):
        PairHysteresis(enter=1.5, exit=1.0)


def test_cue_hysteresis_no_flicker():
    mk = lambda v: Cue(id="x", type="hatch", parameters={"variant": v})
    h = CueHysteresis(hold=3)
    out = h.update({"a": mk("h1"), "b": mk("h2")}, [("a", "b")], 0)
    # solver re-labels (swaps variants, still clash-free): keep the previous assignment
    out2 = h.update({"a": mk("h2"), "b": mk("h1")}, [("a", "b")], 1)
    assert out2["a"].parameters["variant"] == "h1" and out2["b"].parameters["variant"] == "h2"
    # clash after a change of the partner set: adopt the solver's assignment
    out3 = h.update({"a": mk("h1"), "b": mk("h1x")}, [("a", "b")], 2)
    assert out3["a"].parameters["variant"] != out3["b"].parameters["variant"]
    # a cue vanishes only after `hold` frames
    h.update({"a": mk("h1")}, [], 3)
    assert "b" in h.update({"a": mk("h1")}, [], 4)
    assert "b" not in h.update({"a": mk("h1")}, [], 9)


def test_hysteresis_holds_correction_when_solver_stops_acting():
    """Pair inside its exit band: the solver output goes back to the original; the stabiliser keeps the correction."""
    cfg = StabConfig(hysteresis=True, enter=1.0, exit=2.0, dwell=0, hold=0, default_dt=DT)
    st = Stabiliser(cfg)
    objs = {"a": dict(alb=(0.4, 0.4, 0.4)), "b": dict(alb=(0.42, 0.4, 0.4))}
    ratios = {("a", "b"): 0.5}
    shift = lambda f, k: np.array([0.2, 0, 0]) if ratios[("a", "b")] < 1.0 else np.zeros(3)

    def solve_fn(s):
        return fake_solver(shift, ratios)(s)
    r0 = st.step(snap(0, objs), solve_fn)
    assert r0.displayed["a"][0][0] == pytest.approx(0.6)
    ratios[("a", "b")] = 1.5                       # inside (1.0, 2.0): solver stops correcting, decision stays "confusing"
    r1 = st.step(snap(1, objs), solve_fn)
    assert r1.displayed["a"][0][0] == pytest.approx(0.6)
    ratios[("a", "b")] = 2.5                       # beyond exit: correction released
    r2 = st.step(snap(2, objs), solve_fn)
    assert r2.displayed["a"][0][0] == pytest.approx(0.4)


# ---------------------------------------------------------------- triggers
def base_objs():
    return {"a": dict(bbox=(0.1, 0.1, 0.1, 0.1)), "b": dict(bbox=(0.5, 0.5, 0.1, 0.1), tags=[])}


def fire(frames):
    d = TriggerDetector()
    return [d.detect(s) for s in frames]


def test_triggers_fire_on_the_right_events():
    o = base_objs()
    got = fire([snap(0, o), snap(1, o)])
    assert got == [["init"], []]
    # enter view: new id, and bbox crossing into the viewport
    o2 = {**o, "c": dict(bbox=(0.3, 0.3, 0.1, 0.1))}
    assert fire([snap(0, o), snap(1, o2)])[1] == ["enter_view"]
    off, on = {**o, "c": dict(bbox=(1.2, 0.3, 0.1, 0.1))}, {**o, "c": dict(bbox=(0.95, 0.3, 0.1, 0.1))}
    assert fire([snap(0, off), snap(1, on)])[1] == ["enter_view"]
    # collision event
    ev = [("collision", "a", {"other": "b"})]
    assert fire([snap(0, o), snap(1, o, events=ev)])[1] == ["collision"]
    # state change: by diff alone, and by event
    o3 = {**o, "a": dict(state="alt")}
    assert fire([snap(0, o), snap(1, o3)])[1] == ["state_change"]
    assert fire([snap(0, o), snap(1, o, events=[("state_change", "a", {})])])[1] == ["state_change"]
    # lighting: step fires, 1.5% jitter does not
    assert fire([snap(0, o), snap(1, o, expo=1.5)])[1] == ["lighting_change"]
    assert fire([snap(0, o), snap(1, o, expo=1.015)])[1] == []
    assert fire([snap(0, o), snap(1, o, events=[("lighting_change", None, {})])])[1] == ["lighting_change"]
    # importance jump (hazard tag appears)
    o4 = {**o, "b": dict(bbox=(0.5, 0.5, 0.1, 0.1), tags=["hazard"])}
    assert fire([snap(0, o), snap(1, o4)])[1] == ["importance_jump"]
    # profile change
    assert fire([snap(0, o, obs="p1"), snap(1, o, obs="p2")])[1] == ["profile_change"]
    # new confusing pair (from the hysteresis monitor)
    d = TriggerDetector()
    d.detect(snap(0, o))
    assert d.detect(snap(1, o), new_pairs={("a", "b")}) == ["new_pair"]
    # nothing else fires on a quiet frame with moving boxes
    o5 = {"a": dict(bbox=(0.11, 0.1, 0.1, 0.1)), "b": o["b"]}
    assert fire([snap(0, o), snap(1, o5)])[1] == []


def test_event_driven_resolves_only_on_triggers():
    log = []
    st = Stabiliser(StabConfig(event_driven=True, default_dt=DT))
    solve_fn = fake_solver(lambda f, k: np.zeros(3), log=log)
    an_fn = lambda s: Analysis()
    o = base_objs()
    frames = [snap(f, o) for f in range(5)] + [snap(5, o, events=[("collision", "a", {"other": "b"})]),
                                                snap(6, o), snap(7, o, expo=2.0), snap(8, o, expo=2.0)]
    res = [st.step(s, solve_fn, an_fn) for s in frames]
    assert log == [0, 5, 7]
    assert res[5].triggers == ["collision"] and res[7].triggers == ["lighting_change"]
    assert res[6].overrides.diagnostics.solver_tier == "cached"
    assert not res[3].solved and res[5].solved
    assert st.n_solves == 3


def test_event_driven_new_pair_trigger_via_monitor():
    log = []
    st = Stabiliser(StabConfig(event_driven=True, hysteresis=True, default_dt=DT))
    ratios = {("a", "b"): 5.0}
    solve_fn = fake_solver(lambda f, k: np.zeros(3), ratios, log)
    an_fn = lambda s: Analysis(dict(ratios), {"a": 0, "b": 1})
    o = base_objs()
    st.step(snap(0, o), solve_fn, an_fn)
    st.step(snap(1, o), solve_fn, an_fn)
    ratios[("a", "b")] = 0.4
    r = st.step(snap(2, o), solve_fn, an_fn)
    assert r.triggers == ["new_pair"] and log == [0, 2]


# ---------------------------------------------------------------- double buffer
def test_double_buffer_swaps_only_after_verification():
    buf = DoubleBuffer()
    p1 = Palette.build(1, 0, DT, {"a": (0.1, 0.1, 0.1)}, {"a": (0, 0, 0)})
    p2 = Palette.build(2, 1, DT, {"a": (0.9, 0.9, 0.9)}, {"a": (0, 0, 0)})
    buf.stage(p1)
    assert buf.read() is None                       # staged != visible
    assert buf.commit(lambda b, f: True) and buf.read() is p1
    buf.stage(p2)
    assert buf.read() is p1
    assert not buf.commit(lambda b, f: False) and buf.read() is p1 and buf.rejected == 1
    with pytest.raises(TypeError):
        p1.albedo["a"] = (0, 0, 0)                  # immutable
    bad = Palette.build(3, 2, DT, {"a": (0.5, 0.5, 0.5)}, {"a": (0, 0, 0)})
    object.__setattr__(bad, "digest", "tampered")
    buf.stage(bad)
    assert not buf.commit() and buf.read() is p1    # failed integrity: never exposed


def test_double_buffer_never_exposes_half_updated_palette():
    buf = DoubleBuffer()
    n_ent, stop, seen, errors = 24, threading.Event(), set(), []

    def reader():
        while not stop.is_set():
            p = buf.read()
            if p is None:
                continue
            vals = {v[0] for v in p.albedo.values()}
            if len(p.albedo) != n_ent or len(vals) != 1 or not p.intact() or vals != {p.version / 100}:
                errors.append((p.version, len(p.albedo), vals))
            seen.add(p.version)

    ts = [threading.Thread(target=reader) for _ in range(4)]
    [t.start() for t in ts]
    for v in range(1, 41):
        pb = PaletteBuilder()                       # the back buffer is filled entry by entry, slowly
        for k in range(n_ent):
            pb.set(f"e{k}", (v / 100,) * 3, (0, 0, 0))
            if k % 6 == 0:
                time.sleep(0.0002)
        buf.stage(pb.freeze(v, v, DT))
        assert buf.commit()
    time.sleep(0.01)
    stop.set()
    [t.join() for t in ts]
    assert not errors and len(seen) > 3


def test_stabiliser_rejects_bad_candidate_and_keeps_front():
    st = Stabiliser(StabConfig(slew=True, r_max=30.0, default_dt=DT), verify=lambda back, front: back.frame != 1)
    o = base_objs()
    good = fake_solver(lambda f, k: np.full(3, 0.3 * (f % 2)))
    r0 = st.step(snap(0, o), good)
    r1 = st.step(snap(1, o), good)
    assert r1.rejected and st.buffer.read().frame == r0.palette.frame == 0       # front kept
    assert [p.albedo for p in r1.overrides.palette_overrides] == [p.albedo for p in r0.overrides.palette_overrides]
    r2 = st.step(snap(2, o), good)                  # slew restarts from the validated front: bound still holds
    assert not r2.rejected and st.buffer.read().frame == 2
    for i in ("a", "b"):
        assert de00(r2.displayed[i][0], r0.displayed[i][0]) <= 30.0 * DT + 1e-6


def test_compute_delay_publishes_later_but_reader_sees_validated_front():
    st = Stabiliser(StabConfig(compute_frames=2, default_dt=DT))
    o = base_objs()
    solve_fn = fake_solver(lambda f, k: np.full(3, 0.05 * f))
    shown = [st.step(snap(f, o), solve_fn).palette for f in range(5)]
    assert shown[0] is None and shown[1] is None and shown[2].frame == 0 and shown[4].frame == 2


# ---------------------------------------------------------------- determinism
def test_replay_determinism_and_isolation():
    rng = np.random.default_rng(3)
    sh = {f: rng.uniform(-0.3, 0.3, (3, 3)) for f in range(40)}
    o = {"a": dict(), "b": dict(emis=(1, 1, 0)), "c": dict(alb=(0.9, 0.2, 0.2))}
    frames = [snap(f, o, expo=1 + 0.01 * (f % 3) + (0.5 if f > 20 else 0),
                   events=[("collision", "a", {"other": "b"})] if f == 10 else []) for f in range(40)]
    ratios = {("a", "b"): 0.5}
    solve_fn = fake_solver(lambda f, k: sh[f][k], ratios)
    an_fn = lambda s: Analysis(dict(ratios), {"a": 0, "b": 1, "c": 2})
    for cfg in (StabConfig(), StabConfig.all_on(compute_frames=1), StabConfig.only("slew"), StabConfig.only("event_driven")):
        a = replay(frames, cfg, solve_fn, an_fn)
        b = replay(frames, cfg, solve_fn, an_fn)
        assert [x.overrides.model_dump_json() for x in a] == [x.overrides.model_dump_json() for x in b]
    # a different session does not leak state
    s1, s2 = Stabiliser(StabConfig.all_on()), Stabiliser(StabConfig.all_on())
    for s in frames[:10]:
        s1.step(s, solve_fn, an_fn)
    fresh = s2.step(frames[0], solve_fn, an_fn)
    again = Stabiliser(StabConfig.all_on()).step(frames[0], solve_fn, an_fn)
    assert fresh.overrides.model_dump_json() == again.overrides.model_dump_json()


def test_real_solver_replay_deterministic():
    from adapters.synthetic.stream import StreamConfig, generate_stream
    from bench.experiments.stability_ablation import Ctx
    from core.observers import heldout_observers, train_observers
    cfg = StreamConfig(n_frames=14, first_event=3, spacing=2, n_fillers=1)
    snaps, events = generate_stream(cfg, 4, "dev")
    snaps2, _ = generate_stream(cfg, 4, "dev")
    assert [s.model_dump_json() for s in snaps] == [s.model_dump_json() for s in snaps2]
    tr = [o for o in train_observers(60, 4) if o.kind == cfg.kind][:2]
    held = [o for o in heldout_observers(40, 4) if o.kind == cfg.kind][:2]
    outs = []
    for _ in range(2):
        ctx = Ctx(snaps, held[0], {o.id: o for o in tr} | {held[0].id: held[0]}, held, dict(step_min=0.2, retries=0))
        outs.append([r.overrides.model_dump_json(exclude={"diagnostics"}) for r in
                     replay(snaps, StabConfig.all_on(), ctx.solve_fn, ctx.analyse_fn)])
    assert outs[0] == outs[1]


# ---------------------------------------------------------------- server
def _post(c, s):
    return c.post("/correct", content=s.model_dump_json(by_alias=True), headers={"content-type": "application/json"})


def test_server_sessions_and_stateless(tmp_path):
    from adapters.synthetic import SynthConfig, generate
    c = TestClient(create_app(gt_dir=tmp_path / "gt", profiles_db=tmp_path / "p.db"))
    frames = list(generate(SynthConfig(n_objects=6, n_frames=3, palette="confusable"), 0))
    frames = [f.model_copy(update={"observer_id": "tr-0-000" if False else f.observer_id}) for f in frames]
    obs_id = c.get("/observers").json()["train"][0]["id"]
    frames = [f.model_copy(update={"observer_id": obs_id, "budget_ms": 3000}) for f in frames]
    r0 = _post(c, frames[0])                                    # stateless: no session_id -> new session, id returned
    assert r0.status_code == 200
    sid = r0.json()["diagnostics"]["session_id"]
    assert sid and "init" in r0.json()["diagnostics"]["triggers"]
    r1 = _post(c, frames[1].model_copy(update={"session_id": sid}))
    d1 = r1.json()["diagnostics"]
    assert d1["session_id"] == sid and "init" not in d1["triggers"] and d1["slew_clamped"] >= 0
    r1b = _post(c, frames[1])                                   # a new anonymous call is a fresh session
    assert r1b.json()["diagnostics"]["session_id"] != sid and "init" in r1b.json()["diagnostics"]["triggers"]
    other = _post(c, frames[1].model_copy(update={"session_id": "other"}))
    assert "init" in other.json()["diagnostics"]["triggers"]
    assert _post(c, frames[0].model_copy(update={"observer_id": "nope"})).status_code == 404
