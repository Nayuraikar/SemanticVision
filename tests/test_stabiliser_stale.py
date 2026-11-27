"""Toy streams: groups the solver did not change must show the CURRENT original colour at every frame."""
import numpy as np
import pytest

from core.stabiliser import Analysis, StabConfig, Stabiliser
from protocol.models import Overrides, PaletteOverride
from tests.test_stabiliser import DT, snap

RED, GREEN = (3.0, 0.0, 0.0), (0.0, 3.0, 0.0)
SHIFT = np.array([0.2, 0.0, 0.0])
CONFIGS = {"event": StabConfig.only("event_driven", default_dt=DT), "all_on": StabConfig.all_on(default_dt=DT),
           "all_on_delay": StabConfig.all_on(default_dt=DT, compute_frames=2)}


def stream(trigger, n=12, switch=5):
    """Lamp red -> green at `switch`. trigger: the semantic state changes too; else only the emission does."""
    out = []
    for f in range(n):
        green = f >= switch
        out.append(snap(f, {"lamp": dict(emis=GREEN if green else RED, state="green" if green and trigger else "red"),
                            "wall": dict(alb=(0.4, 0.4, 0.4))}))
    return out


def solve_wall_only(s):
    """Changes only the wall; the lamp comes back as its own colour."""
    po = [PaletteOverride(id=o.id, albedo=tuple(np.array(o.albedo_rgb) + (SHIFT if o.id == "wall" else 0)),
                          emissive=o.emissive_rgb) for o in s.objects]
    ratios = {("lamp", "wall"): 0.5}
    return Overrides(frame=s.frame, palette_overrides=po), Analysis(ratios, {"lamp": 0, "wall": 1})


def assert_follows(snaps, res, oid):
    for s, r in zip(snaps, res):
        o = next(o for o in s.objects if o.id == oid)
        if oid in r.displayed:
            assert np.array_equal(r.displayed[oid][0], o.albedo_rgb), s.frame
            assert np.array_equal(r.displayed[oid][1], o.emissive_rgb), s.frame


@pytest.mark.parametrize("name", CONFIGS)
@pytest.mark.parametrize("trigger", [True, False], ids=["trigger", "no_trigger"])
def test_unchanged_group_follows_current_original(name, trigger):
    snaps = stream(trigger)
    st = Stabiliser(CONFIGS[name])
    an = solve_wall_only(snaps[0])[1]
    res = [st.step(s, solve_wall_only, lambda s: an) for s in snaps]
    assert res[5].solved == trigger
    assert_follows(snaps, res, "lamp")
    assert res[-1].displayed["wall"][0][0] > 0.55            # the changed group is still corrected


def test_changed_group_is_held_as_offset_from_current_original():
    """Original of a corrected object moves without a re-solve: the correction offset rides on the new original."""
    st = Stabiliser(StabConfig.only("event_driven", default_dt=DT))
    a = snap(0, {"lamp": dict(emis=RED), "wall": dict(alb=(0.4, 0.4, 0.4))})
    an_fn = lambda s: Analysis()
    st.step(a, solve_wall_only, an_fn)
    r = st.step(snap(1, {"lamp": dict(emis=RED), "wall": dict(alb=(0.1, 0.5, 0.3))}), solve_wall_only, an_fn)
    assert not r.solved
    assert np.allclose(r.displayed["wall"][0], np.array((0.1, 0.5, 0.3)) + SHIFT)


@pytest.mark.parametrize("name", CONFIGS)
def test_released_group_follows_when_its_original_changes(name):
    """Corrected while red; after the switch the solver leaves it alone: no held or decaying red correction."""
    def solve_fn(s):
        ov, an = solve_wall_only(s)
        po = [p if p.id == "wall" or s.objects[0].state == "green" else
              p.model_copy(update={"emissive": (3.0, 0.5, 0.0)}) for p in ov.palette_overrides]
        return ov.model_copy(update={"palette_overrides": po}), an
    snaps = stream(True)
    st = Stabiliser(CONFIGS[name])
    an = solve_wall_only(snaps[0])[1]
    res = [st.step(s, solve_fn, lambda s: an) for s in snaps]
    delay = CONFIGS[name].compute_frames
    assert_follows(snaps[5 + delay:], res[5 + delay:], "lamp")
