import { describe, expect, it } from "vitest";
import { exportSnapshot } from "../../../adapters/threejs/snapshot";
import * as trafficScene from "../../../adapters/threejs/scenes/traffic";
import * as dashboardScene from "../../../adapters/threejs/scenes/dashboard";
import { initPhysics } from "../../../adapters/threejs/physics";
import { DEFAULT_FLAGS } from "../overlays";
import { SimDriver, precomputeMarkers } from "./driver";
import { FxLayer, wrap } from "./fxLayer";
import {
  DEFAULT_FX, cinematicBlockers, effectiveFx, followWeight, groupKey, topGroups,
} from "./fxPolicy";
import { captureLights, lerp, startRelight } from "./relight";
import {
  advanceClock, clampStep, crossedMarker, extractMarkers, fractionOfStep, keyAction, nextMarker, scrub, seekPlan, stepFromFraction,
} from "./timeline";

const snap = (d: SimDriver) => exportSnapshot(d.scene, d.camera, { frame: 0, t: 1, observerId: "t0", budgetMs: 40 });
const now = (fn: () => void) => fn();

describe("event extraction", () => {
  const sc = (object_id: string) => ({ type: "state_change" as const, object_id });
  it("groups by kind and step, merges routine changes, ignores triggers", () => {
    const ms = extractMarkers([
      { step: 100, events: [sc("veh0")] },
      { step: 10, events: [sc("sig_E_red"), sc("sig_W_red"), { type: "trigger_enter", object_id: "veh0" }] },
      { step: 110, events: [sc("veh1"), { type: "collision", object_id: "veh0", data: { other: "veh1", point: [1, 0, 2] } }] },
      { step: 200, events: [sc("master_warning"), sc("veh0")] },
    ]);
    expect(ms.map((m) => [m.step, m.kind])).toEqual([[10, "signal"], [100, "state"], [110, "collision"], [200, "cascade"], [200, "state"]]);
    expect(ms[0].ids).toEqual(["sig_E_red", "sig_W_red"]);
    expect(ms[2]).toMatchObject({ ids: ["veh0", "veh1"], point: [1, 0, 2] });
    expect(nextMarker(ms, 100)!.step).toBe(110);
    expect(nextMarker(ms, 500)!.step).toBe(10);
    expect(nextMarker([], 0)).toBeNull();
    expect(crossedMarker(ms, 99, 300)!.kind).toBe("collision");
    expect(crossedMarker(ms, 10, 100)).toBeNull();
  });
  it("precomputes the traffic collision and the dashboard cascade from the sims' own logs", async () => {
    const t = (await precomputeMarkers("traffic", undefined, now))!;
    expect(t.filter((m) => m.kind === "collision")).toHaveLength(1);
    expect(t.some((m) => m.kind === "signal")).toBe(true);
    const d = (await precomputeMarkers("dashboard", undefined, now))!;
    expect(d.filter((m) => m.kind === "cascade").length).toBeGreaterThanOrEqual(3);
    expect(await precomputeMarkers("toy", () => true, now)).toBeNull();
    expect((await precomputeMarkers("toy", undefined, now))!.length).toBeGreaterThan(0);
  });
});

describe("seek and scrub", () => {
  it("maps fractions, clamps, steps and plans rebuilds", () => {
    expect([stepFromFraction(0.5, 960), stepFromFraction(2, 960), clampStep(-3, 960)]).toEqual([480, 960, 0]);
    expect(fractionOfStep(240, 960)).toBe(0.25);
    expect(fractionOfStep(5, 0)).toBe(0);
    expect([scrub(10, -1, 960), scrub(10, 1, 960)]).toEqual([0, 40]);
    expect(seekPlan(10, 50)).toEqual({ rebuild: false, ticks: 40 });
    expect(seekPlan(50, 10)).toEqual({ rebuild: true, ticks: 10 });
  });
  it("runs the playback clock at the chosen speed", () => {
    const total = (speed: number) => {
      let acc = 0, n = 0;
      for (let i = 0; i < 60; i++) { const c = advanceClock(acc, 1 / 60, speed); acc = c.acc; n += c.steps; }
      return n;
    };
    expect([total(0.25), total(1), total(2)]).toEqual([15, 60, 120]);
    expect(advanceClock(0, 10, 1).steps).toBe(15);
  });
  it("maps shortcuts without stealing keys from form controls", () => {
    expect(keyAction(" ", null)).toBe("toggle");
    expect(keyAction("ArrowLeft", { tagName: "INPUT", type: "range" })).toBe("back");
    expect(keyAction(" ", { tagName: "BUTTON" })).toBeNull();
    expect(keyAction("ArrowRight", { tagName: "SELECT" })).toBeNull();
    expect(keyAction("ArrowRight", { tagName: "BUTTON", role: "radio" })).toBeNull();
  });
  it("seeking back and forth reproduces the same dashboard snapshot, including after the cascade", () => {
    const d = new SimDriver("dashboard", 16 / 9);
    d.seek(250);
    const a = snap(d);
    d.seek(700);
    expect(d.seek(250)).toBe(true);
    expect(snap(d)).toEqual(a);
    d.dispose();
  });
  it("seeking reproduces the traffic snapshot after the collision", async () => {
    await initPhysics();
    const d = new SimDriver("traffic", 16 / 9);
    d.seek(500);
    const a = snap(d);
    d.seek(40);
    d.seek(500);
    expect(snap(d)).toEqual(a);
    d.dispose();
  });
});

describe("effects", () => {
  it("exports an identical snapshot with particles and waypoints on and off", async () => {
    await initPhysics();
    trafficScene.setEnv("rain");
    const d = new SimDriver("traffic", 16 / 9);
    d.seek(300);
    const events = structuredClone(d.scene.userData.svEvents);
    const off = snap(d);
    const children = d.scene.children.length;
    const layer = new FxLayer();
    for (const weather of ["rain", "fog"]) {
      layer.update(d.camera, { dust: true, weather, waypoints: topGroups(off.objects ?? []), animate: true, dt: 0.1, t: 1 });
    }
    d.scene.userData.svEvents = events;
    expect(snap(d)).toEqual(off);
    expect(d.scene.children.length).toBe(children);
    let n = 0;
    layer.scene.traverse((o) => { expect(o.userData.sv).toBeUndefined(); n++; });
    expect(n).toBeGreaterThan(5);
    layer.dispose(); d.dispose(); trafficScene.setEnv("day");
  });
  it("forces cinematic off while measurements are visible or in performance mode", () => {
    const hud = { tier: false, latency: false, stats: false, posterior: true };
    const clean = { graph: false, heatmap: false, bboxes: true, labels: false };
    const on = { ...DEFAULT_FX, cinematic: true };
    expect(DEFAULT_FX.cinematic).toBe(false);
    expect(effectiveFx(on, cinematicBlockers(clean, hud), false, false).cinematic).toBe(true);
    expect(Object.values(DEFAULT_FLAGS).some(Boolean)).toBe(false);
    expect(effectiveFx(on, cinematicBlockers({ ...DEFAULT_FLAGS, graph: true }, hud), false, false).cinematic).toBe(false);
    expect(cinematicBlockers(clean, { ...hud, latency: true })).toEqual(["latency"]);
    expect(effectiveFx(on, [], true, false)).toMatchObject({ cinematic: false, dust: false, weather: false, animate: false });
  });
  it("picks top tagged groups, merging parts", () => {
    const o = (id: string, screen_area: number, task_tags: string[]) => ({ id, screen_area, task_tags, position: [0, 0, 0] }) as any;
    expect([groupKey("veh0_brake"), groupKey("sig_E_red"), groupKey("g_eng_needle"), groupKey("status_hyd")])
      .toEqual(["veh0", "sig_E", "g_eng", "ann_hyd"]);
    const w = topGroups([o("veh0", 0.04, ["hazard"]), o("veh0_ind", 0.01, ["hazard"]), o("sig_E_red", 0.04, ["signal"]), o("road", 0.9, [])]);
    expect(w.map((x) => x.key)).toEqual(["veh0", "sig_E"]);
  });
  it("eases the follow camera in and out and wraps particles", () => {
    expect([followWeight(0), followWeight(1.5), followWeight(10)]).toEqual([0, 1, 0]);
    expect(followWeight(0.45)).toBeCloseTo(0.5);
    expect(lerp(2, 4, 0.5)).toBe(3);
    expect([wrap(5, 4), wrap(-5, 4), wrap(1, 4)]).toEqual([-3, 3, 1]);
  });
  it("relight ends exactly where a direct environment apply would", async () => {
    await initPhysics();
    for (const [id, from, to] of [["traffic", "day", "fog"], ["traffic", "rain", "night"], ["dashboard", "normal", "glare"]]) {
      const set = (e: string) => (id === "traffic" ? trafficScene.setEnv(e as any) : dashboardScene.setEnv(e as any));
      set(from);
      const d = new SimDriver(id, 16 / 9);
      d.run(1);
      const start = captureLights(d.scene).background.getHex();
      const apply = startRelight(d.scene, to);
      expect(captureLights(d.scene).background.getHex()).toBe(start);
      apply(0.5); apply(1);
      set(to);
      const ref = new SimDriver(id, 16 / 9);
      ref.run(1);
      const a = snap(d), b = snap(ref);
      expect(a.lighting).toEqual(b.lighting);
      d.dispose(); ref.dispose();
    }
    trafficScene.setEnv("day"); dashboardScene.setEnv("normal");
  });
});
