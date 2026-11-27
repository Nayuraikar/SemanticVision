import { beforeAll, describe, expect, it } from "vitest";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import {
  TrafficSim, DashboardSim, exportSnapshot, findCollisionStep, initPhysics, recordDashboard, recordTraffic, svMeshes,
} from "../index";
import { actionsFor, invalidatesOverrides, type Action } from "./actions";
import { InputQueue } from "./inputQueue";
import { GROUNDS, setKeys, spawnPlayer, stepPlayer, walkTo, zoom, ZOOM_MAX, ZOOM_MIN } from "./movement";
import { PlaySession, replaySession, runHash, type PlaySceneId } from "./session";
import { PlayDashboardSim, PlayTrafficSim } from "./worlds";

const DT = 1 / 60;
beforeAll(() => initPhysics());

const SCRIPT: Record<PlaySceneId, [number, Action][]> = {
  traffic: [
    [5, { type: "keys", forward: 1, right: 0, turn: 0.5, sprint: true }], [40, { type: "keys", forward: 0, right: 0, turn: 0, sprint: false }],
    [50, { type: "signal", axis: "ew" }], [60, { type: "mission", name: "brake" }], [70, { type: "cross", id: "ped0" }],
    [90, { type: "env", env: "night" }], [100, { type: "collision" }], [110, { type: "mission", name: "lineup" }],
  ],
  dashboard: [
    [5, { type: "fault", system: "env" }], [150, { type: "ack", level: "warning" }], [160, { type: "gauge", system: "hyd", value: 0.05 }],
    [170, { type: "lighting", env: "dim" }], [200, { type: "mission", name: "gauge" }],
  ],
  toy: [[5, { type: "lamp" }], [20, { type: "palette", preset: "lookalike" }], [30, { type: "drag", id: "cone", x: -2, z: 3 }]],
};
const driver = (script: [number, Action][]) => (s: PlaySession) => {
  for (const [at, a] of script) if (at === s.step) s.push(a);
};

describe("input queue", () => {
  it("applies in step and push order, once, and rejects inputs older than the queue", () => {
    const q = new InputQueue(1);
    q.push(2, { type: "lamp" }); q.push(2, { type: "collision" }); q.push(5, { type: "lamp" });
    expect(q.take(1)).toEqual([]);
    expect(q.take(2).map((i) => i.action.type)).toEqual(["lamp", "collision"]);
    expect(q.take(2)).toEqual([]);
    expect(() => q.push(1, { type: "lamp" })).toThrow();
    expect(new InputQueue(1, q.log()).log()).toEqual(q.log());
  });

  it.each(["traffic", "dashboard", "toy"] as PlaySceneId[])("%s: same inputs and recorded-log replay give the same stream hash", (sceneId) => {
    const run = (script: [number, Action][]) => {
      const s = new PlaySession({ sceneId, seed: 3 });
      const hash = runHash(s, 300, 30, driver(script)), log = s.log();
      s.dispose();
      return { hash, log };
    };
    const a = run(SCRIPT[sceneId]);
    expect(run(SCRIPT[sceneId]).hash).toBe(a.hash);
    const r = replaySession(JSON.parse(JSON.stringify(a.log)));
    expect(runHash(r, a.log.steps)).toBe(a.hash);
    r.dispose();
    expect(run([]).hash).not.toBe(a.hash);
  }, 30000);

  it("play sources use no Math.random or wall clock", () => {
    for (const f of ["inputQueue.ts", "actions.ts", "movement.ts", "worlds.ts", "session.ts"]) {
      expect(fs.readFileSync(path.resolve(__dirname, f), "utf8")).not.toMatch(/Math\.random\(|Date\.now|performance\.now|new Date/);
    }
  });
});

describe("movement bounds", () => {
  it.each(["traffic", "dashboard", "toy"] as PlaySceneId[])("%s: sprinting in any direction stays inside and out of blocks", (id) => {
    const g = GROUNDS[id], p = spawnPlayer(g);
    for (const [forward, right, turn] of [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [1, 1, 0.3], [-1, -1, -0.7]]) {
      setKeys(p, { forward, right, turn, sprint: true });
      for (let i = 0; i < 1200; i++) {
        stepPlayer(p, g, DT);
        expect(p.x >= g.minX && p.x <= g.maxX && p.z >= g.minZ && p.z <= g.maxZ).toBe(true);
        expect(g.blocks.some((b) => p.x > b.minX && p.x < b.maxX && p.z > b.minZ && p.z < b.maxZ)).toBe(false);
      }
    }
  });

  it("acceleration eases in, walk goals are clamped and reached, zoom is bounded", () => {
    const g = GROUNDS.toy, p = spawnPlayer(g);
    setKeys(p, { forward: 1, right: 0, turn: 0, sprint: false });
    stepPlayer(p, g, DT);
    const first = Math.hypot(p.vx, p.vz);
    for (let i = 0; i < 60; i++) stepPlayer(p, g, DT);
    expect(first).toBeLessThan(g.speed / 3);
    expect(Math.hypot(p.vx, p.vz)).toBeCloseTo(g.speed, 1);
    setKeys(p, { forward: 0, right: 0, turn: 0, sprint: false });
    walkTo(p, g, 999, 2);
    for (let i = 0; i < 900; i++) stepPlayer(p, g, DT);
    expect(p.goal).toBeNull();
    expect(p.x).toBeCloseTo(g.maxX, 0);
    zoom(p, 100); expect(p.zoom).toBe(ZOOM_MAX);
    zoom(p, 0.001); expect(p.zoom).toBe(ZOOM_MIN);
  });
});

describe("action -> state mapping", () => {
  it("maps clicked ids to scene actions", () => {
    expect(actionsFor("traffic", "sig_W_green")[0].action).toEqual({ type: "signal", axis: "ew" });
    expect(actionsFor("traffic", "veh3_brake")[0].action).toEqual({ type: "brake", id: "veh3" });
    expect(actionsFor("traffic", "ped2_head")[0].action).toEqual({ type: "cross", id: "ped2" });
    expect(actionsFor("traffic", "hedge_nw_e")).toEqual([]);
    expect(actionsFor("dashboard", "ann_hyd_2")[0].action).toEqual({ type: "fault", system: "hyd" });
    expect(actionsFor("dashboard", "status_env")[0].action).toEqual({ type: "fault", system: "env" });
    expect(actionsFor("dashboard", "master_warning")[0].action).toEqual({ type: "ack", level: "warning" });
    expect(actionsFor("toy", "lamp")[0].action).toEqual({ type: "lamp" });
    expect(invalidatesOverrides({ type: "lighting", env: "dim" })).toBe(true);
    expect(invalidatesOverrides({ type: "zoom", factor: 1 })).toBe(false);
  });

  it("traffic: signal cycles, a car brakes with its light on, the scripted collision runs on demand", () => {
    const s = new PlaySession({ sceneId: "traffic", seed: 1 });
    const state = (id: string) => svMeshes(s.scene).find((m) => m.userData.sv.id === id)?.userData.sv.state;
    const order = ["red", "amber", "green"];
    const was = s.facts().signals!.ew;
    s.push({ type: "signal", axis: "ew" }); s.tick();
    const now = order[(order.indexOf(was) + 1) % 3];
    expect(s.facts().signals!.ew).toBe(now);
    expect(state(`sig_E_${now}`)).toBe(now);
    expect(state(`sig_W_${now}`)).toBe(now);
    const car = svMeshes(s.scene).find((m) => m.userData.sv.cls === "car" && m.userData.sv.state === "moving")!.userData.sv.id as string;
    s.push({ type: "brake", id: car });
    for (let i = 0; i < 10; i++) s.tick();
    expect(s.facts().braking).toContain(car);
    expect(state(`${car}_brake`)).toBe("on");
    s.push({ type: "collision" });
    for (let i = 0; i < 600 && !s.facts().alarms; i++) s.tick();
    const f = s.facts();
    expect(f.alarms).toBe(1);
    expect(f.crash).toContain("hazard_beacon");
    expect(f.hazards.length).toBeGreaterThanOrEqual(2);
    expect(state(f.crash![0])).toBe("damaged");
    s.dispose();
  });

  it("traffic: a pedestrian told to cross crosses", () => {
    const sim = new PlayTrafficSim({ seed: 1, scenario: "normal", env: "day" });
    const ped = () => svMeshes(sim.scene).find((m) => m.userData.sv.id === "ped0")!.userData.sv.state;
    sim.cross("ped0"); sim.tick();
    expect(ped()).toBe("crossing");
    sim.physics!.free();
  });

  it("dashboard: injected fault cascades caution then warning, acknowledge clears the master, gauges push", () => {
    const sim = new PlayDashboardSim({ seed: 1, scenario: "quiet", env: "normal" });
    const s = new PlaySession({ sceneId: "dashboard", seed: 1 });
    s.push({ type: "fault", system: "avnx" });                  // a system without a gauge
    const seen: string[] = [];
    for (let i = 0; i < 300; i++) {
      s.tick();
      const f = s.facts();
      if (f.cautions!.includes("master_caution") && !seen.includes("caution")) seen.push("caution");
      if (f.warnings!.includes("master_warning") && !seen.includes("warning")) seen.push("warning");
    }
    expect(seen).toEqual(["caution", "warning"]);
    expect(s.facts().fault).toBe("avnx");
    s.push({ type: "ack", level: "warning" }); s.tick();
    expect(s.facts().warnings).not.toContain("master_warning");
    expect(s.facts().warnings).toContain("ann_avnx_0");
    s.push({ type: "mission", name: "end" }); s.tick();
    expect(s.facts().hazards).toEqual([]);
    sim.push("hyd", 0.05);
    expect(sim.facts().outGauges).toEqual(["hyd"]);
    sim.tick();
    expect(sim.facts().gauges!.hyd).toBe(0.05);
    sim.push("hyd", null);
    expect(sim.facts().outGauges).toEqual([]);
    s.dispose();
  });

  it("toy: lamp toggles and stays, drag is bounded, separate mission starts together", () => {
    const s = new PlaySession({ sceneId: "toy", seed: 1 });
    const lamp = () => svMeshes(s.scene).find((m) => m.userData.sv.id === "lamp")!.userData.sv.state;
    for (let i = 0; i < 200; i++) s.tick();
    expect(lamp()).toBe("red");
    s.push({ type: "lamp" }); s.tick();
    expect(lamp()).toBe("green");
    s.push({ type: "mission", name: "separate" }); s.tick();
    expect(s.facts().separation!).toBeLessThan(1.2);
    s.push({ type: "drag", id: "cone", x: -999, z: 999 }); s.tick();
    expect(svMeshes(s.scene).find((m) => m.userData.sv.id === "cone")!.position.x).toBe(-7);
    expect(s.facts().separation!).toBeGreaterThan(3);
    s.dispose();
  });
});

describe("Watch mode is unchanged", () => {
  const FIX = (n: string) => path.resolve(__dirname, `../../../tests/fixtures/${n}.ndjson`);
  const sha = (n: string) => crypto.createHash("sha256").update(fs.readFileSync(FIX(n))).digest("hex");
  const near = (a: unknown, b: unknown, p = ""): void => {
    if (typeof a === "number" && typeof b === "number") return void expect(a, p).toBeCloseTo(b, 4);
    if (a && b && typeof a === "object") {
      expect(Object.keys(a as object).sort(), p).toEqual(Object.keys(b as object).sort());
      for (const k of Object.keys(a as object)) near((a as any)[k], (b as any)[k], `${p}.${k}`);
      return;
    }
    expect(a, p).toEqual(b);
  };
  const saved = (n: string) => fs.readFileSync(FIX(n), "utf8").trim().split("\n").map((l) => JSON.parse(l));

  it("recorded Watch streams still match the committed fixtures, even after a play session ran", async () => {
    const before = ["traffic_day", "dashboard_normal", "toy_scene"].map(sha);
    const s = new PlaySession({ sceneId: "traffic", seed: 1 });
    runHash(s, 200, 50, driver(SCRIPT.traffic));
    s.dispose();
    const c = await findCollisionStep({ seed: 1, scenario: "collision", env: "day" });
    const first = (Math.floor(c / 6) - 1) * 6;
    const t = await recordTraffic({ seed: 1, scenario: "collision", env: "day", steps: first + 18, exportEvery: 6, startStep: first });
    saved("traffic_day").forEach((x, i) => near(t.snaps[i], x, `traffic.${i}`));
    expect(t.snaps.length).toBe(saved("traffic_day").length);
    expect(["traffic_day", "dashboard_normal", "toy_scene"].map(sha)).toEqual(before);
  }, 30000);

  it("play subclasses without inputs reproduce the scripted sims exactly", async () => {
    const snap = (sim: TrafficSim | DashboardSim) =>
      exportSnapshot(sim.scene, sim.camera, { frame: 0, t: sim.time, observerId: "x", budgetMs: 1 });
    const o = { seed: 1, scenario: "collision" as const, env: "day" as const };
    const a = new TrafficSim(o), b = new PlayTrafficSim(o);
    const d = new DashboardSim({ seed: 1, scenario: "alarm", env: "normal" }), e = new PlayDashboardSim({ seed: 1, scenario: "alarm", env: "normal" });
    for (let i = 1; i <= 480; i++) {
      a.tick(); b.tick(); d.tick(); e.tick();
      if (i % 40) continue;
      expect(snap(b)).toEqual(snap(a));
      const x = snap(d), y = snap(e);
      // play tags only the faulted gauge as hazard, exactly as the script does
      expect(y).toEqual(x);
    }
    a.physics!.free(); b.physics!.free();
    expect((await recordDashboard({ seed: 1, scenario: "alarm", env: "normal", steps: 24, exportEvery: 12 })).snaps.length).toBe(2);
  }, 30000);
});
