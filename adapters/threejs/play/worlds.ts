// Play-mode worlds. They subclass / wrap the sims; the scripted Watch scenarios are never modified.
import * as THREE from "three";
import { svMeshes, type SVMesh } from "../snapshot";
import { Rng } from "../rng";
import { DT } from "../physics";
import { TrafficSim, signalAt, type Ped, type TrafficOpts, type Vehicle } from "../trafficSim";
import {
  DEPENDS, DashboardSim, GAUGE_SYSTEMS, LAMP_EMIT, LAMP_LENS, SYSTEMS, bandOf,
  type DashLighting, type Gauge, type System,
} from "../dashboardSim";
import { applyEnvironment, type EnvName } from "../environments";
import { LIGHT_STATES, buildToyScene, stepToyScene } from "../toyScene";
import type { Action, Axis, MissionName, PaletteName } from "./actions";

type V3 = [number, number, number];
type Light = "red" | "amber" | "green";
const LIGHTS: Light[] = ["red", "amber", "green"];

export interface Hazard { id: string; x: number; z: number }
/** What the HUD, the minimap and mission scoring read. */
export interface Facts {
  hazards: Hazard[];
  /** increases whenever a new alarm starts (collision, master warning) */
  alarms: number;
  signals?: Record<Axis, Light>;
  fault?: System | null;
  warnings?: string[];
  cautions?: string[];
  outGauges?: System[];
  gauges?: Record<string, number>;
  braking?: string[];
  crash?: string[];
  redCar?: string | null;
  separation?: number;
}

export interface World {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  /** one fixed step; `step` is the session step after the tick */
  tick(step: number): void;
  apply(a: Action, rng: Rng): void;
  facts(): Facts;
  dispose(): void;
}

// ------------------------------------------------------------------ traffic
const WARMUP = 360, RELOAD_S = 110, HOLD = 180, MISSION_HOLD = 1800;
const RED_CAR: V3 = [0.55, 0.04, 0.03], BROWN_CAR: V3 = [0.22, 0.08, 0.03];

export class PlayTrafficSim extends TrafficSim {
  forced: Partial<Record<Axis, Light>> = {};
  held = new Map<string, number>();
  private crossing = new Set<string>();
  private parked: SVMesh[] = [];
  redCar: string | null = null;
  private crashes = 0;

  constructor(opts: TrafficOpts) { super({ ...opts }); }

  protected signals(): ReturnType<typeof signalAt> { return { ...super.signals(), ...this.forced }; }

  protected control(v: Vehicle, sig: ReturnType<typeof signalAt>, pos: V3) {
    const was = v.halt;
    if ((this.held.get(v.id) ?? 0) > this.step) v.halt = true;
    super.control(v, sig, pos);
    v.halt = was;
  }

  protected pedControl(p: Ped, sig: ReturnType<typeof signalAt>, pos: V3) {
    if (this.crossing.delete(p.id) && !p.crossing && !p.halt) { p.crossing = true; this.setState(p.mesh, "crossing"); }
    super.pedControl(p, sig, pos);
  }

  cycleSignal(axis: Axis) {
    this.forced[axis] = LIGHTS[(LIGHTS.indexOf(this.signals()[axis]) + 1) % LIGHTS.length];
    this.updateSignals();
  }
  brake(id: string, steps = HOLD) {
    if (this.vehicles.some((v) => v.id === id && !v.damaged)) this.held.set(id, this.step + steps);
  }
  cross(id: string) { if (this.peds.some((p) => p.id === id)) this.crossing.add(id); }

  /** (Re)start free-flowing traffic (warmed up so cars are present) or the scripted collision scenario. */
  restart(scenario: "normal" | "collision") {
    this.opts.scenario = scenario;
    this.forced = {}; this.held.clear(); this.crossing.clear();
    this.reset();
    if (scenario === "normal") for (let i = 0; i < WARMUP; i++) this.tick();
    this.scene.userData.svEvents = [];
  }

  lineup(redFirst: boolean) {
    this.unlineup();
    [redFirst, !redFirst].forEach((red, i) => {
      const mat = new THREE.MeshStandardMaterial({ roughness: 0.4, metalness: 0 });
      mat.color.setRGB(...(red ? RED_CAR : BROWN_CAR));
      const m = new THREE.Mesh(new THREE.BoxGeometry(4.2, 1.4, 1.9), mat) as SVMesh;
      m.userData.sv = { id: `park_${"ab"[i]}`, cls: "car", state: "stopped", tags: ["vehicle"] };
      m.position.set(-8, 0.85, 9 + 3.5 * i);
      this.scene.add(m);
      this.parked.push(m);
      if (red) this.redCar = `park_${"ab"[i]}`;
    });
  }
  unlineup() {
    for (const m of this.parked) { this.scene.remove(m); m.geometry.dispose(); m.material.dispose(); }
    this.parked = []; this.redCar = null;
  }

  mission(name: MissionName, rng: Rng) {
    this.unlineup(); this.forced = {}; this.held.clear();
    if (name === "brake") {
      const candidates = () => this.vehicles.filter((v) => !v.damaged && Math.abs(v.mesh.position.x + v.mesh.position.z) < 36);
      if (this.opts.scenario !== "normal" || !candidates().length) this.restart("normal");
      const xs = candidates();
      if (xs.length) this.brake(rng.pick(xs).id, MISSION_HOLD);
    } else if (name === "signal") {
      const ew = rng.pick(LIGHTS);
      this.forced = { ew, ns: ew === "red" ? "green" : "red" };
    } else if (name === "crash") this.restart("collision");
    else if (name === "lineup") this.lineup(rng.next() < 0.5);
    this.updateSignals();
  }

  tick() {
    const before = this.collisionPoint;
    super.tick();
    if (!before && this.collisionPoint) this.crashes++;
  }
  get alarmCount() { return this.crashes ?? 0; }

  facts(): Facts {
    const hazards: Hazard[] = [], braking: string[] = [], crash: string[] = [];
    for (const v of this.vehicles) {
      if (v.damaged || v.hazard) hazards.push({ id: v.id, x: v.mesh.position.x, z: v.mesh.position.z });
      if (v.brake.userData.sv.state === "on") braking.push(v.id, `${v.id}_brake`);
      if (v.damaged) crash.push(v.id, `${v.id}_brake`, `${v.id}_ind`, `${v.id}_ind_r`);
    }
    if (this.collisionPoint) crash.push("hazard_beacon", "smoke");
    const s = this.signals();
    return { hazards, alarms: this.alarmCount, signals: { ew: s.ew, ns: s.ns }, braking, crash, redCar: this.redCar };
  }
}

export function trafficWorld(seed: number, env: EnvName, aspect: number): World {
  const sim = new PlayTrafficSim({ seed, scenario: "normal", env, aspect });
  sim.restart("normal");
  return {
    scene: sim.scene, camera: sim.camera,
    tick: () => {
      // spawns run out after 120 s: reload free-flowing traffic (step-based, so replayable)
      if (sim.opts.scenario === "normal" && sim.time > RELOAD_S) sim.restart("normal");
      sim.tick();
    },
    apply: (a, rng) => {
      if (a.type === "signal") sim.cycleSignal(a.axis);
      else if (a.type === "brake") sim.brake(a.id);
      else if (a.type === "cross") sim.cross(a.id);
      else if (a.type === "collision") sim.restart("collision");
      else if (a.type === "env") applyEnvironment(sim.scene, a.env);
      else if (a.type === "mission") sim.mission(a.name, rng);
    },
    facts: () => sim.facts(),
    dispose: () => { sim.physics?.free(); sim.physics = null; },
  };
}

// ------------------------------------------------------------------ dashboard
export class PlayDashboardSim extends DashboardSim {
  // read through `?.`: the base constructor updates gauges before this field exists
  private pushed: Map<System, number> | undefined = new Map();
  private alarms = 0;

  protected gaugeValue(g: Gauge, v: number): number { return this.pushed?.get(g.def.sys) ?? v; }

  protected updateGauges() {
    super.updateGauges();
    for (const g of this.gauges) {
      const faulted = this.cascade?.system === g.def.sys && this.step >= this.cascade.caution;
      const out = faulted || (!!this.pushed?.has(g.def.sys) && bandOf(g.def.bad, g.v) !== "normal");
      for (const m of [g.needle, g.readout, g.face, ...g.arcs]) m.userData.sv.tags = out ? ["hazard"] : [];
    }
  }

  /** Back to the quiet panel without restarting the clock. */
  clearFaults() {
    this.cascade = null;
    this.pushed?.clear();
    for (const [id, l] of this.lamps) {
      if (id.startsWith("status_")) {
        l.level = "normal"; l.emit = LAMP_EMIT.normal;
        l.mesh.material.color.setRGB(...LAMP_LENS.normal);
        delete l.mesh.userData.svOrig;
      }
      l.mesh.userData.sv.tags = ["signal"];
      this.light(id, l.level === "normal" && l.sys !== null);
    }
    this.updateGauges();
  }

  inject(system: System, rng: Rng) {
    this.clearFaults();
    const sec = (s: number) => Math.round(s / DT);
    const caution = this.step + sec(0.4);
    const warning = caution + sec(rng.uniform(1.0, 1.8));
    this.cascade = { system, related: DEPENDS[system], caution, warning, spread: warning + sec(rng.uniform(0.5, 1.2)) };
  }

  acknowledge(level: "warning" | "caution") { this.light(`master_${level}`, false); this.updateGauges(); }

  push(system: System, value: number | null) {
    if (!GAUGE_SYSTEMS.includes(system)) return;
    if (value === null) this.pushed!.delete(system);
    else this.pushed!.set(system, Math.min(1, Math.max(0, value)));
    this.updateGauges();
  }

  mission(name: MissionName, rng: Rng) {
    this.clearFaults();
    if (name === "fault" || name === "levels") this.inject(rng.pick(SYSTEMS), rng);
    else if (name === "gauge") {
      const g = rng.pick(this.gauges);
      this.push(g.def.sys, g.def.bad === "high" ? 0.93 : 0.08);
    }
  }

  tick() {
    super.tick();
    if (this.cascade && this.step === this.cascade.warning) this.alarms = (this.alarms ?? 0) + 1;
  }

  facts(): Facts {
    const hazards: Hazard[] = [], warnings: string[] = [], cautions: string[] = [];
    for (const [id, l] of this.lamps) {
      if (!l.on || l.level === "normal") continue;
      (l.level === "warning" ? warnings : cautions).push(id);
      hazards.push({ id, x: l.mesh.position.x, z: l.mesh.position.z });
    }
    return {
      hazards, warnings, cautions, alarms: this.alarms, fault: this.cascade?.system ?? null,
      outGauges: this.gauges.filter((g) => bandOf(g.def.bad, g.v) !== "normal").map((g) => g.def.sys),
      gauges: Object.fromEntries(this.gauges.map((g) => [g.def.sys, g.v])),
    };
  }
}

export function dashboardWorld(seed: number, env: DashLighting, aspect: number): World {
  const sim = new PlayDashboardSim({ seed, scenario: "quiet", env, aspect });
  return {
    scene: sim.scene, camera: sim.camera,
    tick: () => sim.tick(),
    apply: (a, rng) => {
      if (a.type === "fault") sim.inject(a.system, rng);
      else if (a.type === "ack") sim.acknowledge(a.level);
      else if (a.type === "gauge") sim.push(a.system, a.value);
      else if (a.type === "lighting") sim.setLighting(a.env);
      else if (a.type === "mission") sim.mission(a.name, rng);
    },
    facts: () => sim.facts(),
    dispose: () => {},
  };
}

// ------------------------------------------------------------------ toy
export const PALETTES: Record<PaletteName, Record<"car" | "bush" | "cone", V3>> = {
  default: { car: [0.8, 0.05, 0.05], bush: [0.1, 0.5, 0.08], cone: [0.9, 0.35, 0.02] },
  lookalike: { car: [0.35, 0.2, 0.04], bush: [0.3, 0.3, 0.04], cone: [0.45, 0.25, 0.03] },
  swapped: { car: [0.1, 0.5, 0.08], bush: [0.8, 0.05, 0.05], cone: [0.9, 0.35, 0.02] },
};
export const DRAG_BOUNDS = { minX: -7, maxX: 7, minZ: -6, maxZ: 6 };
export const SEPARATION_GOAL = 3;
const DRAGGABLE = ["car", "bush", "cone", "lamp"];

export function toyWorld(aspect: number): World {
  const { scene, camera } = buildToyScene(aspect);
  const meshes = new Map(svMeshes(scene).map((m) => [m.userData.sv.id as string, m]));
  const dragged = new Map<string, [number, number]>();
  let lamp = "red";
  const place = (id: string, x: number, z: number) => {
    if (!DRAGGABLE.includes(id)) return;
    const c = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
    dragged.set(id, [c(x, DRAG_BOUNDS.minX, DRAG_BOUNDS.maxX), c(z, DRAG_BOUNDS.minZ, DRAG_BOUNDS.maxZ)]);
  };
  const paint = (preset: PaletteName) => {
    for (const [id, rgb] of Object.entries(PALETTES[preset])) {
      const m = meshes.get(id)!;
      m.material.color.setRGB(...rgb);
      delete m.userData.svOrig;
    }
  };
  const hold = () => {
    const m = meshes.get("lamp")!, e = LIGHT_STATES[lamp];
    m.userData.sv.state = lamp;                       // the player controls the lamp, not the timer
    if (m.userData.svOrig) m.userData.svOrig.emissive = [...e];
    if (!m.userData.svOrig || !m.userData.svHasOv) m.material.emissive.setRGB(...e);
    for (const [id, [x, z]] of dragged) { const o = meshes.get(id)!; o.position.x = x; o.position.z = z; }
  };
  return {
    scene, camera,
    tick: (step) => { stepToyScene(scene, step * DT); hold(); },
    apply: (a) => {
      if (a.type === "lamp") lamp = lamp === "red" ? "green" : "red";
      else if (a.type === "palette") paint(a.preset);
      else if (a.type === "drag") place(a.id, a.x, a.z);
      else if (a.type === "mission" && a.name === "separate") {
        const b = meshes.get("bush")!.position;
        paint("lookalike"); place("bush", b.x, b.z); place("cone", b.x + 0.9, b.z + 0.2);
      }
      hold();
    },
    facts: () => {
      const a = meshes.get("cone")!.position, b = meshes.get("bush")!.position;
      return { hazards: [{ id: "car", x: meshes.get("car")!.position.x, z: meshes.get("car")!.position.z }], alarms: 0,
        separation: Math.hypot(a.x - b.x, a.z - b.z) };
    },
    dispose: () => {},
  };
}
