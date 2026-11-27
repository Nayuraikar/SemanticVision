// Instrument panel / monitor dashboard (SPEC 6, second environment). HELD-OUT: nothing in the pipeline is tuned on it.
// Annunciator grid, round gauges with range arcs, status lamps, glass displays, legend strips, cascading alarms.
// Deterministic: all randomness from Rng(seed), time = step * DT. Same exportSnapshot contract as every scene.
// Colours were chosen once from panel conventions (14 CFR 25.1322 / 23.1549, IEC 60073): red = warning/limit,
// amber = caution, green = normal, blue/white = advisory; cyan/magenta display symbology. Not adjusted afterwards.
import * as THREE from "three";
import type { Event, Snapshot } from "./protocol.gen";
import { exportSnapshot, type SVMesh, type SVTag } from "./snapshot";
import { DT, FixedStep } from "./physics";
import { Rng } from "./rng";

type V3 = [number, number, number];
export type DashLighting = "normal" | "dim" | "glare";
export const DASH_LIGHTINGS: DashLighting[] = ["normal", "dim", "glare"];
export type DashScenario = "quiet" | "alarm";
export interface DashboardOpts { seed: number; scenario: DashScenario; env: DashLighting; aspect?: number }

export interface DashPreset {
  hemi: { sky: V3; ground: V3; intensity: number };
  key: { rgb: V3; intensity: number; pos: V3 };       // overhead flood / sun through the windscreen
  exposure: number;
  white_point: V3;                                     // XYZ (Y=1)
  dimmer: number;                                      // annunciator / display BRT-DIM setting (scales emission)
}
export const DASH_PRESETS: Record<DashLighting, DashPreset> = {
  normal: {
    hemi: { sky: [0.9, 0.92, 0.95], ground: [0.35, 0.33, 0.3], intensity: 1.1 },
    key: { rgb: [1.0, 0.97, 0.92], intensity: 1.4, pos: [0.5, 2.0, 1.8] },
    exposure: 1.0, white_point: [0.9505, 1.0, 1.089], dimmer: 0.85,
  },
  dim: {                                               // dimmed cabin, warm ~3500 K flood, lamps on DIM
    hemi: { sky: [1.0, 0.76, 0.52], ground: [0.2, 0.14, 0.09], intensity: 0.16 },
    key: { rgb: [1.0, 0.74, 0.48], intensity: 0.22, pos: [-0.6, 1.8, 1.2] },
    exposure: 1.3, white_point: [1.0396, 1.0, 0.5329], dimmer: 0.5,
  },
  glare: {                                             // direct low sun on the panel (~D55), lamps on BRT
    hemi: { sky: [0.8, 0.88, 1.0], ground: [0.45, 0.4, 0.32], intensity: 1.8 },
    key: { rgb: [1.0, 0.95, 0.86], intensity: 6.0, pos: [1.2, 1.1, 1.4] },
    exposure: 0.8, white_point: [0.9568, 1.0, 0.9214], dimmer: 1.0,
  },
};

export type Level = "warning" | "caution" | "normal";
export const SYSTEMS = ["eng", "hyd", "elec", "fuel", "env", "avnx"] as const;
export type System = (typeof SYSTEMS)[number];
export const GAUGE_SYSTEMS: System[] = ["eng", "hyd", "elec", "fuel"];
/** a fault in the key system propagates cautions to these systems, in this order */
export const DEPENDS: Record<System, [System, System]> = {
  eng: ["elec", "hyd"], hyd: ["eng", "avnx"], elec: ["avnx", "env"],
  fuel: ["eng", "hyd"], env: ["avnx", "elec"], avnx: ["elec", "env"],
};
/** annunciator rows, top to bottom: fixed legend colour per tile, as on a real panel */
export const ROWS: Level[] = ["warning", "caution", "caution", "normal", "normal"];

export const LAMP_EMIT: Record<Level, V3> = { warning: [1.0, 0.03, 0.02], caution: [1.0, 0.45, 0.0], normal: [0.03, 0.85, 0.2] };
export const LAMP_LENS: Record<Level, V3> = { warning: [0.09, 0.025, 0.022], caution: [0.1, 0.065, 0.022], normal: [0.022, 0.075, 0.035] };
const LIT_NAME: Record<Level, string> = { warning: "red", caution: "amber", normal: "green" };
const BLUE: V3 = [0.03, 0.22, 1.0], WHITE: V3 = [0.8, 0.8, 0.77], CYAN: V3 = [0.0, 0.75, 0.8], MAGENTA: V3 = [0.8, 0.05, 0.7];
const PANEL: V3 = [0.055, 0.058, 0.062], DIAL: V3 = [0.012, 0.012, 0.014], GLASS: V3 = [0.015, 0.018, 0.022];
const ALU: V3 = [0.52, 0.53, 0.54], GLARESHIELD: V3 = [0.018, 0.016, 0.015], LEGEND: V3 = [0.5, 0.5, 0.47];
const ZERO: V3 = [0, 0, 0];
const ARC_GAIN = 0.7, LEGEND_GLOW = 0.12, FLASH_HZ = 2;

// ---- layout (panel plane z = 0, camera on +z). Tiles abut (1 mm overlap) like a real annunciator matrix.
export const GRID = { cols: SYSTEMS.length, rows: ROWS.length, pitchX: 0.19, pitchY: 0.085, top: 0.55 };
const GAUGE_X = [-0.9, -0.3, 0.3, 0.9], GAUGE_Y = -0.18, GAUGE_R = 0.215;
export const ARC_START = (210 * Math.PI) / 180, ARC_SPAN = (240 * Math.PI) / 180;   // value 0 -> 210 deg, 1 -> -30 deg
export interface GaugeDef { sys: System; nominal: number; bad: "high" | "low" }
const GAUGES: GaugeDef[] = [
  { sys: "eng", nominal: 0.45, bad: "high" },     // exhaust gas temperature
  { sys: "hyd", nominal: 0.72, bad: "low" },      // hydraulic pressure
  { sys: "elec", nominal: 0.7, bad: "low" },      // bus voltage
  { sys: "fuel", nominal: 0.66, bad: "low" },     // fuel quantity / feed pressure
];
/** [from, to, level] as fractions of the scale */
const BANDS: Record<"high" | "low", [number, number, Level][]> = {
  high: [[0, 0.65, "normal"], [0.65, 0.85, "caution"], [0.85, 1, "warning"]],
  low: [[0, 0.15, "warning"], [0.15, 0.35, "caution"], [0.35, 1, "normal"]],
};
export const bandOf = (bad: "high" | "low", v: number): Level =>
  (BANDS[bad].find(([a, b]) => v >= a && v < b) ?? BANDS[bad][BANDS[bad].length - 1])[2];

export interface Cascade {
  system: System; related: [System, System];
  /** fixed-step indices: first caution, warning escalation (+ first related caution), second related caution */
  caution: number; warning: number; spread: number;
}

export interface Lamp { mesh: SVMesh; sys: System | null; level: Level; on: boolean; flash: boolean; emit: V3 }
export interface Gauge { def: GaugeDef; needle: SVMesh; readout: SVMesh; arcs: SVMesh[]; face: SVMesh; v: number; phase: number; rate: number }

function mesh(geo: THREE.BufferGeometry, albedo: V3, sv: SVTag, rough = 0.7, metal = 0): SVMesh {
  const mat = new THREE.MeshStandardMaterial({ roughness: rough, metalness: metal });
  mat.color.setRGB(...albedo);
  const m = new THREE.Mesh(geo, mat) as SVMesh;
  m.userData.sv = { tags: [], ...sv };
  return m;
}
/** Authored emissive change that survives applyOverrides. */
function setEmissive(m: SVMesh, e: V3, k = 1) {
  const s: V3 = [e[0] * k, e[1] * k, e[2] * k];
  if (m.userData.svOrig) m.userData.svOrig.emissive = [...s];
  if (!m.userData.svOrig || !m.userData.svHasOv) { m.material.emissive.setRGB(...s); m.material.emissiveIntensity = 1; }
}

export class DashboardSim {
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly opts: DashboardOpts;
  stepper = new FixedStep(DT);
  step = 0;
  lighting: DashLighting;
  cascade: Cascade | null = null;
  protected lamps = new Map<string, Lamp>();
  protected gauges: Gauge[] = [];
  private statics: { mesh: SVMesh; emit: V3; k: number }[] = [];   // emissive symbology scaled by the dimmer only
  private lights: THREE.Object3D[] = [];

  constructor(opts: DashboardOpts) {
    this.opts = opts;
    this.lighting = opts.env;
    this.camera = new THREE.PerspectiveCamera(40, opts.aspect ?? 16 / 9, 0.1, 20);
    this.camera.position.set(0, 0.12, 2.45);
    this.camera.lookAt(0, 0, 0);
    this.camera.updateMatrixWorld(true);
    this.scene.userData.svDash = this;
    this.scene.userData.svEvents = [];
    this.build();
    this.reset();
  }

  get time() { return this.step * DT; }

  private put<T extends THREE.Object3D>(m: T, p: V3): T { m.position.set(...p); this.scene.add(m); return m; }

  private box(id: string, cls: string, size: V3, p: V3, albedo: V3, rough = 0.7, metal = 0, state = "static"): SVMesh {
    return this.put(mesh(new THREE.BoxGeometry(...size), albedo, { id, cls, state }, rough, metal), p);
  }

  private glow(m: SVMesh, emit: V3, k = 1) { this.statics.push({ mesh: m, emit, k }); return m; }

  private legend(id: string, w: number, p: V3) { this.glow(this.box(id, "label", [w, 0.024, 0.006], p, LEGEND), WHITE, LEGEND_GLOW); }

  private lamp(id: string, m: SVMesh, sys: System | null, level: Level, emit: V3 = LAMP_EMIT[level]) {
    this.lamps.set(id, { mesh: m, sys, level, on: false, flash: false, emit });
  }

  private build() {
    this.box("panel", "panel", [2.7, 1.5, 0.04], [0, 0, -0.05], PANEL, 0.8);
    this.box("glareshield", "glareshield", [2.7, 0.07, 0.12], [0, 0.745, 0.0], GLARESHIELD, 0.9);
    this.box("trim_mid", "trim", [2.5, 0.024, 0.02], [0, 0.12, -0.015], ALU, 0.25, 0.3);
    this.box("trim_low", "trim", [2.5, 0.024, 0.02], [0, -0.475, -0.015], ALU, 0.25, 0.3);
    // annunciator matrix
    const x0 = -((GRID.cols - 1) / 2) * GRID.pitchX;
    SYSTEMS.forEach((sys, c) => {
      this.legend(`hdr_${sys}`, 0.13, [x0 + c * GRID.pitchX, 0.625, 0]);
      ROWS.forEach((level, r) => {
        const id = `ann_${sys}_${r}`;
        const m = this.box(id, "annunciator", [GRID.pitchX + 0.001, GRID.pitchY + 0.001, 0.02],
          [x0 + c * GRID.pitchX, GRID.top - r * GRID.pitchY, 0], LAMP_LENS[level], 0.35, 0, "off");
        this.lamp(id, m, sys, level);
      });
    });
    // master warning / caution pushbuttons, side by side
    (["warning", "caution"] as Level[]).forEach((level, i) => {
      const id = `master_${level}`;
      this.lamp(id, this.box(id, "pushbutton", [0.161, 0.11, 0.03], [-0.94 + 0.16 * i, 0.5, 0], LAMP_LENS[level], 0.35, 0, "off"), null, level);
    });
    this.legend("master_legend", 0.3, [-0.86, 0.42, 0]);
    // advisory lamps (blue = system selected on, white = neutral status)
    ([["adv_antiice", BLUE], ["adv_apu", WHITE]] as [string, V3][]).forEach(([id, c], i) => {
      const m = this.box(id, "advisory", [0.151, 0.065, 0.02], [-0.935 + 0.15 * i, 0.3, 0], [0.03, 0.035, 0.05], 0.35, 0, "on");
      this.glow(m, c);
    });
    // per-system status lamps with legend strips
    SYSTEMS.forEach((sys, i) => {
      const id = `status_${sys}`, y = 0.585 - 0.076 * i;
      const m = mesh(new THREE.SphereGeometry(0.03, 16, 12), LAMP_LENS.normal, { id, cls: "status_lamp", state: "off" }, 0.3);
      this.put(m, [0.76, y, 0.01]);
      this.lamp(id, m, sys, "normal");
      this.legend(`status_${sys}_legend`, 0.2, [0.89, y, 0]);
    });
    // gauges: dial face, coloured range arcs, white needle, digital readout, legend
    GAUGES.forEach((def, i) => {
      const x = GAUGE_X[i], id = `g_${def.sys}`;
      const face = this.put(mesh(new THREE.CircleGeometry(GAUGE_R, 48), DIAL, { id: `${id}_face`, cls: "dial", state: "static" }, 0.5), [x, GAUGE_Y, 0]);
      const arcs = BANDS[def.bad].map(([a, b, level]) => {
        const geo = new THREE.RingGeometry(0.155, 0.19, Math.max(4, Math.round(48 * (b - a))), 1, ARC_START - ARC_SPAN * b, ARC_SPAN * (b - a));
        const m = this.put(mesh(geo, LAMP_LENS[level], { id: `${id}_arc_${level}`, cls: "gauge_arc", state: level }, 0.5), [x, GAUGE_Y, 0.004]);
        return this.glow(m, LAMP_EMIT[level], ARC_GAIN);
      });
      const needle = mesh(new THREE.BoxGeometry(0.175, 0.013, 0.006).translate(0.0875, 0, 0), [0.6, 0.6, 0.58], { id: `${id}_needle`, cls: "needle", state: "normal" }, 0.4);
      this.put(needle, [x, GAUGE_Y, 0.012]);
      this.glow(needle, WHITE, 0.6);
      const readout = this.box(`${id}_readout`, "readout", [0.14, 0.05, 0.006], [x, GAUGE_Y - 0.105, 0.008], DIAL, 0.3, 0, "normal");
      this.legend(`${id}_legend`, 0.24, [x, GAUGE_Y - 0.25, 0]);
      this.gauges.push({ def, needle, readout, arcs, face, v: def.nominal, phase: 0, rate: 0 });
    });
    // glass monitor displays with symbology traces
    [-0.62, 0.62].forEach((x, i) => {
      this.glow(this.box(`mfd${i}_glass`, "display", [1.08, 0.19, 0.012], [x, -0.605, -0.01], GLASS, 0.08), [0.02, 0.03, 0.045]);
      ([["cyan", CYAN, 0.0], ["magenta", MAGENTA, -0.055], ["white", WHITE, 0.055]] as [string, V3, number][]).forEach(([n, c, dy], j) => {
        const w = 0.5 - 0.09 * j;
        this.glow(this.box(`mfd${i}_trace_${n}`, "symbology", [w, 0.018, 0.004], [x - 0.45 + w / 2, -0.605 + dy, 0], DIAL), c, 0.8);
      });
    });
  }

  setLighting(name: DashLighting) {
    const p = DASH_PRESETS[name];
    for (const l of this.lights) this.scene.remove(l);
    const hemi = new THREE.HemisphereLight(new THREE.Color(...p.hemi.sky), new THREE.Color(...p.hemi.ground), p.hemi.intensity);
    const key = new THREE.DirectionalLight(new THREE.Color(...p.key.rgb), p.key.intensity);
    key.position.set(...p.key.pos);
    this.lights = [hemi, key, key.target];
    this.scene.add(...this.lights);
    this.scene.background = new THREE.Color(0, 0, 0);
    this.scene.userData.svEnv = { name, exposure: p.exposure, white_point: [...p.white_point] };
    if (name !== this.lighting) this.emit({ type: "lighting_change", data: { from: this.lighting, to: name, step: this.step } });
    this.lighting = name;
    this.refresh();
  }

  reset() {
    this.step = 0; this.stepper = new FixedStep(DT); this.cascade = null;
    this.scene.userData.svEvents = [];
    const rng = new Rng(this.opts.seed * 6007 + (this.opts.scenario === "alarm" ? 1 : 0));
    for (const g of this.gauges) { g.phase = rng.uniform(0, 2 * Math.PI); g.rate = rng.uniform(0.15, 0.45); g.v = g.def.nominal; }
    for (const l of this.lamps.values()) {
      l.on = l.level === "normal" && l.sys !== null;           // every "normal" tile and status lamp lit
      l.flash = false;
      l.mesh.userData.sv.tags = ["signal"];
      l.mesh.userData.sv.state = l.on ? LIT_NAME[l.level] : "off";
    }
    for (const g of this.gauges) {
      for (const m of [g.needle, g.readout, g.face, ...g.arcs]) m.userData.sv.tags = [];
      g.needle.userData.sv.state = g.readout.userData.sv.state = "normal";
    }
    if (this.opts.scenario === "alarm") {
      const system = rng.pick(GAUGE_SYSTEMS);
      const sec = (s: number) => Math.round(s / DT);
      const caution = sec(rng.uniform(2.0, 3.0));
      const warning = caution + sec(rng.uniform(1.0, 1.8));
      this.cascade = { system, related: DEPENDS[system], caution, warning, spread: warning + sec(rng.uniform(0.5, 1.2)) };
      // one standing caution in an unrelated system (deferred maintenance item), as on real panels
      const free = SYSTEMS.filter((s) => s !== system && !DEPENDS[system].includes(s));
      const s = this.lamps.get(`ann_${rng.pick(free)}_2`)!;
      s.on = true; s.mesh.userData.sv.state = "amber";
    }
    this.setLighting(this.lighting);
    this.updateGauges();
  }

  private emit(e: Event) { (this.scene.userData.svEvents as Event[]).push(e); }

  protected setState(m: SVMesh, s: string) {
    const sv = m.userData.sv as SVTag;
    if (sv.state === s) return;
    this.emit({ type: "state_change", object_id: sv.id, data: { from: sv.state ?? "default", to: s, step: this.step } });
    sv.state = s;
  }

  private hazard(m: SVMesh) {
    const sv = m.userData.sv as SVTag;
    sv.tags = [...new Set([...(sv.tags ?? []), "hazard"])].sort();
  }

  protected light(id: string, on: boolean, flash = false) {
    const l = this.lamps.get(id)!;
    l.on = on; l.flash = flash;
    this.setState(l.mesh, on ? LIT_NAME[l.level] : "off");
  }

  /** status lamp: one lens whose colour follows the system's worst level */
  private status(sys: System, level: Level) {
    const l = this.lamps.get(`status_${sys}`)!;
    l.level = level; l.emit = LAMP_EMIT[level];
    l.mesh.material.color.setRGB(...LAMP_LENS[level]);
    delete l.mesh.userData.svOrig;                             // authored colours changed
    this.setState(l.mesh, LIT_NAME[level]);
    this.hazard(l.mesh);
  }

  private degrade(sys: System, row: number) {
    this.light(`ann_${sys}_${row}`, true);
    this.light(`ann_${sys}_3`, false);                         // "normal" legend extinguishes
    this.hazard(this.lamps.get(`ann_${sys}_${row}`)!.mesh);
    this.hazard(this.lamps.get(`ann_${sys}_3`)!.mesh);
  }

  private fire() {
    const c = this.cascade;
    if (!c) return;
    const g = this.gauges.find((x) => x.def.sys === c.system);   // none for systems without a gauge (play mode)
    if (this.step === c.caution) {
      this.degrade(c.system, 1);
      this.status(c.system, "caution");
      this.light("master_caution", true, true);
      this.hazard(this.lamps.get("master_caution")!.mesh);
      if (g) for (const m of [g.needle, g.readout, g.face, ...g.arcs]) this.hazard(m);
    }
    if (this.step === c.warning) {
      ROWS.forEach((_, r) => this.hazard(this.lamps.get(`ann_${c.system}_${r}`)!.mesh));   // whole faulted column
      this.light(`ann_${c.system}_0`, true, true);
      this.status(c.system, "warning");
      this.light("master_warning", true, true);
      this.hazard(this.lamps.get("master_warning")!.mesh);
      this.degrade(c.related[0], 1);
      this.status(c.related[0], "caution");
    }
    if (this.step === c.spread) {
      this.degrade(c.related[1], 2);
      this.status(c.related[1], "caution");
    }
  }

  /** Displayed gauge value; play mode overrides it. */
  protected gaugeValue(_g: Gauge, v: number): number { return v; }

  protected updateGauges() {
    const c = this.cascade;
    for (const g of this.gauges) {
      let v = g.def.nominal + 0.025 * Math.sin(g.phase + 2 * Math.PI * g.rate * this.time);
      if (c && c.system === g.def.sys && this.step >= c.caution) {
        // drift: into the caution band at the first alarm, into the red band at the escalation
        const [amber, red] = g.def.bad === "high" ? [0.74, 0.93] : [0.25, 0.08];
        const k1 = Math.min(1, (this.step - c.caution + 1) / 12);
        v = g.def.nominal + k1 * (amber - g.def.nominal);
        if (this.step >= c.warning) v = amber + Math.min(1, (this.step - c.warning + 1) / 3) * (red - amber);
      }
      v = this.gaugeValue(g, v);
      g.v = v;
      g.needle.rotation.z = ARC_START - ARC_SPAN * v;
      const band = bandOf(g.def.bad, v);
      this.setState(g.needle, band);
      this.setState(g.readout, band);
    }
    this.refresh();
  }

  /** authored emission for the current state, dimmer setting and flash phase */
  private refresh() {
    const k = DASH_PRESETS[this.lighting].dimmer;
    const flash = Math.floor(this.time * FLASH_HZ * 2) % 2 === 0;
    const since = this.cascade ? this.step - this.cascade.warning : -1;
    for (const l of this.lamps.values()) {
      // flashing lamps start lit at their onset, then follow the common flasher
      const lit = l.on && (!l.flash || flash || (since >= 0 && since < 12));
      setEmissive(l.mesh, lit ? l.emit : ZERO, k);
    }
    for (const s of this.statics) setEmissive(s.mesh, s.emit, s.k * k);
    for (const g of this.gauges) {                             // readout digits take the colour of the band
      const band = g.readout.userData.sv.state as Level;
      setEmissive(g.readout, LAMP_EMIT[band] ?? ZERO, ARC_GAIN * k);
    }
  }

  /** One fixed step (DT). */
  tick() {
    this.step++;
    this.fire();
    this.updateGauges();
  }

  advance(elapsed: number) { return this.stepper.advance(elapsed, () => this.tick()); }
}

export interface DashRecordOpts extends DashboardOpts {
  steps: number; exportEvery: number; observerId?: string; budgetMs?: number;
  /** first step to export (earlier events are dropped); frame numbers stay global */
  startStep?: number;
}

/** Headless recorder (same pattern and exportSnapshot as recordTraffic). */
export async function recordDashboard(o: DashRecordOpts, sink?: (s: Snapshot) => void): Promise<{ snaps: Snapshot[]; sim: DashboardSim }> {
  const sim = new DashboardSim(o);
  const prev = new Map<string, [number, number]>();
  const snaps: Snapshot[] = [];
  let prevT: number | undefined;
  for (let s = 1; s <= o.steps; s++) {
    sim.tick();
    if (s % o.exportEvery) continue;
    if (s < (o.startStep ?? 0)) { sim.scene.userData.svEvents = []; continue; }
    const t = sim.time;
    const snap = exportSnapshot(sim.scene, sim.camera, { frame: s / o.exportEvery - 1, t, observerId: o.observerId ?? "tr-0-000", budgetMs: o.budgetMs ?? 40, prev, prevT });
    snap.session_id = `dashboard-${o.scenario}-${o.env}-${o.seed}`;
    prevT = t;
    if (sink) sink(snap); else snaps.push(snap);
  }
  return { snaps, sim };
}
