// Traffic intersection (SPEC 6): Rapier physics, fixed timestep, seeded scenarios, physics events -> snapshot events.
// Collision => involved vehicles damaged + hazard flashers, beacon + smoke marker, hazard tags on involved and nearby.
// Deterministic: all randomness from Rng(seed), time = step * DT. Engine code lives only in adapters/.
import * as THREE from "three";
import type { Event, Snapshot } from "./protocol.gen";
import { exportSnapshot, type SVMesh, type SVTag } from "./snapshot";
import { DT, FixedStep, G_PED, G_VEH, Physics, initPhysics, physicsReady, type PhysEvent, type V3 } from "./physics";
import { Rng } from "./rng";
import { applyEnvironment, type EnvName } from "./environments";

export type Scenario = "normal" | "collision";
export interface TrafficOpts { seed: number; scenario: Scenario; env: EnvName; aspect?: number }

// ---- layout (metres). Intersection at origin; roads 8 m wide (2 lanes); right-hand traffic.
const LANE = 2, ROAD_HALF = 4, CW = 6, STOP_U = -7.5, SPAWN_U = -50, DESPAWN_U = 50;
const VEH_HALF = { len: 2.1, h: 0.7, w: 0.95 };
const A_UP = 2.5, A_DOWN = 6, A_PLAN = 3;
export const NEAR_R = 15;          // nearby radius for the hazard / importance bump (crash scene incl. stop-line queues)
const FLASH_HZ = 1.5;

type Axis = "ew" | "ns";
interface Approach { name: "E" | "W" | "S" | "N"; dir: V3; right: V3; axis: Axis }
export const APPROACHES: Approach[] = [
  { name: "E", dir: [1, 0, 0], right: [0, 0, 1], axis: "ew" },
  { name: "W", dir: [-1, 0, 0], right: [0, 0, -1], axis: "ew" },
  { name: "S", dir: [0, 0, 1], right: [-1, 0, 0], axis: "ns" },
  { name: "N", dir: [0, 0, -1], right: [1, 0, 0], axis: "ns" },
];
const AP = (n: Approach["name"]) => APPROACHES.find((a) => a.name === n)!;

type Light = "red" | "amber" | "green";
const CYCLE: [number, Light, Light][] = [[8, "green", "red"], [2, "amber", "red"], [1, "red", "red"], [8, "red", "green"], [2, "red", "amber"], [1, "red", "red"]];
const CYCLE_S = CYCLE.reduce((s, c) => s + c[0], 0);
/** Signal state per axis at time t (s). Also seconds elapsed in the current phase. */
export function signalAt(t: number): { ew: Light; ns: Light; phase: number; inPhase: number } {
  let x = ((t % CYCLE_S) + CYCLE_S) % CYCLE_S;
  for (let i = 0; i < CYCLE.length; i++) {
    if (x < CYCLE[i][0]) return { ew: CYCLE[i][1], ns: CYCLE[i][2], phase: i, inPhase: x };
    x -= CYCLE[i][0];
  }
  return { ew: "red", ns: "red", phase: 5, inPhase: 0 };
}

// Lamp emission in display-linear sRGB at ECE/SAE chromaticities (red x.69 y.31, amber x.575 y.425, signal green
// bluish), peak channel 1.0 so day exposure keeps the hue instead of clipping amber to yellow / green to cyan.
const LAMP_EMIT: Record<Light, V3> = { red: [1.0, 0.03, 0.02], amber: [1.0, 0.38, 0.0], green: [0.02, 0.85, 0.4] };
const LAMP_LENS: Record<Light, V3> = { red: [0.25, 0.02, 0.02], amber: [0.3, 0.18, 0.02], green: [0.02, 0.2, 0.08] };
const BRAKE_ON: V3 = [1.0, 0.03, 0.02], AMBER_ON: V3 = [1.0, 0.38, 0.0], ZERO: V3 = [0, 0, 0];
const BRAKE_LENS: V3 = [0.3, 0.02, 0.02], AMBER_LENS: V3 = [0.35, 0.2, 0.02];
// Real-world colour conventions (linear albedo). Car paints: red, maroon/brown, dark green, orange, taxi yellow
// next to the common white/silver/black/blue. Safety orange cones (~sRGB 255,100,0), fluorescent hi-vis orange and
// yellow-green vests (EN ISO 20471), privet/box hedge green.
const CAR_COLOURS: V3[] = [
  [0.55, 0.04, 0.03], [0.08, 0.3, 0.06], [0.22, 0.08, 0.03], [0.3, 0.12, 0.04],
  [0.05, 0.1, 0.45], [0.7, 0.7, 0.68], [0.25, 0.25, 0.26], [0.7, 0.5, 0.03], [0.75, 0.2, 0.02], [0.03, 0.03, 0.03],
];
const CLOTHES: V3[] = [[0.6, 0.05, 0.05], [0.1, 0.35, 0.08], [0.45, 0.35, 0.05], [0.1, 0.12, 0.5], [0.5, 0.5, 0.5],
  [0.2, 0.22, 0.07], [0.4, 0.33, 0.18], [0.22, 0.1, 0.04], [0.3, 0.03, 0.05], [0.7, 0.2, 0.03]];   // + olive, khaki, brown, burgundy, orange
const BYSTANDERS: [number, number][] = [[-7.6, -4.7], [-4.7, -7.8], [7.6, -4.7], [-7.6, 4.7], [-4.7, 7.6], [4.8, -7.8]];
const CONE: V3 = [0.8, 0.12, 0.01], HIVIS_ORANGE: V3 = [0.9, 0.18, 0.02], HIVIS_YELLOW: V3 = [0.6, 0.85, 0.02];
const HEDGE: V3 = [0.05, 0.18, 0.03], WORKWEAR: V3 = [0.03, 0.04, 0.12];

const add3 = (a: V3, b: V3, k = 1): V3 => [a[0] + k * b[0], a[1] + k * b[1], a[2] + k * b[2]];
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const scl = (a: V3, k: number): V3 => [a[0] * k, a[1] * k, a[2] * k];

function mesh(geo: THREE.BufferGeometry, albedo: V3, sv?: SVTag, emissive?: V3, rough = 0.7): SVMesh {
  const mat = new THREE.MeshStandardMaterial({ roughness: rough, metalness: 0 });
  mat.color.setRGB(...albedo);
  if (emissive) { mat.emissive.setRGB(...emissive); mat.emissiveIntensity = 1; }
  const m = new THREE.Mesh(geo, mat) as SVMesh;
  if (sv) m.userData.sv = { tags: [], ...sv };
  return m;
}

/** Authored emissive change that survives applyOverrides (toy-scene pattern). */
function setEmissive(m: SVMesh, e: V3) {
  if (m.userData.svOrig) m.userData.svOrig.emissive = [...e];
  if (!m.userData.svOrig || !m.userData.svHasOv) { m.material.emissive.setRGB(...e); m.material.emissiveIntensity = 1; }
}

interface Vehicle {
  id: string; ap: Approach; mesh: SVMesh; brake: SVMesh; ind: SVMesh; indR: SVMesh; half: V3;
  v: number; cruise: number; runner: boolean; ignoreObs: boolean; halt: boolean; damaged: boolean; hazard: boolean;
}
interface Ped {
  id: string; mesh: SVMesh; head: SVMesh; cw: number; side: number; speed: number; crossing: boolean; halt: boolean;
  cycleDone: number;
}
interface Spawn { t: number; ap: Approach; speed: number; colour: V3; runner?: boolean; ignoreObs?: boolean }

// crosswalk k: across the NS road at z=+-CW (walk along x) or across the EW road at x=+-CW (walk along z)
const CROSSWALKS = [
  { id: "cw_n", across: "ns" as Axis, c: -CW }, { id: "cw_s", across: "ns" as Axis, c: CW },
  { id: "cw_w", across: "ew" as Axis, c: -CW }, { id: "cw_e", across: "ew" as Axis, c: CW },
];

export class TrafficSim {
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly opts: TrafficOpts;
  physics: Physics | null = null;
  stepper = new FixedStep(DT);
  step = 0;
  /** all physics events since reset (for tests/bench) */
  log: PhysEvent[] = [];
  collisionPoint: V3 | null = null;
  private rng!: Rng;
  private offset = 0;
  private vehicles: Vehicle[] = [];
  private peds: Ped[] = [];
  private spawns: Spawn[] = [];
  private lamps = new Map<string, { mesh: SVMesh; axis: Axis; colour: Light }>();
  private collided = new Set<string>();
  private hazardObjs: SVMesh[] = [];
  private workers: SVMesh[] = [];
  private bystanders: SVMesh[] = [];
  private worksLamps: SVMesh[] = [];
  private nextVeh = 0;
  private lastLight: Record<Axis, Light | null> = { ew: null, ns: null };

  constructor(opts: TrafficOpts) {
    this.opts = opts;
    this.camera = new THREE.PerspectiveCamera(50, opts.aspect ?? 16 / 9, 0.1, 400);
    this.camera.position.set(16, 11, 24);
    this.camera.lookAt(0, 1.5, 0);
    this.camera.updateMatrixWorld(true);
    this.scene.userData.svSim = this;
    this.scene.userData.svEvents = [];
    this.buildStatic();
    applyEnvironment(this.scene, opts.env);
    if (physicsReady()) this.reset();
  }

  get time() { return this.step * DT; }

  // ---------------------------------------------------------------- static world
  private buildStatic() {
    const S = this.scene;
    const put = (m: THREE.Object3D, p: V3) => { m.position.set(...p); S.add(m); return m; };
    const ground = mesh(new THREE.BoxGeometry(240, 0.02, 240), [0.08, 0.2, 0.05], { id: "ground", cls: "grass" });
    put(ground, [0, -0.02, 0]);
    for (const [id, geo] of [["road_ew", new THREE.BoxGeometry(240, 0.02, 2 * ROAD_HALF)], ["road_ns", new THREE.BoxGeometry(2 * ROAD_HALF, 0.02, 240)]] as const) {
      const r = mesh(geo, [0.07, 0.07, 0.075], { id, cls: "road" }, undefined, 0.9);
      r.userData.svEnvRole = "road";
      put(r, [0, id === "road_ew" ? 0.005 : 0.006, 0]);
    }
    let k = 0;
    for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
      put(mesh(new THREE.BoxGeometry(56, 0.15, 56), [0.32, 0.31, 0.3], { id: `sidewalk_${k++}`, cls: "sidewalk" }), [sx * 32, 0.075, sz * 32]);
    }
    put(mesh(new THREE.BoxGeometry(16, 0.04, 16), [0.09, 0.28, 0.05], { id: "lawn_se", cls: "grass" }), [18, 0.17, 18]);
    put(mesh(new THREE.BoxGeometry(16, 0.04, 16), [0.09, 0.28, 0.05], { id: "lawn_nw", cls: "grass" }), [-24, 0.17, -14]);
    const bld: [string, V3, V3, V3][] = [
      ["bld_nw", [-20, 0, -26], [16, 18, 14], [0.42, 0.2, 0.12]],
      ["bld_ne", [20, 0, -20], [14, 24, 16], [0.5, 0.48, 0.44]],
      ["bld_sw", [-20, 0, 20], [14, 12, 14], [0.35, 0.3, 0.22]],
    ];
    for (const [id, p, s, c] of bld) put(mesh(new THREE.BoxGeometry(...s), c, { id, cls: "building" }), [p[0], s[1] / 2 + 0.15, p[2]]);
    for (const [id, p] of [["tree_se", [12, 0, 12]], ["tree_nw", [-12, 0, -11]]] as [string, V3][]) {
      put(mesh(new THREE.CylinderGeometry(0.25, 0.3, 3, 10), [0.2, 0.12, 0.06], { id: `${id}_trunk`, cls: "bark" }), [p[0], 1.65, p[2]]);
      put(mesh(new THREE.SphereGeometry(2, 16, 12), [0.06, 0.25, 0.04], { id: `${id}_canopy`, cls: "vegetation" }), [p[0], 4.5, p[2]]);
    }
    // hedges / bushes on the corners
    put(mesh(new THREE.BoxGeometry(5, 1.1, 0.9), HEDGE, { id: "hedge_se_e", cls: "vegetation" }), [13, 0.7, 5.6]);
    put(mesh(new THREE.BoxGeometry(0.9, 1.1, 6.5), HEDGE, { id: "hedge_se_s", cls: "vegetation" }), [5.2, 0.7, 8.75]);
    // kerbside hedges bordering the NW / NE / SW corner plots (behind the junction from the camera)
    const hedges: [string, V3, V3][] = [
      ["hedge_nw_e", [-11.5, 0.75, -6], [8, 1.2, 0.9]], ["hedge_nw_n", [-6, 0.75, -11.5], [0.9, 1.2, 8]],
      ["hedge_ne_w", [11.5, 0.75, -6], [8, 1.2, 0.9]], ["hedge_sw_e", [-11.5, 0.75, 6], [8, 1.2, 0.9]],
    ];
    for (const [id, p, s] of hedges) put(mesh(new THREE.BoxGeometry(...s), HEDGE, { id, cls: "vegetation" }), p);
    // utility works on the SE corner: cones along the kerb, flashing amber barrier lamp, two hi-vis workers
    [[4.6, 4.6], [6, 4.5], [7.5, 4.5], [9, 4.5], [10.5, 4.5], [12, 4.5]].forEach(([x, z], i) => {
      put(mesh(new THREE.CylinderGeometry(0.05, 0.2, 0.7, 12), CONE, { id: `cone_${i}`, cls: "cone", tags: ["roadworks"] }), [x, 0.5, z]);
    });
    const barrier = put(mesh(new THREE.BoxGeometry(2.2, 0.25, 0.08), [0.7, 0.7, 0.68], { id: "works_barrier", cls: "barrier", tags: ["roadworks"] }), [7.5, 0.85, 7.5]);
    barrier.rotation.y = Math.PI / 4;
    const lamp = mesh(new THREE.SphereGeometry(0.12, 10, 8), AMBER_LENS, { id: "works_lamp", cls: "light", state: "flashing", tags: ["roadworks", "signal"] }, ZERO);
    put(lamp, [7.5, 1.12, 7.5]);
    this.worksLamps.push(lamp);
    for (const [i, x, z, vest] of [[0, 8.6, 6.4, HIVIS_ORANGE], [1, 6.4, 8.6, HIVIS_YELLOW]] as [number, number, number, V3][]) {
      const legs = mesh(new THREE.CylinderGeometry(0.22, 0.22, 0.85, 10), WORKWEAR, { id: `worker${i}_legs`, cls: "clothing" });
      put(legs, [x, 0.575, z]);
      const torso = mesh(new THREE.CylinderGeometry(0.3, 0.28, 0.65, 10), vest, { id: `worker${i}`, cls: "worker", state: "working", tags: ["roadworks", "vulnerable"] });
      put(torso, [x, 1.33, z]);
      const head = mesh(new THREE.SphereGeometry(0.17, 12, 8), [0.55, 0.36, 0.26], { id: `worker${i}_head`, cls: "skin" });
      put(head, [x, 1.84, z]);
      this.workers.push(torso);
    }
    // bystanders standing on the corners (clothes re-drawn per seed in reset)
    BYSTANDERS.forEach(([x, z], i) => {
      const body = mesh(new THREE.CylinderGeometry(0.27, 0.27, 1.25, 10), CLOTHES[0], { id: `bystander${i}`, cls: "pedestrian", state: "standing", tags: ["vulnerable"] });
      const head = mesh(new THREE.SphereGeometry(0.17, 12, 8), [0.55, 0.36, 0.26], { id: `bystander${i}_head`, cls: "skin" });
      head.position.set(0, 0.82, 0);
      body.add(head);
      put(body, [x, 0.775, z]);
      this.bystanders.push(body);
    });
    const yel: V3 = [0.7, 0.5, 0.03];
    put(mesh(new THREE.BoxGeometry(100, 0.02, 0.2), yel, { id: "mark_e", cls: "road_marking" }), [ROAD_HALF + 4 + 50, 0.02, 0]);
    put(mesh(new THREE.BoxGeometry(100, 0.02, 0.2), yel, { id: "mark_w", cls: "road_marking" }), [-(ROAD_HALF + 4 + 50), 0.02, 0]);
    put(mesh(new THREE.BoxGeometry(0.2, 0.02, 100), yel, { id: "mark_s", cls: "road_marking" }), [0, 0.02, ROAD_HALF + 4 + 50]);
    put(mesh(new THREE.BoxGeometry(0.2, 0.02, 100), yel, { id: "mark_n", cls: "road_marking" }), [0, 0.02, -(ROAD_HALF + 4 + 50)]);
    for (const cw of CROSSWALKS) {
      const geo = cw.across === "ns" ? new THREE.BoxGeometry(2 * ROAD_HALF, 0.02, 2) : new THREE.BoxGeometry(2, 0.02, 2 * ROAD_HALF);
      put(mesh(geo, [0.75, 0.75, 0.72], { id: cw.id, cls: "crosswalk" }), cw.across === "ns" ? [0, 0.025, cw.c] : [cw.c, 0.025, 0]);
    }
    const sky = mesh(new THREE.PlaneGeometry(600, 200), [0, 0, 0], { id: "sky", cls: "sky" }, [0.35, 0.55, 0.95]);
    sky.userData.svEnvRole = "sky";
    put(sky, [0, 80, -140]);
    // signal heads: far-right corner of each approach, facing oncoming traffic
    for (const ap of APPROACHES) {
      const base = add3(scl(ap.dir, CW + 0.5), scl(ap.right, ROAD_HALF + 1));
      const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.08, 4.4, 8), new THREE.MeshStandardMaterial({ color: 0x222222 }));
      put(pole, [base[0], 2.2, base[2]]);
      const housing = new THREE.Mesh(new THREE.BoxGeometry(0.5, 1.4, 0.5), new THREE.MeshStandardMaterial({ color: 0x111111 }));
      put(housing, [base[0], 4.9, base[2]]);
      const face = add3(base, scl(ap.dir, -0.3));
      (["red", "amber", "green"] as Light[]).forEach((c, i) => {
        const m = mesh(new THREE.BoxGeometry(0.32, 0.32, 0.32), LAMP_LENS[c], { id: `sig_${ap.name}_${c}`, cls: "light", state: "off", tags: ["signal"] }, ZERO);
        put(m, [face[0], 5.35 - 0.45 * i, face[2]]);
        this.lamps.set(m.userData.sv.id, { mesh: m, axis: ap.axis, colour: c });
      });
    }
  }

  // ---------------------------------------------------------------- scenario
  reset() {
    if (!physicsReady()) throw new Error("initPhysics() not awaited");
    for (const v of this.vehicles) this.scene.remove(v.mesh);
    for (const p of this.peds) this.scene.remove(p.mesh);
    for (const h of this.hazardObjs) this.scene.remove(h);
    this.physics?.free();
    this.physics = new Physics();
    this.vehicles = []; this.peds = []; this.hazardObjs = []; this.collided.clear(); this.log = [];
    this.collisionPoint = null; this.step = 0; this.nextVeh = 0; this.stepper = new FixedStep(DT);
    this.lastLight = { ew: null, ns: null };
    this.scene.userData.svEvents = [];
    for (const w of this.workers) w.userData.sv.tags = ["roadworks", "vulnerable"];
    const rng = (this.rng = new Rng(this.opts.seed * 7919 + (this.opts.scenario === "collision" ? 1 : 0)));
    const P = this.physics;
    for (const cw of CROSSWALKS) {
      P.addSensor(cw.id, cw.across === "ns" ? [0, 0.6, cw.c] : [cw.c, 0.6, 0], cw.across === "ns" ? [ROAD_HALF, 1, 1] : [1, 1, ROAD_HALF]);
    }
    P.addSensor("box_junction", [0, 0.6, 0], [ROAD_HALF, 1, ROAD_HALF]);
    this.spawns = [];
    if (this.opts.scenario === "normal") {
      this.offset = rng.uniform(0, CYCLE_S);
      for (const ap of APPROACHES) {
        let t = rng.uniform(0, 3);
        while (t < 120) {
          this.spawns.push({ t, ap, speed: rng.uniform(8, 11), colour: rng.pick(CAR_COLOURS) });
          t += rng.uniform(3, 7);
        }
      }
    } else {
      // red-light runner (N approach, NS red) hits a victim crossing on green (E approach); both arrive at the
      // conflict point together. Followers/queued cars react to the wreck.
      this.offset = rng.uniform(0, 1);
      const vic = AP("E"), run = AP("N");
      const conflict = add3(scl(vic.right, LANE), scl(run.right, LANE));
      const dV = dot(conflict, vic.dir) - SPAWN_U, dR = dot(conflict, run.dir) - SPAWN_U;
      const sv = rng.uniform(8.5, 10), sr = rng.uniform(11, 13), tv = rng.uniform(0.5, 1.2);
      const tr = tv + dV / sv - dR / sr + rng.uniform(-0.04, 0.04);
      const colours = [...CAR_COLOURS];
      const take = () => colours.splice(rng.int(colours.length), 1)[0];
      this.spawns.push({ t: tv, ap: vic, speed: sv, colour: take(), ignoreObs: true });
      this.spawns.push({ t: tr, ap: run, speed: sr, colour: take(), runner: true, ignoreObs: true });
      this.spawns.push({ t: rng.uniform(0.1, 0.6), ap: AP("S"), speed: rng.uniform(8, 10), colour: take() });
      this.spawns.push({ t: tv + rng.uniform(2.6, 3.2), ap: vic, speed: rng.uniform(8, 9), colour: take() });
      this.spawns.push({ t: tv + dV / sv + rng.uniform(0.3, 0.8), ap: AP("W"), speed: rng.uniform(8, 10), colour: take() });
    }
    this.spawns.sort((a, b) => a.t - b.t || a.ap.name.localeCompare(b.ap.name));
    CROSSWALKS.forEach((cw, k) => {
      const side = rng.next() < 0.5 ? -1 : 1;
      const id = `ped${k}`;
      const body = mesh(new THREE.CylinderGeometry(0.28, 0.28, 1.2, 10), rng.pick(CLOTHES), { id, cls: "pedestrian", state: "waiting", tags: ["vulnerable"] });
      const head = mesh(new THREE.SphereGeometry(0.17, 12, 8), [0.55, 0.36, 0.26], { id: `${id}_head`, cls: "skin" });
      head.position.set(0, 0.8, 0);
      body.add(head);
      const p: Ped = { id, mesh: body, head, cw: k, side, speed: rng.uniform(1.3, 1.6), crossing: false, halt: false, cycleDone: -1 };
      const pos = this.pedPos(p, side * (ROAD_HALF + 1));
      body.position.set(...pos);
      this.scene.add(body);
      P.addMover(id, pos, [0.28, 0.6, 0.28], G_PED);
      this.peds.push(p);
    });
    for (const b of this.bystanders) {        // drawn last: traffic / pedestrian randomness unchanged
      b.userData.sv.tags = ["vulnerable"];
      b.material.color.setRGB(...rng.pick(CLOTHES));
      delete b.userData.svOrig;
    }
    this.updateSignals();
  }

  private pedPos(p: Ped, s: number): V3 {
    const cw = CROSSWALKS[p.cw];
    const off = p.cw % 2 ? 0.4 : -0.4;   // two lanes of the crosswalk, never overlapping sensors of the other walk
    return cw.across === "ns" ? [s, 0.75, cw.c + off] : [cw.c + off, 0.75, s];
  }

  private emit(e: Event) { (this.scene.userData.svEvents as Event[]).push(e); }

  private setState(m: SVMesh, s: string) {
    const sv = m.userData.sv as SVTag;
    if (sv.state === s) return;
    this.emit({ type: "state_change", object_id: sv.id, data: { from: sv.state ?? "default", to: s, step: this.step } });
    sv.state = s;
  }

  private addTags(m: SVMesh, tags: string[]) {
    const sv = m.userData.sv as SVTag;
    sv.tags = [...new Set([...(sv.tags ?? []), ...tags])].sort();
  }

  private spawnVehicle(s: Spawn) {
    const id = `veh${this.nextVeh++}`;
    const ap = s.ap;
    const pos = add3(add3(scl(ap.dir, SPAWN_U), scl(ap.right, LANE)), [0, VEH_HALF.h, 0]);
    const body = mesh(new THREE.BoxGeometry(2 * VEH_HALF.len, 2 * VEH_HALF.h, 2 * VEH_HALF.w), s.colour,
      { id, cls: "car", state: "moving", tags: s.runner ? ["vehicle", "runner"] : ["vehicle"] }, undefined, 0.4);
    body.rotation.y = Math.atan2(-ap.dir[2], ap.dir[0]);
    const brake = mesh(new THREE.BoxGeometry(0.06, 0.2, 1.5), BRAKE_LENS, { id: `${id}_brake`, cls: "indicator", state: "off", tags: ["signal"] }, ZERO);
    brake.position.set(-VEH_HALF.len - 0.04, 0.25, 0);
    const ind = mesh(new THREE.BoxGeometry(0.06, 0.14, 1.7), AMBER_LENS, { id: `${id}_ind`, cls: "indicator", state: "off", tags: ["signal"] }, ZERO);
    ind.position.set(VEH_HALF.len + 0.04, 0.2, 0);
    // rear amber indicator directly below the brake light (rear lamp cluster); flashes with the front one
    const indR = mesh(new THREE.BoxGeometry(0.06, 0.1, 1.5), AMBER_LENS, { id: `${id}_ind_r`, cls: "indicator", state: "off", tags: ["signal"] }, ZERO);
    indR.position.set(-VEH_HALF.len - 0.04, 0.1, 0);
    body.add(brake, ind, indR);
    body.position.set(...pos);
    this.scene.add(body);
    const half: V3 = ap.axis === "ew" ? [VEH_HALF.len, VEH_HALF.h, VEH_HALF.w] : [VEH_HALF.w, VEH_HALF.h, VEH_HALF.len];
    const rb = this.physics!.addMover(id, pos, half, G_VEH);
    rb.setLinvel({ x: ap.dir[0] * s.speed, y: 0, z: ap.dir[2] * s.speed }, true);
    this.vehicles.push({ id, ap, mesh: body, brake, ind, indR, half, v: s.speed, cruise: s.speed, runner: !!s.runner,
      ignoreObs: !!s.ignoreObs, halt: false, damaged: false, hazard: false });
  }

  private posOf(id: string): V3 {
    const t = this.physics!.bodies.get(id)!.translation();
    return [t.x, t.y, t.z];
  }

  private updateSignals() {
    const sig = signalAt(this.time + this.offset);
    for (const ax of ["ew", "ns"] as Axis[]) {
      if (this.lastLight[ax] === sig[ax]) continue;
      this.lastLight[ax] = sig[ax];
      for (const { mesh: m, axis, colour } of this.lamps.values()) {
        if (axis !== ax) continue;
        const on = sig[ax] === colour;
        this.setState(m, on ? colour : "off");
        setEmissive(m, on ? LAMP_EMIT[colour] : ZERO);
      }
    }
    return sig;
  }

  private control(v: Vehicle, sig: ReturnType<typeof signalAt>, pos: V3) {
    if (v.damaged) return;
    const u = dot(pos, v.ap.dir);
    let target = v.halt ? 0 : v.cruise;
    const light = sig[v.ap.axis];
    if (!v.runner) {
      const ds = STOP_U - (u + VEH_HALF.len);
      if (ds > -0.5 && light !== "green") {
        const canStop = (v.v * v.v) / (2 * A_DOWN) <= ds + 0.5;
        if (light === "red" || canStop) target = Math.min(target, Math.sqrt(2 * A_PLAN * Math.max(ds - 0.2, 0)));
      }
    }
    if (!v.ignoreObs) {
      for (const o of this.vehicles) {
        if (o === v) continue;
        const rel = add3(this.posOf(o.id), pos, -1);
        const along = dot(rel, v.ap.dir);
        const lat = Math.abs(dot(rel, v.ap.right));
        const oAlong = Math.abs(dot(o.half, v.ap.dir.map(Math.abs) as V3));
        const oLat = Math.abs(dot(o.half, v.ap.right.map(Math.abs) as V3));
        if (along > 0 && lat < VEH_HALF.w + oLat + 0.4) {
          const gap = along - VEH_HALF.len - oAlong;
          target = Math.min(target, Math.sqrt(2 * A_PLAN * Math.max(gap - 2.0, 0)));
        }
      }
    }
    const prev = v.v;
    v.v = target > v.v ? Math.min(target, v.v + A_UP * DT) : Math.max(target, v.v - A_DOWN * DT);
    const st = v.v < 0.15 && target < 0.3 ? "stopped" : target < prev - 0.3 || v.v < prev - 1e-6 ? "braking" : "moving";
    this.setState(v.mesh, st);
    const brakeOn = st !== "moving";
    this.setState(v.brake, brakeOn ? "on" : "off");
    setEmissive(v.brake, brakeOn ? BRAKE_ON : ZERO);
    this.physics!.bodies.get(v.id)!.setLinvel({ x: v.ap.dir[0] * v.v, y: 0, z: v.ap.dir[2] * v.v }, true);
  }

  private pedControl(p: Ped, sig: ReturnType<typeof signalAt>, pos: V3) {
    const cw = CROSSWALKS[p.cw];
    // crossing the NS road is allowed while EW is green (NS red), and vice versa
    const allowed = (cw.across === "ns" ? sig.ew : sig.ns) === "green";
    const cycle = Math.floor((this.time + this.offset) / CYCLE_S);
    const rb = this.physics!.bodies.get(p.id)!;
    const s = cw.across === "ns" ? pos[0] : pos[2];
    if (!p.crossing) {
      const delay = new Rng(this.opts.seed * 131 + p.cw * 17 + cycle).uniform(0, 1.5);
      if (allowed && !p.halt && p.cycleDone !== cycle && sig.inPhase >= delay && sig.inPhase < 2.5) {
        p.crossing = true;
        this.setState(p.mesh, "crossing");
      }
      rb.setLinvel({ x: 0, y: 0, z: 0 }, true);
    }
    if (p.crossing) {
      const goal = -p.side * (ROAD_HALF + 1);
      if ((goal - s) * -p.side <= 0) {
        p.crossing = false; p.side = -p.side; p.cycleDone = cycle;
        this.setState(p.mesh, "waiting");
        rb.setLinvel({ x: 0, y: 0, z: 0 }, true);
        return;
      }
      const sp = -p.side * p.speed;
      rb.setLinvel(cw.across === "ns" ? { x: sp, y: 0, z: 0 } : { x: 0, y: 0, z: sp }, true);
    }
  }

  private onCollision(e: PhysEvent) {
    const key = `${e.a}|${e.b}`;
    if (this.collided.has(key)) return;        // one semantic event per pair (re-contacts are the same crash)
    this.collided.add(key);
    const va = this.vehicles.find((v) => v.id === e.a), vb = this.vehicles.find((v) => v.id === e.b);
    if (!va || !vb) return;
    const pa = this.posOf(va.id), pb = this.posOf(vb.id);
    const mid: V3 = [(pa[0] + pb[0]) / 2, 0, (pa[2] + pb[2]) / 2];
    this.collisionPoint = mid;
    const rel = Math.hypot(va.ap.dir[0] * va.v - vb.ap.dir[0] * vb.v, va.ap.dir[2] * va.v - vb.ap.dir[2] * vb.v);
    this.emit({ type: "collision", object_id: va.id, data: { other: vb.id, step: this.step, point: mid.map((x) => Math.round(x * 1e3) / 1e3), rel_speed: Math.round(rel * 1e3) / 1e3 } });
    for (const v of [va, vb]) {
      v.damaged = true; v.hazard = true; v.v = 0;
      this.physics!.bodies.get(v.id)!.setLinearDamping(2.5);
      this.setState(v.mesh, "damaged");
      this.setState(v.brake, "on"); setEmissive(v.brake, BRAKE_ON);     // foot stays on the brake after impact
      this.setState(v.ind, "hazard"); this.setState(v.indR, "hazard");
      this.addTags(v.mesh, ["hazard", "collision"]);
      this.addTags(v.ind, ["hazard"]); this.addTags(v.indR, ["hazard"]);
    }
    // hazard indicators at the crash site
    const beacon = mesh(new THREE.BoxGeometry(0.4, 0.4, 0.4), AMBER_LENS, { id: "hazard_beacon", cls: "light", state: "hazard", tags: ["hazard"] }, AMBER_ON);
    beacon.position.set(mid[0] + 1.5, 2.6, mid[2] + 1.5);
    const smoke = mesh(new THREE.SphereGeometry(1, 16, 12), [0.16, 0.15, 0.15], { id: "smoke", cls: "smoke", state: "rising", tags: ["hazard"] }, undefined, 1);
    smoke.position.set(mid[0], 1.6, mid[2]);
    smoke.scale.setScalar(0.5);
    this.scene.add(beacon, smoke);
    this.hazardObjs.push(beacon, smoke);
    // importance bump + reaction of everything nearby
    for (const v of this.vehicles) {
      if (v === va || v === vb) continue;
      const p = this.posOf(v.id);
      if (Math.hypot(p[0] - mid[0], p[2] - mid[2]) > NEAR_R) continue;
      v.halt = true; v.hazard = true;
      this.addTags(v.mesh, ["hazard", "near_collision"]);
      this.setState(v.ind, "hazard"); this.setState(v.indR, "hazard");
    }
    for (const w of [...this.workers, ...this.bystanders]) {
      if (Math.hypot(w.position.x - mid[0], w.position.z - mid[2]) > NEAR_R) continue;
      this.addTags(w, ["hazard", "near_collision"]);
    }
    for (const p of this.peds) {
      const q = this.posOf(p.id);
      if (Math.hypot(q[0] - mid[0], q[2] - mid[2]) > NEAR_R) continue;
      p.halt = true;
      this.addTags(p.mesh, ["hazard", "near_collision"]);
    }
  }

  /** One fixed physics step (DT). */
  tick() {
    if (!this.physics) this.reset();
    const P = this.physics!;
    const t = this.time;
    while (this.spawns.length && this.spawns[0].t <= t) {
      const s = this.spawns[0];
      const busy = this.vehicles.some((v) => v.ap === s.ap && dot(this.posOf(v.id), s.ap.dir) < SPAWN_U + 10);
      if (busy) { s.t += 0.5; this.spawns.sort((a, b) => a.t - b.t || a.ap.name.localeCompare(b.ap.name)); break; }
      this.spawns.shift();
      this.spawnVehicle(s);
    }
    const sig = this.updateSignals();
    for (const v of this.vehicles) this.control(v, sig, this.posOf(v.id));
    for (const p of this.peds) this.pedControl(p, sig, this.posOf(p.id));
    const evs = P.tick();
    this.step++;
    for (const e of evs) {
      this.log.push(e);
      if (e.kind === "collision") this.onCollision(e);
      else this.emit({ type: "trigger_enter", object_id: e.a, data: { trigger: e.b, step: e.step } });
    }
    // sync meshes, despawn, flashing hazard lights, smoke growth
    const flash = Math.floor(this.time * FLASH_HZ * 2) % 2 === 0;
    for (const v of [...this.vehicles]) {
      const p = this.posOf(v.id);
      v.mesh.position.set(...p);
      if (v.hazard) { setEmissive(v.ind, flash ? AMBER_ON : ZERO); setEmissive(v.indR, flash ? AMBER_ON : ZERO); }
      if (!v.damaged && dot(p, v.ap.dir) > DESPAWN_U) {
        this.scene.remove(v.mesh); P.remove(v.id);
        this.vehicles.splice(this.vehicles.indexOf(v), 1);
      }
    }
    for (const p of this.peds) p.mesh.position.set(...this.posOf(p.id));
    const worksOn = Math.floor(this.time * 2) % 2 === 0;     // road-works warning lamp, 1 Hz
    for (const l of this.worksLamps) setEmissive(l, worksOn ? AMBER_ON : ZERO);
    for (const h of this.hazardObjs) {
      if (h.userData.sv.id === "hazard_beacon") setEmissive(h, flash ? ZERO : AMBER_ON);
      else {
        const s = Math.min(2.5, h.scale.x + 0.5 * DT);
        h.scale.setScalar(s); h.position.y = 1.6 + 0.5 * s;
      }
    }
  }

  /** Accumulator: run as many fixed steps as `elapsed` seconds cover. */
  advance(elapsed: number) { return this.stepper.advance(elapsed, () => this.tick()); }

  vehicleIds() { return this.vehicles.map((v) => v.id); }
}

export interface RecordOpts extends TrafficOpts {
  steps: number; exportEvery: number; observerId?: string; budgetMs?: number;
  /** first step to export (earlier exports are skipped and their events dropped); frame numbers stay global */
  startStep?: number;
}

/** Physics-only run (no export): step of the first collision event, or -1. */
export async function findCollisionStep(o: TrafficOpts, maxSteps = 1200): Promise<number> {
  await initPhysics();
  const sim = new TrafficSim(o);
  for (let s = 1; s <= maxSteps; s++) {
    sim.tick();
    if (sim.log.some((e) => e.kind === "collision")) { sim.physics!.free(); return s; }
  }
  sim.physics!.free();
  return -1;
}

/** Headless recorder (no WebGL): steps the scenario and exports a snapshot every `exportEvery` fixed steps. */
export async function recordTraffic(o: RecordOpts, sink?: (s: Snapshot) => void): Promise<{ snaps: Snapshot[]; sim: TrafficSim }> {
  await initPhysics();
  const sim = new TrafficSim(o);
  const prev = new Map<string, [number, number]>();
  const snaps: Snapshot[] = [];
  let prevT: number | undefined;
  for (let s = 1; s <= o.steps; s++) {
    sim.tick();
    if (s % o.exportEvery) continue;
    if (s < (o.startStep ?? 0)) { sim.scene.userData.svEvents = []; continue; }
    const t = sim.time;
    const snap = exportSnapshot(sim.scene, sim.camera, { frame: s / o.exportEvery - 1, t, observerId: o.observerId ?? "tr-0-000", budgetMs: o.budgetMs ?? 40, prev, prevT });
    snap.session_id = `traffic-${o.scenario}-${o.env}-${o.seed}`;
    prevT = t;
    if (sink) sink(snap); else snaps.push(snap);
  }
  sim.physics?.free();
  sim.physics = null;
  return { snaps, sim };
}
