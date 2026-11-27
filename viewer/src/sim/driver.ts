// Owns one live sim instance: seek(t) = (fresh sim if going back) + fixed-step ticks. Same sim for live and replay.
import * as THREE from "three";
import type { Event } from "../../../adapters/threejs/protocol.gen";
import { getScene } from "../../../adapters/threejs/scenes/registry";
import * as traffic from "../../../adapters/threejs/scenes/traffic";
import * as dashboard from "../../../adapters/threejs/scenes/dashboard";
import { initPhysics, physicsReady } from "../../../adapters/threejs/physics";
import type { TrafficSim } from "../../../adapters/threejs/trafficSim";
import type { DashboardSim } from "../../../adapters/threejs/dashboardSim";
import { EVENT_WINDOW, STEP_S, extractMarkers, seekPlan, type LogEntry, type Marker } from "./timeline";

const TOY_S = 8;
export const durationSteps = (sceneId: string) =>
  Math.round((sceneId === traffic.id ? traffic.LOOP_S : sceneId === dashboard.id ? dashboard.LOOP_S : TOY_S) / STEP_S);
export const envOf = (sceneId: string): string =>
  sceneId === traffic.id ? traffic.ENV : sceneId === dashboard.id ? dashboard.ENV : "default";

/** Free GPU resources of a scene (geometries, materials, textures). */
export function disposeObject(root: THREE.Object3D) {
  root.traverse((o) => {
    const m = o as THREE.Mesh;
    m.geometry?.dispose?.();
    for (const mat of m.material ? [m.material].flat() : []) {
      (mat as THREE.MeshBasicMaterial).map?.dispose();
      mat.dispose();
    }
  });
}

export class SimDriver {
  scene!: THREE.Scene;
  camera!: THREE.PerspectiveCamera;
  step = 0;
  readonly total: number;
  /** every top-level object ever seen: despawned vehicles must be disposed too */
  private seen = new Set<THREE.Object3D>();

  constructor(readonly sceneId: string, readonly aspect: number) {
    this.total = durationSteps(sceneId);
    this.build();
  }

  private get sim() {
    return (this.scene.userData.svSim ?? this.scene.userData.svDash) as TrafficSim | DashboardSim | undefined;
  }

  get ready() { return this.sceneId !== traffic.id || physicsReady(); }

  private build() {
    const def = getScene(this.sceneId);
    if (!def) throw new Error(`unknown scene ${this.sceneId}`);
    const b = def.build(this.aspect);
    this.scene = b.scene; this.camera = b.camera; this.step = 0;
    this.track();
  }

  private track() { for (const c of this.scene.children) this.seen.add(c); }

  private tick() {
    const sim = this.sim;
    if (sim) sim.tick();
    else getScene(this.sceneId)!.step(this.scene, (this.step + 1) * STEP_S);
    this.step++;
  }

  /** Advance forwards by n fixed steps. */
  run(n: number) {
    if (!this.ready) return;
    const sim = this.sim as TrafficSim | undefined;
    if (sim && "physics" in sim && !sim.physics) sim.reset();
    for (let i = 0; i < n; i++) { this.tick(); this.track(); }
  }

  /** Returns true when the scene object was replaced. */
  seek(target: number): boolean {
    if (!this.ready) return false;
    const p = seekPlan(this.step, target);
    if (p.rebuild) { this.dispose(); this.build(); }
    const early = p.ticks - EVENT_WINDOW;
    if (early > 0) { this.run(early); this.scene.userData.svEvents = []; this.run(EVENT_WINDOW); }
    else this.run(p.ticks);
    return p.rebuild;
  }

  dispose() {
    const sim = this.sim as TrafficSim | undefined;
    if (sim && "physics" in sim) { sim.physics?.free(); sim.physics = null; }
    for (const o of this.seen) disposeObject(o);
    this.seen.clear();
  }
}

type Idle = (fn: () => void) => void;
const idle: Idle = (fn) => {
  if (typeof requestIdleCallback === "function") requestIdleCallback(() => fn(), { timeout: 200 });
  else setTimeout(fn, 0);
};

const cache = new Map<string, Marker[]>();
/** Seeds are fixed by the scene registry, so scene + environment identifies a run. */
export const markerKey = (sceneId: string) => `${sceneId}|seed1|${envOf(sceneId)}`;

/** Headless pass over a private sim in idle-time slices, reading the sim's own event log. null when cancelled. */
export async function precomputeMarkers(
  sceneId: string, cancelled: () => boolean = () => false, schedule: Idle = idle, slice = 60,
): Promise<Marker[] | null> {
  const key = markerKey(sceneId);
  const hit = cache.get(key);
  if (hit) return hit;
  if (sceneId === traffic.id) await initPhysics();
  if (cancelled() || markerKey(sceneId) !== key) return null;
  const d = new SimDriver(sceneId, 16 / 9);
  const log: LogEntry[] = [];
  const toyStates = new Map<string, string>();
  const drain = () => {
    const events = (d.scene.userData.svEvents ?? []) as Event[];
    d.scene.userData.svEvents = [];
    if (!d.scene.userData.svSim && !d.scene.userData.svDash) {
      // the toy scene keeps no event log: its only events are lamp state flips
      d.scene.traverse((o) => {
        const sv = o.userData.sv as { id: string; state?: string } | undefined;
        if (!sv?.state) return;
        const was = toyStates.get(sv.id);
        if (was !== undefined && was !== sv.state) events.push({ type: "state_change", object_id: sv.id });
        toyStates.set(sv.id, sv.state);
      });
    }
    if (events.length) log.push({ step: d.step, events });
  };
  try {
    d.run(0);
    drain(); log.length = 0; // initial states set at reset are not events
    while (d.step < d.total) {
      await new Promise<void>((r) => schedule(r));
      if (cancelled()) return null;
      for (let i = 0; i < slice && d.step < d.total; i++) { d.run(1); drain(); }
    }
  } finally { d.dispose(); }
  const out = extractMarkers(log);
  cache.set(key, out);
  return out;
}
