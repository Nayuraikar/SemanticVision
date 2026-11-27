// Registry wrapper: instrument panel / monitor dashboard with a cascading alarm (loops).
import type * as THREE from "three";
import { DashboardSim, type DashLighting } from "../dashboardSim";

export const id = "dashboard";
export const title = "Dashboard";
export const description = "Instrument panel: annunciator grid, gauges, status lamps and a cascading alarm.";
export const tasks = ["Find the faulted system", "Tell warnings from cautions", "Read which gauge left its normal range"];
export const tags = ["ui", "status", "annunciator", "emissive"];

export const LOOP_S = 12;
export let ENV: DashLighting = "normal";
export const setEnv = (e: DashLighting) => { ENV = e; };

export function build(aspect = 16 / 9) {
  const sim = new DashboardSim({ seed: 1, scenario: "alarm", env: ENV, aspect });
  return { scene: sim.scene, camera: sim.camera };
}

/** Fixed-step accumulator driven by the caller's clock t (s); the sim itself never reads wall-clock time. */
export function step(scene: THREE.Scene, t: number) {
  const sim = scene.userData.svDash as DashboardSim | undefined;
  if (!sim) return;
  const last = scene.userData.svLastT as number | undefined;
  scene.userData.svLastT = t;
  if (last === undefined) return;
  sim.advance(t - last);
  if (sim.time > LOOP_S) sim.reset();
}
