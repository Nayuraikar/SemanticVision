// Registry wrapper: traffic intersection with Rapier physics (scripted red-light-runner collision, loops).
import type * as THREE from "three";
import { initPhysics, physicsReady } from "../physics";
import { TrafficSim } from "../trafficSim";
import type { EnvName } from "../environments";

export const id = "traffic";
export const title = "Traffic intersection";
export const description = "Signalised junction with physics: brake lights, indicators, pedestrians, a red-light runner collision.";
export const tasks = ["Read the signal state (red/amber/green)", "Spot braking vehicles", "Find the crash and its hazard lights"];
export const tags = ["traffic", "signals", "emissive", "physics"];

export const LOOP_S = 16;
export let ENV: EnvName = "day";
export const setEnv = (e: EnvName) => { ENV = e; };

export function build(aspect = 16 / 9) {
  void initPhysics();
  const sim = new TrafficSim({ seed: 1, scenario: "collision", env: ENV, aspect });
  return { scene: sim.scene, camera: sim.camera };
}

/** Fixed-step accumulator driven by the caller's clock t (s); the sim itself never reads wall-clock time. */
export function step(scene: THREE.Scene, t: number) {
  const sim = scene.userData.svSim as TrafficSim | undefined;
  if (!sim || !physicsReady()) return;
  const last = scene.userData.svLastT as number | undefined;
  scene.userData.svLastT = t;
  if (!sim.physics) sim.reset();
  if (last === undefined) return;
  sim.advance(t - last);
  if (sim.time > LOOP_S) sim.reset();
}
