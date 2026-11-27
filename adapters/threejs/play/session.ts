// A play session: one world + the player, driven only by the seeded step-indexed input queue.
import * as THREE from "three";
import { DT } from "../physics";
import { Rng } from "../rng";
import { exportSnapshot } from "../snapshot";
import type { EnvName } from "../environments";
import type { DashLighting } from "../dashboardSim";
import { invalidatesOverrides, type Action } from "./actions";
import { InputQueue, type Input } from "./inputQueue";
import { GROUNDS, flyTo, look, setKeys, spawnPlayer, stepPlayer, walkTo, zoom, type Ground, type Player } from "./movement";
import { dashboardWorld, toyWorld, trafficWorld, type Facts, type World } from "./worlds";

export type PlaySceneId = keyof typeof GROUNDS;
export interface SessionOpts { sceneId: PlaySceneId; seed: number; env?: string; aspect?: number }
export interface InputLog { sceneId: PlaySceneId; seed: number; env: string; steps: number; inputs: Input[] }

export class PlaySession {
  step = 0;
  readonly ground: Ground;
  readonly player: Player;
  private world: World;
  private queue: InputQueue;
  private seen = new Set<THREE.Object3D>();
  readonly env: string;

  /** For the traffic scene initPhysics() must have been awaited. */
  constructor(readonly opts: SessionOpts, log: readonly Input[] = []) {
    const aspect = opts.aspect ?? 16 / 9;
    this.env = opts.env ?? (opts.sceneId === "traffic" ? "day" : opts.sceneId === "dashboard" ? "normal" : "default");
    this.world = opts.sceneId === "traffic" ? trafficWorld(opts.seed, this.env as EnvName, aspect)
      : opts.sceneId === "dashboard" ? dashboardWorld(opts.seed, this.env as DashLighting, aspect) : toyWorld(aspect);
    this.ground = GROUNDS[opts.sceneId];
    this.player = spawnPlayer(this.ground);
    this.queue = new InputQueue(opts.seed, log);
    this.track();
  }

  get scene() { return this.world.scene; }
  get camera() { return this.world.camera; }

  push(a: Action) { this.queue.push(this.step, a); }
  /** True when the next tick changes authored colours. */
  willInvalidate() { return this.queue.pending(this.step).some((i) => invalidatesOverrides(i.action)); }

  private apply(i: Input) {
    const a = i.action, p = this.player;
    if (a.type === "keys") setKeys(p, a);
    else if (a.type === "look") look(p, a.yaw, a.pitch);
    else if (a.type === "zoom") zoom(p, a.factor);
    else if (a.type === "walk") walkTo(p, this.ground, a.x, a.z);
    else if (a.type === "fly") flyTo(p, this.ground, a.id);
    else {
      if (a.type === "mission") {
        const to = a.name === "signal" ? "far" : a.name === "lineup" ? "lineup" : a.name === "end" ? null : this.ground.beacons[0].id;
        if (to) flyTo(p, this.ground, to);
      }
      // every action draws from its own stream: seed + step + order
      this.world.apply(a, new Rng(this.opts.seed * 100003 + i.step * 31 + i.seq));
    }
  }

  private track() { for (const c of this.world.scene.children) this.seen.add(c); }

  /** One fixed step: due inputs, then the world, then the player. */
  tick() {
    for (const i of this.queue.take(this.step)) this.apply(i);
    this.track();
    this.step++;
    this.world.tick(this.step);
    stepPlayer(this.player, this.ground, DT);
    this.track();
  }

  facts(): Facts { return this.world.facts(); }

  /** Writes the player's pose into `out` (field of view is the authored one times the player zoom). */
  pose(out: THREE.PerspectiveCamera) {
    const p = this.player;
    out.position.set(p.x, this.ground.eye, p.z);
    out.rotation.set(p.pitch, p.yaw, 0, "YXZ");
    out.near = this.camera.near; out.far = this.camera.far;
    out.updateMatrixWorld(true);
    return this.camera.fov * p.zoom;
  }

  log(): InputLog {
    return { sceneId: this.opts.sceneId, seed: this.opts.seed, env: this.env, steps: this.step, inputs: this.queue.log() };
  }

  dispose() {
    this.world.dispose();
    for (const root of this.seen) root.traverse((o) => {
      const m = o as THREE.Mesh;
      m.geometry?.dispose?.();
      for (const mat of m.material ? [m.material].flat() : []) mat.dispose();
    });
    this.seen.clear();
  }
}

/** FNV-1a over the exported snapshot stream (+ player pose), every `every` steps. */
export function runHash(s: PlaySession, steps: number, every = 30, drive?: (s: PlaySession) => void): string {
  let h = 0x811c9dc5;
  const cam = new THREE.PerspectiveCamera(50, s.opts.aspect ?? 16 / 9);
  while (s.step < steps) {
    drive?.(s);
    s.tick();
    if (s.step % every) continue;
    cam.fov = s.pose(cam); cam.updateProjectionMatrix();
    const snap = exportSnapshot(s.scene, cam, { frame: s.step, t: s.step * DT, observerId: "replay", budgetMs: 40 });
    const text = JSON.stringify([snap, s.player.x, s.player.z, s.player.yaw, s.player.pitch]);
    for (let i = 0; i < text.length; i++) h = Math.imul(h ^ text.charCodeAt(i), 0x01000193) >>> 0;
  }
  return h.toString(16).padStart(8, "0");
}

export const replaySession = (log: InputLog) =>
  new PlaySession({ sceneId: log.sceneId, seed: log.seed, env: log.env }, log.inputs);
