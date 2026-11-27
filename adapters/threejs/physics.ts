// Rapier3D physics for the Three.js adapter (SPEC 6). Fixed timestep + accumulator; contact/sensor events -> PhysEvent.
// Engine code lives only in adapters/. No wall-clock time or Math.random here.
import RAPIER from "@dimforge/rapier3d-compat";

export const DT = 1 / 60;
let _ready: Promise<void> | null = null;
let _ok = false;

/** Load the (inlined) wasm once. Await before creating a Physics. */
export function initPhysics(): Promise<void> {
  if (!_ready) _ready = RAPIER.init().then(() => { _ok = true; });
  return _ready;
}
export const physicsReady = () => _ok;

/** Fixed-step accumulator: advance(elapsed) runs fn once per whole dt; leftover carries over. */
export class FixedStep {
  acc = 0;
  steps = 0;
  constructor(readonly dt = DT, readonly maxSteps = 240) {}
  advance(elapsed: number, fn: (step: number) => void): number {
    this.acc += Math.max(0, elapsed);
    let n = 0;
    while (this.acc >= this.dt - 1e-9 && n < this.maxSteps) {
      fn(this.steps);
      this.steps++; this.acc -= this.dt; n++;
    }
    if (n === this.maxSteps) this.acc = 0;   // spiral-of-death guard
    return n;
  }
}

export type V3 = [number, number, number];
export interface PhysEvent { kind: "collision" | "trigger_enter"; a: string; b: string; step: number }

// collision groups: membership << 16 | filter
export const G_VEH = 1, G_PED = 2, G_SENSOR = 4;
const groups = (member: number, filter: number) => ((member & 0xffff) << 16) | (filter & 0xffff);

export class Physics {
  readonly world: RAPIER.World;
  private queue: RAPIER.EventQueue;
  private idOf = new Map<number, string>();
  private sensors = new Set<number>();
  readonly bodies = new Map<string, RAPIER.RigidBody>();
  step = 0;

  constructor() {
    if (!_ok) throw new Error("initPhysics() not awaited");
    this.world = new RAPIER.World({ x: 0, y: 0, z: 0 });   // planar traffic: no gravity, y locked
    this.world.timestep = DT;
    this.queue = new RAPIER.EventQueue(true);
  }

  /** Dynamic box body moving in the ground plane (y translation + rotations locked). */
  addMover(id: string, pos: V3, half: V3, group: number, damping = 0): RAPIER.RigidBody {
    const b = this.world.createRigidBody(
      RAPIER.RigidBodyDesc.dynamic().setTranslation(...pos).enabledTranslations(true, false, true).lockRotations()
        .setLinearDamping(damping).setCcdEnabled(true),
    );
    const filter = group === G_VEH ? G_VEH | G_SENSOR : G_SENSOR;
    const c = this.world.createCollider(
      RAPIER.ColliderDesc.cuboid(...half).setDensity(group === G_VEH ? 150 : 80).setFriction(0.6).setRestitution(0.1)
        .setCollisionGroups(groups(group, filter)).setActiveEvents(RAPIER.ActiveEvents.COLLISION_EVENTS),
      b,
    );
    this.idOf.set(c.handle, id);
    this.bodies.set(id, b);
    return b;
  }

  /** Fixed trigger volume (sensor). */
  addSensor(id: string, pos: V3, half: V3): void {
    const b = this.world.createRigidBody(RAPIER.RigidBodyDesc.fixed().setTranslation(...pos));
    const c = this.world.createCollider(
      RAPIER.ColliderDesc.cuboid(...half).setSensor(true).setCollisionGroups(groups(G_SENSOR, G_VEH | G_PED))
        .setActiveEvents(RAPIER.ActiveEvents.COLLISION_EVENTS)
        .setActiveCollisionTypes(RAPIER.ActiveCollisionTypes.DEFAULT | RAPIER.ActiveCollisionTypes.KINEMATIC_FIXED),
      b,
    );
    this.idOf.set(c.handle, id);
    this.sensors.add(c.handle);
  }

  remove(id: string): void {
    const b = this.bodies.get(id);
    if (!b) return;
    for (let i = 0; i < b.numColliders(); i++) this.idOf.delete(b.collider(i).handle);
    this.world.removeRigidBody(b);
    this.bodies.delete(id);
  }

  /** One fixed step; returns contact-start events (sorted, deterministic). */
  tick(): PhysEvent[] {
    this.world.step(this.queue);
    const out: PhysEvent[] = [];
    this.queue.drainCollisionEvents((h1, h2, started) => {
      if (!started) return;
      const a = this.idOf.get(h1), b = this.idOf.get(h2);
      if (a === undefined || b === undefined) return;
      const s1 = this.sensors.has(h1), s2 = this.sensors.has(h2);
      if (s1 && s2) return;
      if (s1 || s2) out.push({ kind: "trigger_enter", a: s1 ? b : a, b: s1 ? a : b, step: this.step });
      else out.push({ kind: "collision", a: a < b ? a : b, b: a < b ? b : a, step: this.step });
    });
    this.step++;
    return out.sort((x, y) => (x.kind + x.a + x.b < y.kind + y.a + y.b ? -1 : 1));
  }

  free(): void {
    this.queue.free();
    this.world.free();
  }
}
