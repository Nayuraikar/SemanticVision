// Player (viewer camera) movement on a bounded ground path, integrated at the fixed step.
export interface Box { minX: number; maxX: number; minZ: number; maxZ: number }
export interface Beacon { id: string; label: string; x: number; z: number; look: [number, number, number] }
export interface Ground extends Box {
  eye: number; speed: number; blocks: Box[]; beacons: Beacon[];
  start: { x: number; z: number; look: [number, number, number] };
}
export interface Keys { forward: number; right: number; turn: number; sprint: boolean }
export interface Goal { x: number; z: number; yaw?: number; pitch?: number }
export interface Player {
  x: number; z: number; yaw: number; pitch: number; zoom: number; vx: number; vz: number;
  keys: Keys; goal: Goal | null;
}

export const NO_KEYS: Keys = { forward: 0, right: 0, turn: 0, sprint: false };
export const SPRINT = 2, TURN_RATE = 1.8, EASE = 7, PITCH_LIMIT = 1.2, ZOOM_MIN = 0.35, ZOOM_MAX = 1.3;

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const unit = (v: number) => clamp(Number.isFinite(v) ? v : 0, -1, 1);
/** shortest signed angle from a to b */
export const angleTo = (a: number, b: number) => Math.atan2(Math.sin(b - a), Math.cos(b - a));

/** yaw 0 looks along -z; positive pitch looks up. */
export function facing(x: number, y: number, z: number, look: [number, number, number]) {
  const dx = look[0] - x, dz = look[2] - z;
  return { yaw: Math.atan2(-dx, -dz), pitch: Math.atan2(look[1] - y, Math.hypot(dx, dz)) };
}

export function spawnPlayer(g: Ground): Player {
  const f = facing(g.start.x, g.eye, g.start.z, g.start.look);
  return { x: g.start.x, z: g.start.z, ...f, zoom: 1, vx: 0, vz: 0, keys: { ...NO_KEYS }, goal: null };
}

const inside = (b: Box, x: number, z: number) => x > b.minX && x < b.maxX && z > b.minZ && z < b.maxZ;

export function setKeys(p: Player, k: Keys) {
  p.keys = { forward: unit(k.forward), right: unit(k.right), turn: unit(k.turn), sprint: !!k.sprint };
  if (p.keys.forward || p.keys.right || p.keys.turn) p.goal = null;
}
export function look(p: Player, dYaw: number, dPitch: number) {
  p.yaw += dYaw; p.pitch = clamp(p.pitch + dPitch, -PITCH_LIMIT, PITCH_LIMIT); p.goal = null;
}
export const zoom = (p: Player, factor: number) => { p.zoom = clamp(p.zoom * factor, ZOOM_MIN, ZOOM_MAX); };
export function walkTo(p: Player, g: Ground, x: number, z: number) {
  p.goal = { x: clamp(x, g.minX, g.maxX), z: clamp(z, g.minZ, g.maxZ) };
}
export function flyTo(p: Player, g: Ground, id: string) {
  const b = g.beacons.find((x) => x.id === id);
  if (b) p.goal = { x: b.x, z: b.z, ...facing(b.x, g.eye, b.z, b.look) };
}

export function stepPlayer(p: Player, g: Ground, dt: number) {
  const k = p.keys;
  p.yaw -= k.turn * TURN_RATE * dt;      // turn +1 = right
  const top = g.speed * (k.sprint ? SPRINT : 1);
  let wx = 0, wz = 0;
  if (k.forward || k.right) {
    const s = Math.sin(p.yaw), c = Math.cos(p.yaw);
    const x = -s * k.forward + c * k.right, z = -c * k.forward - s * k.right;
    const n = Math.max(1, Math.hypot(x, z));
    wx = (x / n) * top; wz = (z / n) * top;
  } else if (p.goal) {
    const x = p.goal.x - p.x, z = p.goal.z - p.z, d = Math.hypot(x, z);
    const v = Math.min(2 * g.speed, 3 * d);
    if (d > 1e-9) { wx = (x / d) * v; wz = (z / d) * v; }
  }
  const a = 1 - Math.exp(-EASE * dt);
  p.vx += (wx - p.vx) * a; p.vz += (wz - p.vz) * a;
  // axes are resolved separately so the player slides along walls
  const move = (nx: number, nz: number) => {
    const x = clamp(nx, g.minX, g.maxX), z = clamp(nz, g.minZ, g.maxZ);
    if (g.blocks.some((b) => inside(b, x, z))) return false;
    p.x = x; p.z = z;
    return true;
  };
  // fly-to beacons pass over obstacles
  if (p.goal?.yaw !== undefined) { p.x = clamp(p.x + p.vx * dt, g.minX, g.maxX); p.z = clamp(p.z + p.vz * dt, g.minZ, g.maxZ); }
  else {
    if (!move(p.x + p.vx * dt, p.z)) p.vx = 0;
    if (!move(p.x, p.z + p.vz * dt)) p.vz = 0;
  }
  if (!p.goal) return;
  let turned = true;
  if (p.goal.yaw !== undefined) {
    const dy = angleTo(p.yaw, p.goal.yaw), dq = (p.goal.pitch ?? 0) - p.pitch;
    p.yaw += dy * a; p.pitch += dq * a;
    turned = Math.abs(dy) < 0.01 && Math.abs(dq) < 0.01;
  }
  const near = Math.hypot(p.goal.x - p.x, p.goal.z - p.z) < 0.03 * g.speed;
  const stuck = p.goal.yaw === undefined && Math.hypot(p.vx, p.vz) < 1e-3;
  if ((near && turned) || stuck) { p.goal = null; p.vx = 0; p.vz = 0; }
}

const box = (minX: number, maxX: number, minZ: number, maxZ: number): Box => ({ minX, maxX, minZ, maxZ });
const spot = (id: string, label: string, x: number, z: number, look_: [number, number, number]): Beacon =>
  ({ id, label, x, z, look: look_ });

export const GROUNDS: Record<"traffic" | "dashboard" | "toy", Ground> = {
  traffic: {
    ...box(-30, 30, -30, 30), eye: 3, speed: 6, start: { x: 12, z: 18, look: [0, 1.5, 0] },
    blocks: [box(-28.5, -11.5, -33.5, -18.5), box(12.5, 27.5, -28.5, -11.5), box(-27.5, -12.5, 12.5, 27.5)],
    beacons: [
      spot("overview", "Overview", 12, 18, [0, 1.5, 0]), spot("stopline", "West approach", -18, 7, [0, 2, 0]),
      spot("works", "Road works", 13, 13, [7.5, 1, 7.5]), spot("far", "Far corner", 28, 22, [-6.5, 4.9, -5]),
      spot("lineup", "Parking bay", 3, 17, [-8, 1, 10.7]),
    ],
  },
  dashboard: {
    ...box(-1.1, 1.1, 0.6, 3), eye: 0.12, speed: 0.8, start: { x: 0, z: 2.45, look: [0, 0, 0] }, blocks: [],
    beacons: [
      spot("seat", "Seat", 0, 2.45, [0, 0, 0]), spot("annunciators", "Annunciators", -0.4, 1.35, [-0.4, 0.4, 0]),
      spot("gauges", "Gauges", 0, 1.7, [0, -0.18, 0]), spot("status", "Status lamps", 0.65, 1.35, [0.75, 0.4, 0]),
    ],
  },
  toy: {
    ...box(-10, 10, -8, 13), eye: 3.2, speed: 4, start: { x: 0, z: 10, look: [0, 0.5, 0] }, blocks: [],
    beacons: [
      spot("overview", "Overview", 0, 10, [0, 0.5, 0]), spot("bush", "Vegetation", 4, 4, [2, 0.6, -1]),
      spot("lamp", "Signal lamp", 6, 3, [4, 2, -2]),
    ],
  },
};
