// Viewer-only decoration: dust, rain, fog puffs and waypoint markers live in their OWN scene, drawn after the CVD
// composite. They are never added to the sim scene, so snapshots, ground truth and measured colours cannot see them.
import * as THREE from "three";
import type { Waypoint } from "./fxPolicy";
import { disposeObject } from "./driver";

export interface FxFrame { dust: boolean; weather: string | null; waypoints: Waypoint[]; animate: boolean; dt: number; t: number }
type Kind = "dust" | "rain" | "fog";
const SPEC: Record<Kind, { n: number; fall: number; drift: number }> = {
  dust: { n: 140, fall: 0.03, drift: 0.04 }, rain: { n: 500, fall: 5, drift: 0.25 }, fog: { n: 40, fall: 0, drift: 0.12 },
};
const STREAK = 0.16, MAX_MARKERS = 3;
/** wrap into [-r, r) */
export const wrap = (x: number, r: number) => ((((x + r) % (2 * r)) + 2 * r) % (2 * r)) - r;

function softDot(): THREE.DataTexture {
  const n = 32, d = new Uint8Array(n * n * 4);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const r = Math.hypot(x + 0.5 - n / 2, y + 0.5 - n / 2) / (n / 2);
    d.set([255, 255, 255, Math.round(255 * Math.max(0, 1 - r) ** 2)], 4 * (y * n + x));
  }
  const t = new THREE.DataTexture(d, n, n);
  t.needsUpdate = true;
  return t;
}

export class FxLayer {
  readonly scene = new THREE.Scene();
  /** follows the camera, so one particle box serves every scene scale */
  private rig = new THREE.Group();
  private systems = new Map<Kind, THREE.Points | THREE.LineSegments>();
  private markers: THREE.Group[] = [];

  constructor() {
    this.rig.matrixAutoUpdate = false;
    this.scene.add(this.rig);
  }

  private system(kind: Kind) {
    let s = this.systems.get(kind);
    if (s) return s;
    const { n } = SPEC[kind], per = kind === "rain" ? 2 : 1;
    const pos = new Float32Array(n * per * 3);
    for (let i = 0; i < n; i++) {
      const z = -(1.5 + Math.random() * 6), x = (Math.random() - 0.5) * -z * 2.6, y = (Math.random() - 0.5) * 8;
      pos.set([x, y, z], i * per * 3);
      if (per === 2) pos.set([x - 0.01, y - STREAK, z], i * 6 + 3);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    const base = { transparent: true, depthTest: false, depthWrite: false };
    s = kind === "rain"
      ? new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ ...base, color: 0xaac4e0, opacity: 0.35 }))
      : new THREE.Points(geo, new THREE.PointsMaterial({
        ...base, map: softDot(), color: kind === "dust" ? 0xffd9a0 : 0xd8dde2,
        size: kind === "dust" ? 0.035 : 2.2, opacity: kind === "dust" ? 0.5 : 0.09,
      }));
    s.frustumCulled = false;
    this.systems.set(kind, s);
    this.rig.add(s);
    return s;
  }

  private marker(i: number) {
    if (this.markers[i]) return this.markers[i];
    const mat = (opacity: number) => new THREE.MeshBasicMaterial({
      color: 0xffb25c, transparent: true, opacity, depthTest: false, depthWrite: false,
      blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
    });
    const g = new THREE.Group();
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.8, 1, 40), mat(0.85));
    const beacon = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.14, 3, 8, 1, true), mat(0.4));
    beacon.position.y = 2.3;
    g.add(ring, beacon);
    this.markers[i] = g;
    this.scene.add(g);
    return g;
  }

  update(camera: THREE.Camera, f: FxFrame) {
    camera.updateMatrixWorld();
    this.rig.matrix.copy(camera.matrixWorld);
    this.rig.matrixWorldNeedsUpdate = true;
    const on: Record<Kind, boolean> = { dust: f.dust, rain: f.weather === "rain", fog: f.weather === "fog" };
    for (const kind of Object.keys(on) as Kind[]) {
      if (!on[kind]) { const s = this.systems.get(kind); if (s) s.visible = false; continue; }
      const s = this.system(kind);
      s.visible = true;
      const a = s.geometry.getAttribute("position") as THREE.BufferAttribute;
      const per = kind === "rain" ? 2 : 1, { fall, drift } = SPEC[kind];
      for (let i = 0; i < a.count; i += per) {
        const x = a.getX(i) + drift * Math.sin(f.t * 0.4 + i) * f.dt;
        const y = wrap(a.getY(i) - fall * f.dt, 4);
        a.setXY(i, x, y);
        if (per === 2) a.setXY(i + 1, x - 0.01, y - STREAK);
      }
      a.needsUpdate = true;
    }
    const n = Math.min(MAX_MARKERS, f.waypoints.length);
    for (let i = 0; i < n; i++) {
      const g = this.marker(i), w = f.waypoints[i];
      g.visible = true;
      g.position.set(...w.position);
      const pulse = f.animate ? 1 + 0.12 * Math.sin(f.t * 3 + i) : 1;
      g.scale.setScalar(camera.position.distanceTo(g.position) * 0.022);
      g.children[0].scale.setScalar(pulse);
      g.children[0].quaternion.copy(camera.quaternion);
    }
    for (let i = n; i < this.markers.length; i++) this.markers[i].visible = false;
  }

  dispose() { disposeObject(this.scene); this.systems.clear(); this.markers = []; this.rig.clear(); this.scene.clear(); }
}
