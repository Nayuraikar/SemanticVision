// Three.js adapter: export per-frame Snapshot (SPEC 4.2). Engine code lives only here.
import * as THREE from "three";
import type { SceneObject, Snapshot } from "./protocol.gen";
import { exportLightingSH, irradiance, litColor, shadowVisibility } from "./lighting";
import { SCHEMA_VERSION } from "./protocol.gen";

/** Semantic tag carried on each exported Mesh via mesh.userData.sv. */
export interface SVTag {
  id: string;
  cls: string;
  state?: string;
  tags?: string[];
}
export interface OrigColors {
  albedo: [number, number, number];
  emissive: [number, number, number];
}

export type Vec3 = [number, number, number];
export const r6 = (x: number) => Math.round(x * 1e6) / 1e6;
const c01 = (x: number) => Math.min(1, Math.max(0, x));

export type SVMesh = THREE.Mesh<THREE.BufferGeometry, THREE.MeshStandardMaterial>;

export function svMeshes(scene: THREE.Object3D): SVMesh[] {
  const out: SVMesh[] = [];
  scene.traverse((o) => {
    if ((o as THREE.Mesh).isMesh && o.userData.sv) out.push(o as SVMesh);
  });
  return out;
}

/** Colours as authored (before any override), linear RGB. */
export function originalColors(m: SVMesh): OrigColors {
  const saved = m.userData.svOrig as OrigColors | undefined;
  if (saved) return saved;
  const e = m.material.emissive;
  const k = m.material.emissiveIntensity;
  return { albedo: m.material.color.toArray().slice(0, 3) as Vec3, emissive: [e.r * k, e.g * k, e.b * k] };
}

export interface ExportOpts {
  frame: number;
  t: number;
  observerId: string;
  budgetMs: number;
  /** previous centre positions (normalised viewport) for velocity; mutated in place. */
  prev?: Map<string, [number, number]>;
  prevT?: number;
}

const _c = new THREE.Vector3();

/** World normal of the surface facing the camera: box -> best-facing local axis; other shapes -> towards the camera. */
function facingNormal(m: THREE.Mesh, box: THREE.Box3, camera: THREE.Camera): THREE.Vector3 {
  const ctr = box.getCenter(new THREE.Vector3());
  const toCam = camera.position.clone().sub(ctr).normalize();
  if (m.geometry.type !== "BoxGeometry") return toCam;
  const q = m.getWorldQuaternion(new THREE.Quaternion());
  let best = toCam, bd = -2;
  for (const ax of [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]) {
    const w = new THREE.Vector3(ax[0], ax[1], ax[2]).applyQuaternion(q);
    const d = w.dot(toCam);
    if (d > bd) { bd = d; best = w; }
  }
  return best;
}

export function exportSnapshot(scene: THREE.Scene, camera: THREE.PerspectiveCamera, o: ExportOpts): Snapshot {
  scene.updateMatrixWorld(true);
  camera.updateMatrixWorld(true);
  const dt = o.prevT !== undefined ? o.t - o.prevT : 0;
  const objects: SceneObject[] = [];
  const box = new THREE.Box3();
  const meshes = svMeshes(scene);
  const centroid = new THREE.Vector3();
  for (const m of meshes) centroid.add(new THREE.Vector3().setFromMatrixPosition(m.matrixWorld));
  if (meshes.length) centroid.divideScalar(meshes.length);
  const lighting = exportLightingSH(scene, centroid);
  for (const m of meshes) {
    const tag = m.userData.sv as SVTag;
    // geometry-only box: cue children (outline, labels) must not inflate the exported bbox
    if (!m.geometry.boundingBox) m.geometry.computeBoundingBox();
    box.copy(m.geometry.boundingBox!).applyMatrix4(m.matrixWorld);
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity, ahead = 0;
    for (let i = 0; i < 8; i++) {
      _c.set(i & 1 ? box.max.x : box.min.x, i & 2 ? box.max.y : box.min.y, i & 4 ? box.max.z : box.min.z);
      const vz = _c.clone().applyMatrix4(camera.matrixWorldInverse).z;
      if (vz > -camera.near) continue; // behind the camera
      ahead++;
      _c.project(camera);
      const x = (_c.x + 1) / 2, y = (1 - _c.y) / 2;
      x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
    }
    if (!ahead) continue;
    const cx0 = c01(x0), cy0 = c01(y0), cx1 = c01(x1), cy1 = c01(y1);
    const area = (cx1 - cx0) * (cy1 - cy0);
    if (area <= 0) continue; // not in view
    const ctr0 = box.getCenter(new THREE.Vector3());
    const depth = -ctr0.clone().applyMatrix4(camera.matrixWorldInverse).z;
    const cur: [number, number] = [(x0 + x1) / 2, (y0 + y1) / 2];
    const pv = o.prev?.get(tag.id);
    const vel: [number, number] = pv && dt > 0 ? [(cur[0] - pv[0]) / dt, (cur[1] - pv[1]) / dt] : [0, 0];
    o.prev?.set(tag.id, cur);
    const col = originalColors(m);
    const nrm = facingNormal(m, box, camera);
    const pos: Vec3 = [ctr0.x, ctr0.y, ctr0.z];
    const half = 0.5 * Math.abs(nrm.x) * (box.max.x - box.min.x) + 0.5 * Math.abs(nrm.y) * (box.max.y - box.min.y) +
      0.5 * Math.abs(nrm.z) * (box.max.z - box.min.z);
    const facePt = ctr0.clone().addScaledVector(nrm, half + 1e-3);
    const vis = shadowVisibility(scene, m, facePt, 0.5 * Math.max(box.max.x - box.min.x, box.max.y - box.min.y, box.max.z - box.min.z), lighting.dominant);
    const nA: Vec3 = [nrm.x, nrm.y, nrm.z];
    const alb = col.albedo.map((v) => r6(c01(v))) as Vec3;
    const emi = col.emissive.map((v) => r6(Math.max(0, v))) as Vec3;
    const lit = litColor(alb, emi, irradiance(lighting, nA, pos, vis), lighting.exposure ?? 1);
    objects.push({
      id: tag.id,
      class: tag.cls,
      state: tag.state ?? "default",
      albedo_rgb: alb,
      emissive_rgb: emi,
      screen_area: r6(area),
      bbox: { x: r6(cx0), y: r6(cy0), w: r6(cx1 - cx0), h: r6(cy1 - cy0) },
      depth: r6(Math.max(0, depth)),
      velocity: [r6(vel[0]), r6(vel[1])],
      task_tags: tag.tags ?? [],
      lit_color: lit.map(r6) as Vec3,
      normal: nA.map(r6) as Vec3,
      position: pos.map(r6) as Vec3,
      shadow_vis: vis,
    });
  }
  return {
    schema_version: SCHEMA_VERSION,
    frame: o.frame,
    t: r6(o.t),
    objects,
    lighting,
    events: [],
    observer_id: o.observerId,
    budget_ms: o.budgetMs,
  };
}
