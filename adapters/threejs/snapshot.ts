// Three.js adapter: export per-frame Snapshot (SPEC 4.2). Engine code lives only here.
import * as THREE from "three";
import type { AmbientDominantLighting, SceneObject, Snapshot } from "./protocol.gen";
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

export function exportLighting(scene: THREE.Scene): AmbientDominantLighting {
  scene.updateMatrixWorld(true);
  const amb: Vec3 = [0, 0, 0];
  let dom: Vec3 = [0, 0, 0];
  let dir = new THREE.Vector3(0, 1, 0);
  let best = -1;
  scene.traverse((o) => {
    const l = o as THREE.Light;
    if (!l.isLight) return;
    const rgb = l.color.toArray();
    if ((l as THREE.AmbientLight).isAmbientLight) {
      for (let i = 0; i < 3; i++) amb[i] += rgb[i] * l.intensity;
    } else if ((l as THREE.HemisphereLight).isHemisphereLight) {
      const g = (l as THREE.HemisphereLight).groundColor.toArray();
      for (let i = 0; i < 3; i++) amb[i] += 0.5 * (rgb[i] + g[i]) * l.intensity;
    } else if ((l as THREE.DirectionalLight).isDirectionalLight) {
      const p = new THREE.Vector3().setFromMatrixPosition(l.matrixWorld);
      const t = (l as THREE.DirectionalLight).target;
      const d = p.sub(new THREE.Vector3().setFromMatrixPosition(t.matrixWorld));
      const lum = (rgb[0] + rgb[1] + rgb[2]) * l.intensity;
      if (d.lengthSq() > 0 && lum > best) {
        best = lum;
        dom = [rgb[0] * l.intensity, rgb[1] * l.intensity, rgb[2] * l.intensity];
        dir = d.normalize();
      }
    }
  });
  return {
    kind: "ambient_dominant",
    ambient_rgb: amb.map(r6) as Vec3,
    dominant_rgb: dom.map(r6) as Vec3,
    dominant_dir: [r6(dir.x), r6(dir.y), r6(dir.z)],
    exposure: 1,
  };
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

export function exportSnapshot(scene: THREE.Scene, camera: THREE.PerspectiveCamera, o: ExportOpts): Snapshot {
  scene.updateMatrixWorld(true);
  camera.updateMatrixWorld(true);
  const dt = o.prevT !== undefined ? o.t - o.prevT : 0;
  const objects: SceneObject[] = [];
  const box = new THREE.Box3();
  for (const m of svMeshes(scene)) {
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
    const ctr = box.getCenter(new THREE.Vector3());
    const depth = -ctr.applyMatrix4(camera.matrixWorldInverse).z;
    const cur: [number, number] = [(x0 + x1) / 2, (y0 + y1) / 2];
    const pv = o.prev?.get(tag.id);
    const vel: [number, number] = pv && dt > 0 ? [(cur[0] - pv[0]) / dt, (cur[1] - pv[1]) / dt] : [0, 0];
    o.prev?.set(tag.id, cur);
    const col = originalColors(m);
    objects.push({
      id: tag.id,
      class: tag.cls,
      state: tag.state ?? "default",
      albedo_rgb: col.albedo.map((v) => r6(c01(v))) as Vec3,
      emissive_rgb: col.emissive.map((v) => r6(Math.max(0, v))) as Vec3,
      screen_area: r6(area),
      bbox: { x: r6(cx0), y: r6(cy0), w: r6(cx1 - cx0), h: r6(cy1 - cy0) },
      depth: r6(Math.max(0, depth)),
      velocity: [r6(vel[0]), r6(vel[1])],
      task_tags: tag.tags ?? [],
    });
  }
  return {
    schema_version: SCHEMA_VERSION,
    frame: o.frame,
    t: r6(o.t),
    objects,
    lighting: exportLighting(scene),
    events: [],
    observer_id: o.observerId,
    budget_ms: o.budgetMs,
  };
}
