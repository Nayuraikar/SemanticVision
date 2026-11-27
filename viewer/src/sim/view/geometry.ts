// Screen-space helpers: hull for the glow outline, label collision avoidance, confusable-pair selection, picking.
import * as THREE from "three";
import type { ConfusionPair } from "../../../../adapters/threejs/protocol.gen";
import type { Rect, Pane } from "./layout";

export type Pt = [number, number];
export const DEFAULT_TOP_N = 8;
const HULL_SAMPLES = 400;

export function convexHull(pts: Pt[]): Pt[] {
  if (pts.length < 3) return pts.slice();
  const s = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o: Pt, a: Pt, b: Pt) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const half = (xs: Pt[]) => {
    const out: Pt[] = [];
    for (const p of xs) {
      while (out.length > 1 && cross(out[out.length - 2], out[out.length - 1], p) <= 0) out.pop();
      out.push(p);
    }
    out.pop();
    return out;
  };
  return [...half(s), ...half(s.reverse())];
}

const _v = new THREE.Vector3();
/** World point -> pixels inside `view`; null when behind the camera. */
export function projectPoint(p: THREE.Vector3, camera: THREE.PerspectiveCamera, view: Rect): Pt | null {
  _v.copy(p).applyMatrix4(camera.matrixWorldInverse);
  if (_v.z > -camera.near) return null;
  _v.applyMatrix4(camera.projectionMatrix);
  return [view.x + ((_v.x + 1) / 2) * view.w, view.y + ((1 - _v.y) / 2) * view.h];
}

/** Silhouette of a (near-)convex mesh from a subsample of its vertices. */
export function meshHull(m: THREE.Mesh, camera: THREE.PerspectiveCamera, view: Rect): Pt[] {
  const pos = m.geometry.getAttribute("position");
  if (!pos) return [];
  const stride = Math.max(1, Math.floor(pos.count / HULL_SAMPLES));
  const out: Pt[] = [], w = new THREE.Vector3();
  for (let i = 0; i < pos.count; i += stride) {
    const p = projectPoint(w.fromBufferAttribute(pos, i).applyMatrix4(m.matrixWorld), camera, view);
    if (p) out.push(p);
  }
  return convexHull(out);
}

/** Projected bounding box of a mesh's geometry box; null when off screen. */
export function meshBox(m: THREE.Mesh, camera: THREE.PerspectiveCamera, view: Rect): Rect | null {
  if (!m.geometry.boundingBox) m.geometry.computeBoundingBox();
  const b = m.geometry.boundingBox!, w = new THREE.Vector3();
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < 8; i++) {
    w.set(i & 1 ? b.max.x : b.min.x, i & 2 ? b.max.y : b.min.y, i & 4 ? b.max.z : b.min.z).applyMatrix4(m.matrixWorld);
    const p = projectPoint(w, camera, view);
    if (!p) continue;
    x0 = Math.min(x0, p[0]); x1 = Math.max(x1, p[0]); y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1]);
  }
  if (x1 < view.x || y1 < view.y || x0 > view.x + view.w || y0 > view.y + view.h) return null;
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
}

/** Id of the nearest tagged mesh under the NDC point; untagged geometry in front blocks the pick. */
export function pickObject(scene: THREE.Object3D, camera: THREE.PerspectiveCamera, ndc: Pt): string | null {
  const ray = new THREE.Raycaster();
  ray.setFromCamera(new THREE.Vector2(ndc[0], ndc[1]), camera);
  for (const hit of ray.intersectObjects(scene.children, true)) {
    if (!(hit.object as THREE.Mesh).isMesh) continue;
    let o: THREE.Object3D | null = hit.object, shown = true;
    for (let p: THREE.Object3D | null = o; p; p = p.parent) shown &&= p.visible;
    if (!shown) continue;
    while (o && !o.userData.sv) o = o.parent;
    return o ? (o.userData.sv.id as string) : null;
  }
  return null;
}

export interface LabelIn { id: string; anchor: Pt; w: number; h: number }
export interface LabelOut extends Rect { id: string; anchor: Pt }
export const overlaps = (a: Rect, b: Rect) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;

/** Greedy placement in priority order: try spots around the anchor, drop a label that cannot fit. */
export function placeLabels(labels: LabelIn[], bounds: Rect): LabelOut[] {
  const out: LabelOut[] = [];
  for (const l of labels) {
    const [ax, ay] = l.anchor;
    const spots: Pt[] = [];
    for (let k = 0; k < 4; k++) {
      const dy = k * (l.h + 4);
      spots.push([ax + 10, ay - l.h - 6 - dy], [ax - l.w - 10, ay - l.h - 6 - dy], [ax + 10, ay + 6 + dy], [ax - l.w - 10, ay + 6 + dy]);
    }
    for (const [sx, sy] of spots) {
      const r = {
        x: Math.min(bounds.x + bounds.w - l.w, Math.max(bounds.x, sx)),
        y: Math.min(bounds.y + bounds.h - l.h, Math.max(bounds.y, sy)), w: l.w, h: l.h,
      };
      if (r.x < bounds.x || r.y < bounds.y || out.some((o) => overlaps(o, r))) continue;
      out.push({ ...r, id: l.id, anchor: l.anchor });
      break;
    }
  }
  return out;
}

/**
 * Pairs to draw in a pane. The API reports pairs that were below tau on the original; `feasible` says whether the
 * corrected pair reaches tau. B: all of them. C: only those still below tau. A (unsimulated): none.
 * Worst (smallest dE00) first, at most `n`.
 */
export function topPairs(pairs: ConfusionPair[], pane: Pane, n = DEFAULT_TOP_N): ConfusionPair[] {
  if (pane === "A") return [];
  const de = (p: ConfusionPair) => (pane === "C" ? p.de00_after : p.de00_before);
  return pairs.filter((p) => pane === "B" || !p.feasible).sort((a, b) => de(a) - de(b)).slice(0, Math.max(0, n));
}

export function partnersOf(pairs: ConfusionPair[], id: string): { id: string; before: number; after: number; feasible: boolean }[] {
  return pairs.filter((p) => p.ids[0] === id || p.ids[1] === id)
    .map((p) => ({ id: p.ids[0] === id ? p.ids[1] : p.ids[0], before: p.de00_before, after: p.de00_after, feasible: p.feasible }))
    .sort((a, b) => a.before - b.before || a.id.localeCompare(b.id));
}
