// SH-L2 lighting export + lit colour (SPEC 5.2). Mirrors core/lighting (Python); engine code lives only here.
// Convention: radiance SH c_lm; normalised diffuse irradiance E(n)=sum A_l c_lm Y_lm(n), A=(1,2/3,1/4);
// lit = albedo*E*exposure + emissive*exposure (emission is added, never scaled by irradiance).
// Three.js (r155+): out = albedo/pi * irradiance, so ambient A -> E=A/pi, directional I -> E=(I/pi)cos+.
import * as THREE from "three";
import type { DominantLight, SHLighting } from "./protocol.gen";

type Vec3 = [number, number, number];
const r6 = (x: number) => Math.round(x * 1e6) / 1e6;
const Y00 = 0.282094791773878;
const A_L = [1, 2 / 3, 2 / 3, 2 / 3, 0.25, 0.25, 0.25, 0.25, 0.25];

export function shBasis(n: Vec3): number[] {
  const [x, y, z] = n;
  return [
    Y00, 0.48860251190292 * y, 0.48860251190292 * z, 0.48860251190292 * x,
    1.092548430592079 * x * y, 1.092548430592079 * y * z, 0.31539156525252 * (3 * z * z - 1),
    1.092548430592079 * x * z, 0.54627421529604 * (x * x - y * y),
  ];
}
const unit = (v: Vec3): Vec3 => {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
};
export const zeros = (): Vec3[] => Array.from({ length: 9 }, () => [0, 0, 0] as Vec3);

/** SH of a directional light whose normalised irradiance on a facing surface is rgb. */
export function deltaCoeffs(dir: Vec3, rgb: Vec3): Vec3[] {
  const b = shBasis(unit(dir));
  return b.map((y) => rgb.map((c) => Math.PI * y * c) as Vec3);
}
export function addCoeffs(a: Vec3[], b: Vec3[]): Vec3[] {
  return a.map((v, i) => [v[0] + b[i][0], v[1] + b[i][1], v[2] + b[i][2]] as Vec3);
}

/** Normal-averaged (n null) or by-normal irradiance, clamped >= 0. */
export function shIrradiance(c: Vec3[], n: Vec3 | null): Vec3 {
  if (!n) return c[0].map((v) => Math.max(Y00 * v, 0)) as Vec3;
  const b = shBasis(unit(n));
  const e: Vec3 = [0, 0, 0];
  for (let i = 0; i < 9; i++) for (let k = 0; k < 3; k++) e[k] += A_L[i] * b[i] * c[i][k];
  return e.map((v) => Math.max(v, 0)) as Vec3;
}

function domAt(d: DominantLight, p: Vec3 | null): { dir: Vec3; rgb: Vec3 } {
  if (!d.position || !p) return { dir: unit(d.direction as Vec3), rgb: d.rgb as Vec3 };
  const v: Vec3 = [d.position[0] - p[0], d.position[1] - p[1], d.position[2] - p[2]];
  const dist = Math.max(Math.hypot(...v), 1e-6);
  const f = ((d.ref_dist ?? 1) / dist) ** 2;
  return { dir: [v[0] / dist, v[1] / dist, v[2] / dist], rgb: (d.rgb as Vec3).map((c) => c * f) as Vec3 };
}

/** E = SH(n) - dom_nominal(n) + vis*dom(n,p); plain SH when no dominant or neither vis nor p given. */
export function irradiance(L: SHLighting, n: Vec3 | null, p: Vec3 | null = null, vis: number | null = null): Vec3 {
  const c = L.coeffs as Vec3[];
  const d = L.dominant;
  if (!d || (vis === null && p === null)) return shIrradiance(c, n);
  const lin = (cc: Vec3[]): Vec3 => {   // unclamped (linear) irradiance, as in Python
    if (!n) return cc[0].map((q) => Y00 * q) as Vec3;
    const b = shBasis(unit(n));
    const e: Vec3 = [0, 0, 0];
    for (let i = 0; i < 9; i++) for (let k = 0; k < 3; k++) e[k] += A_L[i] * b[i] * cc[i][k];
    return e;
  };
  const at = domAt(d, p);
  const B = lin(c), N = lin(deltaCoeffs(d.direction as Vec3, d.rgb as Vec3)), P = lin(deltaCoeffs(at.dir, at.rgb));
  const v = vis ?? 1;
  return [0, 1, 2].map((k) => Math.max(B[k] - N[k] + v * P[k], 0)) as Vec3;
}

export function litColor(albedo: Vec3, emissive: Vec3, E: Vec3, exposure: number): Vec3 {
  return [0, 1, 2].map((k) => Math.min(1, Math.max(0, albedo[k] * E[k] * exposure + emissive[k] * exposure))) as Vec3;
}

/** Project the scene's Three.js lights onto SH-L2 (+ dominant descriptor). Normalised (r155+ physical) units. */
export function exportLightingSH(scene: THREE.Scene, centroid: THREE.Vector3): SHLighting {
  scene.updateMatrixWorld(true);
  let coeffs = zeros();
  let best = -1;
  let dom: DominantLight | undefined;
  const add = (cc: Vec3[]) => { coeffs = addCoeffs(coeffs, cc); };
  scene.traverse((o) => {
    const l = o as THREE.Light;
    if (!l.isLight) return;
    const rgb = l.color.toArray().slice(0, 3).map((v) => v * l.intensity) as Vec3;
    if ((l as THREE.AmbientLight).isAmbientLight) {
      const c = zeros(); c[0] = rgb.map((v) => v / Math.PI / Y00) as Vec3; add(c);
    } else if ((l as THREE.HemisphereLight).isHemisphereLight) {
      const g = (l as THREE.HemisphereLight).groundColor.toArray().slice(0, 3).map((v) => v * l.intensity) as Vec3;
      const c = zeros();
      c[0] = rgb.map((v, k) => (v + g[k]) / 2 / Math.PI / Y00) as Vec3;
      c[1] = rgb.map((v, k) => (v - g[k]) / 2 / Math.PI / ((2 / 3) * 0.48860251190292)) as Vec3; // +y band-1 term
      add(c);
    } else if ((l as THREE.DirectionalLight).isDirectionalLight) {
      const p = new THREE.Vector3().setFromMatrixPosition(l.matrixWorld);
      const t = (l as THREE.DirectionalLight).target;
      const d = p.sub(new THREE.Vector3().setFromMatrixPosition(t.matrixWorld));
      if (d.lengthSq() === 0) return;
      d.normalize();
      const rn = rgb.map((v) => v / Math.PI) as Vec3;
      const dir: Vec3 = [d.x, d.y, d.z];
      add(deltaCoeffs(dir, rn));
      const score = rn[0] + rn[1] + rn[2];
      if (score > best) { best = score; dom = { rgb: rn.map(r6) as Vec3, direction: dir.map(r6) as Vec3, ref_dist: 1 }; }
    } else if ((l as THREE.PointLight).isPointLight) {
      const pos = new THREE.Vector3().setFromMatrixPosition(l.matrixWorld);
      const dv = pos.clone().sub(centroid);
      const ref = Math.max(dv.length(), 1e-3);
      const rn = rgb.map((v) => v / Math.PI / (ref * ref)) as Vec3;   // intensity I/(pi d^2) at the centroid
      const dir: Vec3 = [dv.x / ref, dv.y / ref, dv.z / ref];
      add(deltaCoeffs(dir, rn));
      const score = rn[0] + rn[1] + rn[2];
      if (score > best) {
        best = score;
        dom = { rgb: rn.map(r6) as Vec3, direction: dir.map(r6) as Vec3, position: [r6(pos.x), r6(pos.y), r6(pos.z)], ref_dist: r6(ref) };
      }
    }
  });
  return { kind: "sh2", coeffs: coeffs.map((v) => v.map(r6) as Vec3), dominant: dom ?? null, exposure: 1 } as SHLighting;
}

/** Fraction of the dominant light visible from the object's camera-facing surface (other meshes occlude; self excluded). */
export function shadowVisibility(
  scene: THREE.Scene, self: THREE.Object3D, point: THREE.Vector3, halfExtent: number, dom: DominantLight | null | undefined,
): number | null {
  if (!dom) return null;
  const own = new Set<THREE.Object3D>();
  self.traverse((o) => own.add(o));
  const occ: THREE.Object3D[] = [];
  scene.traverse((o) => { if ((o as THREE.Mesh).isMesh && !own.has(o) && (o as THREE.Mesh).visible) occ.push(o); });
  const rc = new THREE.Raycaster();
  const K = 8;
  let free = 0;
  for (let k = 0; k < K; k++) {
    const target = dom.position ? new THREE.Vector3(...(dom.position as Vec3)) : null;
    const l = target ? target.clone().sub(point).normalize() : new THREE.Vector3(...(dom.direction as Vec3)).normalize();
    const a = Math.abs(l.x) > 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
    const t1 = new THREE.Vector3().crossVectors(a, l).normalize();
    const t2 = new THREE.Vector3().crossVectors(l, t1);
    const r = Math.sqrt((k + 0.5) / K) * halfExtent, th = k * 2.399963; // Vogel disk over the face
    const o = point.clone().addScaledVector(t1, r * Math.cos(th)).addScaledVector(t2, r * Math.sin(th));
    rc.set(o, l);
    rc.near = 1e-3;
    rc.far = target ? o.distanceTo(target) : 1e4;
    if (rc.intersectObjects(occ, false).length === 0) free++;
  }
  return r6(free / K);
}
