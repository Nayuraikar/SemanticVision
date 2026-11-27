// Animated environment change on the real sim lights, so the relight is visible and snapshots stay truthful.
// The final frame writes the target values exactly (same state as a direct apply).
import * as THREE from "three";
import { applyEnvironment, type EnvName } from "../../../adapters/threejs/environments";
import type { DashboardSim, DashLighting } from "../../../adapters/threejs/dashboardSim";

export const RELIGHT_MS = 1400;
export const lerp = (a: number, b: number, k: number) => (k >= 1 ? b : a + (b - a) * k);

interface Fog { color: THREE.Color; density: number }
export interface LightState {
  lights: { color: THREE.Color; ground: THREE.Color; intensity: number }[][]; // [hemi, directional, point]
  background: THREE.Color; fog: Fog | null; exposure: number; white: number[];
  mats: Map<THREE.MeshStandardMaterial, { color: THREE.Color; emissive: THREE.Color; roughness: number }>;
}

function lightGroups(scene: THREE.Scene) {
  const g: THREE.Light[][] = [[], [], []];
  for (const o of scene.children) {
    if ((o as THREE.HemisphereLight).isHemisphereLight) g[0].push(o as THREE.Light);
    else if ((o as THREE.DirectionalLight).isDirectionalLight) g[1].push(o as THREE.Light);
    else if ((o as THREE.PointLight).isPointLight) g[2].push(o as THREE.Light);
  }
  return g;
}

function roleMeshes(scene: THREE.Scene) {
  const out: THREE.Mesh<THREE.BufferGeometry, THREE.MeshStandardMaterial>[] = [];
  scene.traverse((o) => { if (o.userData.svEnvRole) out.push(o as (typeof out)[number]); });
  return out;
}

export function captureLights(scene: THREE.Scene): LightState {
  const env = scene.userData.svEnv as { exposure: number; white_point: number[] };
  const fog = scene.fog as THREE.FogExp2 | null;
  return {
    lights: lightGroups(scene).map((g) => g.map((l) => ({
      color: l.color.clone(), intensity: l.intensity,
      ground: ((l as THREE.HemisphereLight).groundColor ?? l.color).clone(),
    }))),
    background: ((scene.background as THREE.Color | null) ?? new THREE.Color(0, 0, 0)).clone(),
    fog: fog ? { color: fog.color.clone(), density: fog.density } : null,
    exposure: env.exposure, white: [...env.white_point],
    mats: new Map(roleMeshes(scene).map((m) => [m.material, {
      color: m.material.color.clone(), emissive: m.material.emissive.clone(), roughness: m.material.roughness,
    }])),
  };
}

function blend(scene: THREE.Scene, a: LightState, b: LightState, k: number) {
  const col = (out: THREE.Color, x: THREE.Color, y: THREE.Color) => { if (k >= 1) out.copy(y); else out.copy(x).lerp(y, k); };
  lightGroups(scene).forEach((g, i) => g.forEach((l, j) => {
    const to = b.lights[i][j];
    if (!to) return;
    const from = a.lights[i][j] ?? { ...to, intensity: 0 };
    col(l.color, from.color, to.color);
    const h = l as THREE.HemisphereLight;
    if (h.groundColor) col(h.groundColor, from.ground, to.ground);
    l.intensity = lerp(from.intensity, to.intensity, k);
  }));
  const bg = new THREE.Color();
  col(bg, a.background, b.background);
  scene.background = bg;
  const fa = a.fog ?? (b.fog && { color: b.fog.color, density: 0 });
  const fb = b.fog ?? (a.fog && { color: a.fog.color, density: 0 });
  if (!fa || !fb || (k >= 1 && !b.fog)) scene.fog = null;
  else {
    const f = new THREE.FogExp2(0, lerp(fa.density, fb.density, k));
    col(f.color, fa.color, fb.color);
    scene.fog = f;
  }
  const env = scene.userData.svEnv as { exposure: number; white_point: number[] };
  env.exposure = lerp(a.exposure, b.exposure, k);
  env.white_point = b.white.map((w, i) => lerp(a.white[i], w, k));
  for (const m of roleMeshes(scene)) {
    const x = a.mats.get(m.material), y = b.mats.get(m.material);
    if (!x || !y) continue;
    col(m.material.color, x.color, y.color);
    col(m.material.emissive, x.emissive, y.emissive);
    m.material.roughness = lerp(x.roughness, y.roughness, k);
    delete m.userData.svOrig;
  }
}

/** Switch `scene` to `env`; returns apply(k), k in [0,1]. Clear colour overrides before calling. */
export function startRelight(scene: THREE.Scene, env: string): (k: number) => void {
  const from = captureLights(scene);
  const dash = scene.userData.svDash as DashboardSim | undefined;
  if (dash) dash.setLighting(env as DashLighting);
  else applyEnvironment(scene, env as EnvName);
  const to = captureLights(scene);
  const apply = (k: number) => blend(scene, from, to, Math.min(1, Math.max(0, k)));
  apply(0);
  return apply;
}
