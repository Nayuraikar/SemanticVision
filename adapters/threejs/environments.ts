// Environment presets (SPEC 6): lights (-> SH-L2 export), exposure, white point (XYZ) and exp2 fog density.
// exportSnapshot reads scene.userData.svEnv + scene.fog. Emission is never scaled by irradiance (lighting.ts).
import * as THREE from "three";

export type EnvName = "day" | "night" | "fog" | "rain";
export const ENVS: EnvName[] = ["day", "night", "fog", "rain"];
type V3 = [number, number, number];

export interface EnvPreset {
  hemi: { sky: V3; ground: V3; intensity: number };
  sun: { rgb: V3; intensity: number; pos: V3 };
  street: number;            // street-lamp point-light intensity (cd), 0 = off
  exposure: number;
  white_point: V3;           // XYZ (Y=1)
  fog: { rgb: V3; density: number } | null;
  sky: V3;                   // sky backdrop emission (linear)
  roadAlbedo: V3;            // wet road is darker
  roadRoughness: number;
}

export const PRESETS: Record<EnvName, EnvPreset> = {
  day: {
    hemi: { sky: [0.75, 0.85, 1.0], ground: [0.35, 0.3, 0.25], intensity: 1.3 },
    sun: { rgb: [1.0, 0.96, 0.88], intensity: 3.0, pos: [30, 50, 40] },
    street: 0, exposure: 1.0, white_point: [0.9505, 1.0, 1.089], fog: null,
    sky: [0.35, 0.55, 0.95], roadAlbedo: [0.07, 0.07, 0.075], roadRoughness: 0.9,
  },
  night: {
    hemi: { sky: [0.25, 0.3, 0.55], ground: [0.06, 0.05, 0.05], intensity: 0.25 },
    sun: { rgb: [0.7, 0.8, 1.0], intensity: 0.3, pos: [-20, 40, 30] },
    street: 90, exposure: 2.2, white_point: [1.0098, 1.0, 0.6441], fog: null,
    sky: [0.01, 0.015, 0.04], roadAlbedo: [0.07, 0.07, 0.075], roadRoughness: 0.9,
  },
  fog: {
    hemi: { sky: [0.8, 0.82, 0.85], ground: [0.45, 0.45, 0.45], intensity: 1.4 },
    sun: { rgb: [1.0, 1.0, 1.0], intensity: 0.8, pos: [20, 50, 30] },
    street: 0, exposure: 1.15, white_point: [0.9497, 1.0, 1.2264], fog: { rgb: [0.62, 0.64, 0.66], density: 0.035 },
    sky: [0.62, 0.64, 0.66], roadAlbedo: [0.07, 0.07, 0.075], roadRoughness: 0.9,
  },
  rain: {
    hemi: { sky: [0.55, 0.6, 0.7], ground: [0.2, 0.2, 0.22], intensity: 1.0 },
    sun: { rgb: [0.9, 0.95, 1.0], intensity: 0.4, pos: [-30, 50, 35] },
    street: 30, exposure: 1.4, white_point: [0.9678, 1.0, 1.1907], fog: { rgb: [0.4, 0.43, 0.48], density: 0.015 },
    sky: [0.3, 0.33, 0.4], roadAlbedo: [0.035, 0.035, 0.04], roadRoughness: 0.25,
  },
};

const ENV_TAG = "svEnvLight";
export const STREET_LAMPS: V3[] = [[-6, 7, 6], [6, 7, -6]];

/** Replace env lights, set fog/background/exposure/white point; recolour objects tagged userData.svEnvRole. */
export function applyEnvironment(scene: THREE.Scene, env: EnvName): void {
  const p = PRESETS[env];
  for (const o of [...scene.children]) if (o.userData[ENV_TAG]) scene.remove(o);
  const add = (o: THREE.Object3D) => { o.userData[ENV_TAG] = true; scene.add(o); };
  const hemi = new THREE.HemisphereLight(new THREE.Color(...p.hemi.sky), new THREE.Color(...p.hemi.ground), p.hemi.intensity);
  add(hemi);
  const sun = new THREE.DirectionalLight(new THREE.Color(...p.sun.rgb), p.sun.intensity);
  sun.position.set(...p.sun.pos);
  add(sun); add(sun.target);
  if (p.street > 0) {
    for (const pos of STREET_LAMPS) {
      const l = new THREE.PointLight(new THREE.Color(1.0, 0.78, 0.5), p.street, 0, 2);
      l.position.set(...pos);
      add(l);
    }
  }
  scene.fog = p.fog ? new THREE.FogExp2(new THREE.Color(...p.fog.rgb).getHex(), p.fog.density) : null;
  if (scene.fog) (scene.fog as THREE.FogExp2).color.setRGB(...p.fog!.rgb);
  scene.background = new THREE.Color(...(p.fog ? p.fog.rgb : p.sky));
  scene.userData.svEnv = { name: env, exposure: p.exposure, white_point: [...p.white_point] };
  scene.traverse((o) => {
    const role = o.userData.svEnvRole as string | undefined;
    if (!role) return;
    const mat = (o as THREE.Mesh).material as THREE.MeshStandardMaterial;
    if (role === "sky") { mat.emissive.setRGB(...p.sky); mat.emissiveIntensity = 1; }
    if (role === "road") { mat.color.setRGB(...p.roadAlbedo); mat.roughness = p.roadRoughness; }
    delete o.userData.svOrig;   // authored colours changed
  });
}
