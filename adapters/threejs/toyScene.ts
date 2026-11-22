// Deterministic toy scene: a few coloured objects, one emissive.
import * as THREE from "three";

function mk(
  geo: THREE.BufferGeometry, id: string, cls: string, state: string, albedo: [number, number, number],
  pos: [number, number, number], tags: string[], emissive?: [number, number, number],
) {
  const mat = new THREE.MeshStandardMaterial({ roughness: 0.7, metalness: 0 });
  mat.color.setRGB(albedo[0], albedo[1], albedo[2]); // linear
  if (emissive) { mat.emissive.setRGB(emissive[0], emissive[1], emissive[2]); mat.emissiveIntensity = 1; }
  const m = new THREE.Mesh(geo, mat);
  m.position.set(...pos);
  m.userData.sv = { id, cls, state, tags };
  return m;
}

export const LIGHT_STATES: Record<string, [number, number, number]> = {
  red: [4, 0.2, 0.1],
  green: [0.1, 4, 0.4],
};

export function buildToyScene(aspect = 16 / 9) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0.05, 0.06, 0.08);
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(30, 30), new THREE.MeshStandardMaterial({ color: 0x303030 }));
  floor.rotation.x = -Math.PI / 2;
  scene.add(floor);
  scene.add(
    mk(new THREE.BoxGeometry(2, 1, 1), "car", "vehicle", "moving", [0.8, 0.05, 0.05], [-3, 0.5, 0], ["hazard"]),
    mk(new THREE.SphereGeometry(0.6, 24, 16), "bush", "vegetation", "static", [0.1, 0.5, 0.08], [2, 0.6, -1], []),
    mk(new THREE.ConeGeometry(0.4, 1, 16), "cone", "marker", "static", [0.9, 0.35, 0.02], [0, 0.5, 2], ["warning"]),
    mk(new THREE.BoxGeometry(0.4, 0.4, 0.4), "lamp", "light", "red", [0.05, 0.05, 0.05], [4, 2, -2], ["signal"], LIGHT_STATES.red),
  );
  const amb = new THREE.AmbientLight(0xffffff, 0.4);
  const sun = new THREE.DirectionalLight(0xfff2e0, 2.0);
  sun.position.set(5, 8, 4);
  scene.add(amb, sun, sun.target);
  const camera = new THREE.PerspectiveCamera(50, aspect, 0.1, 100);
  camera.position.set(0, 4, 10);
  camera.lookAt(0, 0.5, 0);
  camera.updateMatrixWorld(true);
  return { scene, camera };
}

/** Deterministic animation: car drives along x; lamp toggles every 2 s. */
export function stepToyScene(scene: THREE.Scene, t: number) {
  scene.traverse((o) => {
    const sv = o.userData.sv as { id: string; state?: string } | undefined;
    if (!sv) return;
    if (sv.id === "car") o.position.x = -3 + 6 * ((t * 0.25) % 1);
    if (sv.id === "lamp") {
      const st = Math.floor(t / 2) % 2 ? "green" : "red";
      sv.state = st;
      const mat = (o as THREE.Mesh).material as THREE.MeshStandardMaterial;
      const e = LIGHT_STATES[st];
      // authored colour changes; a corrected colour is re-applied by the next applyOverrides
      if (o.userData.svOrig) o.userData.svOrig.emissive = [...e];
      else mat.emissive.setRGB(e[0], e[1], e[2]);
    }
  });
}
