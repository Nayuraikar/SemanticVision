// Apply Overrides (palette_overrides, cues) to Three.js materials (SPEC 4.3).
import * as THREE from "three";
import type { Cue, Overrides } from "./protocol.gen";
import { originalColors, svMeshes, type SVMesh } from "./snapshot";

const CUE_FLAG = "svCue";

function clearCues(m: THREE.Object3D) {
  for (const c of [...m.children]) {
    if (c.userData[CUE_FLAG]) {
      m.remove(c);
      (c as THREE.Mesh).geometry?.dispose?.();
      const mat = (c as THREE.Mesh).material as THREE.Material | undefined;
      mat?.dispose?.();
    }
  }
}

function rgb(p: unknown, def: [number, number, number]): [number, number, number] {
  const a = p as number[] | undefined;
  return Array.isArray(a) && a.length === 3 ? [a[0], a[1], a[2]] : def;
}

function makeCue(m: SVMesh, cue: Cue): THREE.Object3D | null {
  const p = cue.parameters as Record<string, unknown>;
  const col = rgb(p.color, [1, 1, 1]);
  let o: THREE.Object3D | null = null;
  if (cue.type === "outline") {
    const w = typeof p.width === "number" ? p.width : 0.06;
    const mat = new THREE.MeshBasicMaterial({ side: THREE.BackSide, toneMapped: false });
    mat.color.setRGB(col[0], col[1], col[2]);
    o = new THREE.Mesh(m.geometry, mat);
    o.scale.setScalar(1 + w);
  } else if (cue.type === "hatch") {
    const mat = new THREE.MeshBasicMaterial({ wireframe: true, toneMapped: false });
    mat.color.setRGB(col[0], col[1], col[2]);
    o = new THREE.Mesh(m.geometry, mat);
    o.scale.setScalar(1.005);
  } else if (typeof document !== "undefined") {
    // icon | label: text sprite above the object
    const text = String((cue.type === "icon" ? p.glyph : p.text) ?? cue.id);
    const cv = document.createElement("canvas");
    cv.width = 256; cv.height = 64;
    const g = cv.getContext("2d");
    if (g) {
      g.font = "bold 40px sans-serif"; g.textAlign = "center"; g.textBaseline = "middle";
      g.fillStyle = "#fff"; g.strokeStyle = "#000"; g.lineWidth = 6;
      g.strokeText(text, 128, 32); g.fillText(text, 128, 32);
    }
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(cv), toneMapped: false }));
    m.geometry.computeBoundingBox();
    const top = m.geometry.boundingBox?.max.y ?? 0.5;
    sp.position.set(0, top * 1.4 + 0.3, 0);
    sp.scale.set(1.6, 0.4, 1);
    o = sp;
  }
  if (o) o.userData[CUE_FLAG] = true;
  return o;
}

/** Idempotent: objects absent from the overrides are restored to their authored colours. */
export function applyOverrides(scene: THREE.Object3D, ov: Overrides): void {
  const po = new Map((ov.palette_overrides ?? []).map((p) => [p.id, p]));
  const cues = new Map<string, Cue[]>();
  for (const c of ov.cues ?? []) cues.set(c.id, [...(cues.get(c.id) ?? []), c]);
  for (const m of svMeshes(scene)) {
    const id = m.userData.sv.id as string;
    m.userData.svOrig ??= originalColors(m);
    const orig = m.userData.svOrig as ReturnType<typeof originalColors>;
    const p = po.get(id);
    const a = p ? p.albedo : orig.albedo;
    const e = p?.emissive ?? orig.emissive;
    m.material.color.setRGB(a[0], a[1], a[2]);
    // emissive: fold intensity into the colour so HDR values round-trip
    m.material.emissiveIntensity = 1;
    m.material.emissive.setRGB(e[0], e[1], e[2]);
    clearCues(m);
    for (const c of cues.get(id) ?? []) {
      const o = makeCue(m, c);
      if (o) m.add(o);
    }
  }
}
