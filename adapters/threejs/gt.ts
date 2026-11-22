// Ground-truth buffers via offscreen render targets (instance / class / state / colour).
// Training and scoring only; never consumed at runtime.
import * as THREE from "three";
import { svMeshes, type SVTag } from "./snapshot";

export interface GTBuffers {
  width: number;
  height: number;
  legend: { instance: Record<string, number>; class: Record<string, number>; state: Record<string, number> };
  /** RGBA8, row 0 = bottom (WebGL read order). Index 0 = background. */
  buffers: { instance: Uint8Array; class: Uint8Array; state: Uint8Array; color: Uint8Array };
}

/** Pack index into RGB24 (little endian); alpha 255 marks "drawn". */
export function packIndex(i: number): [number, number, number] {
  return [i & 255, (i >> 8) & 255, (i >> 16) & 255];
}
export function unpackIndex(r: number, g: number, b: number): number {
  return r | (g << 8) | (b << 16);
}

function index(names: Iterable<string>): Record<string, number> {
  const out: Record<string, number> = {};
  [...new Set(names)].sort().forEach((n, i) => (out[n] = i + 1));
  return out;
}

export function buildLegend(scene: THREE.Object3D): GTBuffers["legend"] {
  const tags = svMeshes(scene).map((m) => m.userData.sv as SVTag);
  return {
    instance: index(tags.map((t) => t.id)),
    class: index(tags.map((t) => t.cls)),
    state: index(tags.map((t) => t.state ?? "default")),
  };
}

function idPass(
  renderer: THREE.WebGLRenderer, scene: THREE.Scene, camera: THREE.Camera, rt: THREE.WebGLRenderTarget,
  code: (tag: SVTag | null) => number,
): Uint8Array {
  const swapped: [THREE.Mesh, THREE.Material | THREE.Material[], boolean][] = [];
  const mats = new Map<number, THREE.MeshBasicMaterial>();
  const matFor = (i: number) => {
    let m = mats.get(i);
    if (!m) {
      m = new THREE.MeshBasicMaterial({ toneMapped: false, fog: false });
      const [r, g, b] = packIndex(i);
      m.color.setRGB(r / 255, g / 255, b / 255, THREE.LinearSRGBColorSpace);
      mats.set(i, m);
    }
    return m;
  };
  scene.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (!mesh.isMesh) return;
    swapped.push([mesh, mesh.material, mesh.visible]);
    if (mesh.userData.svCue) mesh.visible = false;
    mesh.material = matFor(code((mesh.userData.sv as SVTag) ?? null));
  });
  const prevTarget = renderer.getRenderTarget();
  const prevClear = renderer.getClearColor(new THREE.Color());
  const prevAlpha = renderer.getClearAlpha();
  const prevBg = scene.background;
  scene.background = null;
  renderer.setClearColor(0x000000, 0);
  renderer.setRenderTarget(rt);
  renderer.clear();
  renderer.render(scene, camera);
  const buf = new Uint8Array(rt.width * rt.height * 4);
  renderer.readRenderTargetPixels(rt, 0, 0, rt.width, rt.height, buf);
  renderer.setRenderTarget(prevTarget);
  renderer.setClearColor(prevClear, prevAlpha);
  scene.background = prevBg;
  for (const [m, mat, vis] of swapped) { m.material = mat; m.visible = vis; }
  for (const m of mats.values()) m.dispose();
  return buf;
}

export function renderGroundTruth(
  renderer: THREE.WebGLRenderer, scene: THREE.Scene, camera: THREE.Camera, width: number, height: number,
): GTBuffers {
  const legend = buildLegend(scene);
  const mk = (srgb: boolean) => {
    const rt = new THREE.WebGLRenderTarget(width, height, {
      type: THREE.UnsignedByteType, format: THREE.RGBAFormat,
      minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, depthBuffer: true,
    });
    rt.texture.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
    return rt;
  };
  const rt = mk(false);
  const instance = idPass(renderer, scene, camera, rt, (t) => (t ? legend.instance[t.id] : 0));
  const cls = idPass(renderer, scene, camera, rt, (t) => (t ? legend.class[t.cls] : 0));
  const state = idPass(renderer, scene, camera, rt, (t) => (t ? legend.state[t.state ?? "default"] : 0));
  rt.dispose();
  const crt = mk(true);
  const prevTarget = renderer.getRenderTarget();
  renderer.setRenderTarget(crt);
  renderer.clear();
  renderer.render(scene, camera);
  const color = new Uint8Array(width * height * 4);
  renderer.readRenderTargetPixels(crt, 0, 0, width, height, color);
  renderer.setRenderTarget(prevTarget);
  crt.dispose();
  return { width, height, legend, buffers: { instance, class: cls, state, color } };
}

export function toBase64(a: Uint8Array): string {
  let s = "";
  for (let i = 0; i < a.length; i += 0x8000) s += String.fromCharCode(...a.subarray(i, i + 0x8000));
  return btoa(s);
}

/** JSON body for POST /gt (see core/server.py GTFrame). */
export function encodeGT(frame: number, gt: GTBuffers) {
  const b = gt.buffers;
  return {
    frame, width: gt.width, height: gt.height, legend: gt.legend,
    buffers: { instance: toBase64(b.instance), class: toBase64(b.class), state: toBase64(b.state), color: toBase64(b.color) },
  };
}

export async function postGT(baseUrl: string, frame: number, gt: GTBuffers): Promise<void> {
  const r = await fetch(`${baseUrl}/gt`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(encodeGT(frame, gt)),
  });
  if (!r.ok) throw new Error(`POST /gt ${r.status}`);
}
