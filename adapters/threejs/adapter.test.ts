import { describe, expect, it } from "vitest";
import fs from "node:fs";
import path from "node:path";
import * as THREE from "three";
import {
  exportLightingSH, irradiance, litColor, type SHLighting,
  applyOverrides, buildToyScene, encodeGT, exportSnapshot, packIndex, stepToyScene, svMeshes, unpackIndex,
  type Overrides, type Snapshot,
} from "./index";

const FIXTURE = path.resolve(__dirname, "../../tests/fixtures/toy_scene.ndjson");

function record(n = 24, dt = 0.25): Snapshot[] {
  const { scene, camera } = buildToyScene(16 / 9);
  const prev = new Map<string, [number, number]>();
  const out: Snapshot[] = [];
  let prevT: number | undefined;
  for (let f = 0; f < n; f++) {
    const t = f * dt;
    stepToyScene(scene, t);
    out.push(exportSnapshot(scene, camera, { frame: f, t, observerId: "toy", budgetMs: 16, prev, prevT }));
    prevT = t;
  }
  return out;
}

function close(a: unknown, b: unknown, p = ""): void {
  if (typeof a === "number" && typeof b === "number") return void expect(a, p).toBeCloseTo(b, 4);
  if (a && b && typeof a === "object") {
    expect(Object.keys(a as object).sort(), p).toEqual(Object.keys(b as object).sort());
    for (const k of Object.keys(a as object)) close((a as any)[k], (b as any)[k], `${p}.${k}`);
    return;
  }
  expect(a, p).toEqual(b);
}

describe("three.js adapter", () => {
  it("records the toy scene to the committed NDJSON fixture", () => {
    const snaps = record();
    if (process.env.UPDATE_FIXTURE === "1" || !fs.existsSync(FIXTURE)) {
      fs.mkdirSync(path.dirname(FIXTURE), { recursive: true });
      fs.writeFileSync(FIXTURE, snaps.map((s) => JSON.stringify(s)).join("\n") + "\n");
    }
    const saved = fs.readFileSync(FIXTURE, "utf8").trim().split("\n").map((l) => JSON.parse(l));
    expect(saved.length).toBe(snaps.length);
    saved.forEach((s, i) => close(snaps[i], s, `frame${i}`));
  });

  it("exports all fields with sane values", () => {
    const s = record(8)[7];
    expect(s.objects!.map((o) => o.id).sort()).toEqual(["bush", "car", "cone", "lamp"]);
    for (const o of s.objects!) {
      expect(o.screen_area).toBeGreaterThan(0);
      expect(o.screen_area).toBeLessThanOrEqual(1);
      expect(o.depth).toBeGreaterThan(0);
    }
    const lamp = s.objects!.find((o) => o.id === "lamp")!;
    expect(Math.max(...lamp.emissive_rgb!)).toBeGreaterThan(1); // HDR emissive
    expect(s.objects!.find((o) => o.id === "car")!.velocity![0]).not.toBe(0);
    expect(s.lighting.kind).toBe("sh2");
    for (const o of s.objects!) {
      expect(o.normal).toBeTruthy();
      expect(o.position).toBeTruthy();
      expect(o.shadow_vis).not.toBeNull();
      expect(o.lit_color!.every((v) => v >= 0 && v <= 1)).toBe(true);
    }
  });

  it("exports SH-L2 lighting that reproduces the Three.js diffuse model", () => {
    const { scene } = buildToyScene();
    const L = exportLightingSH(scene, new THREE.Vector3());
    expect(L.coeffs.length).toBe(9);
    const sun = L.dominant!;
    // toy scene: ambient 0.4 (white), sun 2.0 (0xfff2e0): three.js E = A/pi + (I/pi) cos+
    const A = 0.4 / Math.PI, I = 2.0 / Math.PI;
    const sunRgb = new THREE.Color(0xfff2e0).toArray();
    const l = new THREE.Vector3(5, 8, 4).normalize();
    for (const n of [l.clone(), new THREE.Vector3(0, 1, 0), new THREE.Vector3(1, 0, 0), l.clone().negate()]) {
      const E = irradiance(L, [n.x, n.y, n.z]);
      const cos = Math.max(0, n.dot(l));
      for (let k = 0; k < 3; k++) expect(Math.abs(E[k] - (A + I * sunRgb[k] * cos))).toBeLessThan(0.1 * I * sunRgb[k] + 1e-3);
    }
    expect(sun.rgb[0]).toBeCloseTo(I * sunRgb[0], 4);
    // shadow term removes the dominant part only; point-light position changes direction/falloff
    const E1 = irradiance(L, [l.x, l.y, l.z], null, 1), E0 = irradiance(L, [l.x, l.y, l.z], null, 0);
    expect(E1[0]).toBeGreaterThan(E0[0] + 0.5 * I);
    expect(E0[0]).toBeGreaterThan(0);                // ambient stays
    expect(irradiance(L, null)[0]).toBeCloseTo(0.2821 * L.coeffs[0][0], 3);
  });

  it("exports lit_color = albedo*E + emissive (emission added, not scaled)", () => {
    const s = record(1)[0];
    const L = s.lighting as SHLighting;
    for (const o of s.objects!) {
      const E = irradiance(L, o.normal as [number, number, number], o.position as [number, number, number], o.shadow_vis!);
      const want = litColor(o.albedo_rgb as [number, number, number], o.emissive_rgb as [number, number, number], E, L.exposure ?? 1);
      for (let k = 0; k < 3; k++) expect(o.lit_color![k]).toBeCloseTo(want[k], 4);
    }
    const lamp = s.objects!.find((o) => o.id === "lamp")!;
    expect(lamp.lit_color![0]).toBe(1);                // HDR emissive saturates, independent of irradiance
  });

  it("raycasts shadow visibility of the dominant light", () => {
    const { scene, camera } = buildToyScene();
    const car = svMeshes(scene).find((m) => m.userData.sv.id === "car")!;
    const free = exportSnapshot(scene, camera, { frame: 0, t: 0, observerId: "x", budgetMs: 1 }).objects!.find((o) => o.id === "car")!;
    expect(free.shadow_vis).toBe(1);
    // slab between the car and the sun (sun at (5,8,4) looking at the origin)
    const slab = new THREE.Mesh(new THREE.BoxGeometry(6, 6, 0.3), new THREE.MeshStandardMaterial());
    const dir = new THREE.Vector3(5, 8, 4).normalize();
    slab.position.copy(car.position).addScaledVector(dir, 2.5);
    slab.lookAt(slab.position.clone().add(dir));
    scene.add(slab);
    const shaded = exportSnapshot(scene, camera, { frame: 0, t: 0, observerId: "x", budgetMs: 1 }).objects!.find((o) => o.id === "car")!;
    expect(shaded.shadow_vis).toBeLessThan(0.5);
    expect(shaded.lit_color![0]).toBeLessThan(free.lit_color![0]);
  });

  it("applies palette overrides, restores them, and keeps exported albedo authored", () => {
    const { scene, camera } = buildToyScene();
    const car = svMeshes(scene).find((m) => m.userData.sv.id === "car")!;
    const before = car.material.color.toArray().slice(0, 3);
    const ov: Overrides = { palette_overrides: [{ id: "car", albedo: [0.1, 0.2, 0.9] }], cues: [{ id: "car", type: "outline", parameters: {} }] };
    applyOverrides(scene, ov);
    expect(car.material.color.b).toBeCloseTo(0.9);
    expect(car.children.length).toBe(1);
    const snap = exportSnapshot(scene, camera, { frame: 0, t: 0, observerId: "x", budgetMs: 1 });
    const c = snap.objects!.find((o) => o.id === "car")!;
    expect(c.albedo_rgb![0]).toBeCloseTo(before[0], 4);
    expect(c.bbox.w).toBeLessThan(0.5); // outline child does not inflate bbox
    applyOverrides(scene, { palette_overrides: [], cues: [] });
    expect(car.material.color.toArray().slice(0, 3)).toEqual(before);
    expect(car.children.length).toBe(0);
  });

  it("packs ids and encodes GT bodies", () => {
    for (const i of [0, 1, 255, 256, 70000]) expect(unpackIndex(...packIndex(i))).toBe(i);
    const px = new Uint8Array(2 * 2 * 4).fill(7);
    const b = encodeGT(3, { width: 2, height: 2, legend: { instance: {}, class: {}, state: {} }, buffers: { instance: px, class: px, state: px, color: px } });
    expect(Buffer.from(b.buffers.instance, "base64").length).toBe(16);
  });
});
