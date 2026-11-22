import { describe, expect, it } from "vitest";
import fs from "node:fs";
import path from "node:path";
import {
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
    expect(s.lighting.kind).toBe("ambient_dominant");
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
