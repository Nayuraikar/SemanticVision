import { describe, expect, it } from "vitest";
import { buildToyScene, exportSnapshot, stepToyScene, type Snapshot } from "../index";
import { PLANNED, SCENES, getScene, sceneCards } from "./registry";

function run(build: () => { scene: any; camera: any }, step: (s: any, t: number) => void, ts: number[]): Snapshot[] {
  const { scene, camera } = build();
  const prev = new Map<string, [number, number]>();
  let prevT: number | undefined;
  return ts.map((t, f) => {
    step(scene, t);
    const s = exportSnapshot(scene, camera, { frame: f, t, observerId: "x", budgetMs: 16, prev, prevT });
    prevT = t;
    return s;
  });
}

describe("scene registry", () => {
  it("scene shape and unique ids", () => {
    expect(new Set([...SCENES, ...PLANNED].map((s) => s.id)).size).toBe(SCENES.length + PLANNED.length);
    for (const s of SCENES) {
      expect(typeof s.id).toBe("string");
      expect(s.title && s.description).toBeTruthy();
      expect(typeof s.build).toBe("function");
      expect(typeof s.step).toBe("function");
      expect(Array.isArray(s.tasks) && Array.isArray(s.tags)).toBe(true);
    }
    expect(getScene("toy")).toBeDefined();
    expect(getScene("nope")).toBeUndefined();
  });

  it("planned scenes are disabled cards", () => {
    const cards = sceneCards();
    const dis = cards.filter((c) => c.disabled).map((c) => c.id);
    expect(dis).toEqual(["cockpit", "icu", "construction", "rail-airport"]);
    expect(cards.find((c) => c.id === "toy")!.disabled).toBe(false);
    expect(cards.find((c) => c.id === "dashboard")!.disabled).toBe(false);
  });

  it("toy build/step through the registry give the same snapshots as direct calls, with state changes", () => {
    const ts = [0, 0.5, 1.9, 2.1, 3.9, 4.1];
    const toy = getScene("toy")!;
    const a = run(() => toy.build(16 / 9), toy.step, ts);
    const b = run(() => buildToyScene(16 / 9), stepToyScene, ts);
    expect(a).toEqual(b);
    const lamp = (s: Snapshot) => s.objects!.find((o) => o.id === "lamp")!.state;
    expect(a.map(lamp)).toEqual(["red", "red", "red", "green", "green", "red"]);
    const car = (s: Snapshot) => s.objects!.find((o) => o.id === "car")!.bbox;
    expect(car(a[0])).not.toEqual(car(a[2]));
  });
});
