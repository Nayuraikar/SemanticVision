import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { keyAction } from "../timeline";
import {
  DEFAULT_VIEW, MAX_FOV, SAFE_ASPECT, TILE_GAP, fitFov, isVisible, layoutTiles, lerpTiles, ndcIn, tileAt, type ViewState,
} from "./layout";
import { FOCUS_ZOOM, OrbitRig, PITCH_MAX, ZOOM_MAX, ZOOM_MIN, damp } from "./orbit";
import {
  convexHull, meshBox, meshHull, overlaps, partnersOf, pickObject, placeLabels, projectPoint, topPairs,
} from "./geometry";
import { viewKey } from "./keys";
import { inspect, litAfter, posteriorOf } from "../hud/models";
import { barPct, solveInfo } from "../hud/Diagnostics";
import { buildToyScene } from "../../../../adapters/threejs/toyScene";
import { exportSnapshot } from "../../../../adapters/threejs/snapshot";

const view = (p: Partial<ViewState>) => ({ ...DEFAULT_VIEW, ...p });

describe("layouts", () => {
  it("split gives three equal full-height tiles", () => {
    const t = layoutTiles(view({}), 1212, 700);
    expect(t.map((x) => x.pane)).toEqual(["A", "B", "C"]);
    expect(t.map((x) => [x.view.w, x.view.h])).toEqual([[400, 700], [400, 700], [400, 700]]);
    expect(t[1].view.x).toBe(400 + TILE_GAP);
    expect(t[2].view.x + t[2].view.w).toBe(1212);
    expect(t.map((x) => x.sim)).toEqual([0, 1, 1]);
  });
  it("wipe shows two full-frame panes split at the divider", () => {
    const t = layoutTiles(view({ mode: "wipe", divider: 0.25 }), 1000, 500);
    expect(t.filter(isVisible).map((x) => [x.pane, x.clip.x, x.clip.w, x.view.w])).toEqual([["B", 0, 250, 1000], ["C", 250, 750, 1000]]);
    expect(tileAt(t, 100, 10)!.pane).toBe("B");
    expect(tileAt(t, 900, 10)!.pane).toBe("C");
    expect(tileAt(t, 1100, 10)).toBeNull();
  });
  it("single shows one pane and cross-fades the CVD amount", () => {
    for (const [source, pane] of [["original", "A"], ["corrected", "C"]] as const) {
      const shown = layoutTiles(view({ mode: "single", source, feel: 0.4 }), 800, 400).filter(isVisible);
      expect(shown.map((x) => [x.pane, x.sim, x.clip.w])).toEqual([[pane, 0.4, 800]]);
    }
  });
  it("interpolates between layouts", () => {
    const a = layoutTiles(view({}), 1212, 600), b = layoutTiles(view({ mode: "wipe" }), 1212, 600);
    expect(lerpTiles(a, b, 0)).toEqual(a);
    expect(lerpTiles(a, b, 1)).toEqual(b);
    const mid = lerpTiles(a, b, 0.5);
    expect(mid.map((x) => x.view.w)).toEqual([806, 806, 806]);
  });
  it("maps pointer to NDC and widens the field of view for narrow tiles", () => {
    expect(ndcIn({ x: 100, y: 0, w: 200, h: 100 }, 200, 50)).toEqual([0, 0]);
    expect(ndcIn({ x: 0, y: 0, w: 200, h: 100 }, 0, 0)).toEqual([-1, 1]);
    expect(fitFov(40, 2)).toBe(40);
    const v = fitFov(40, 0.8);
    const horizontal = (fov: number, aspect: number) => Math.tan((fov * Math.PI) / 360) * aspect;
    expect(horizontal(v, 0.8)).toBeCloseTo(horizontal(40, SAFE_ASPECT));
    expect(fitFov(50, 0.2)).toBe(MAX_FOV);
  });
});

describe("orbit rig", () => {
  const base = () => {
    const c = new THREE.PerspectiveCamera();
    c.position.set(16, 11, 24); c.lookAt(0, 1.5, 0);
    return c;
  };
  it("keeps the authored pose at rest", () => {
    const out = new THREE.PerspectiveCamera(), b = base();
    new OrbitRig().apply(b, out);
    expect(out.position.distanceTo(b.position)).toBeLessThan(1e-9);
    expect(out.quaternion.angleTo(b.quaternion)).toBeLessThan(1e-6);
  });
  it("eases toward goals, clamps, snaps for reduced motion and resets", () => {
    const r = new OrbitRig();
    r.rotate(1, 10); r.zoomBy(100);
    expect([r.goal.pitch, r.goal.zoom]).toEqual([PITCH_MAX, ZOOM_MAX]);
    r.update(0.05);
    expect(r.cur.yaw).toBeGreaterThan(0);
    expect(r.cur.yaw).toBeLessThan(1);
    r.update(0, true);
    expect(r.cur.yaw).toBe(1);
    r.zoomBy(0);
    expect(r.goal.zoom).toBe(ZOOM_MIN);
    r.reset(); r.update(10);
    expect(r.cur.yaw).toBeCloseTo(0);
    expect(damp(0, 1, 0)).toBe(0);
  });
  it("flies to a focused point and looks at it from closer", () => {
    const r = new OrbitRig(), b = base(), out = new THREE.PerspectiveCamera(), p = new THREE.Vector3(3, 1, -2);
    r.focusOn(p); r.update(0, true); r.apply(b, out);
    expect(r.cur.zoom).toBe(FOCUS_ZOOM);
    const dir = new THREE.Vector3(0, 0, -1).applyQuaternion(out.quaternion);
    expect(dir.angleTo(p.clone().sub(out.position))).toBeLessThan(1e-6);
    expect(out.position.distanceTo(p)).toBeLessThan(b.position.distanceTo(p));
  });
});

describe("overlay geometry", () => {
  it("builds a convex hull", () => {
    expect(convexHull([[0, 0], [2, 0], [2, 2], [0, 2], [1, 1]])).toHaveLength(4);
  });
  it("places labels inside bounds without collisions, dropping what cannot fit", () => {
    const bounds = { x: 0, y: 0, w: 300, h: 200 };
    const labels = ["a", "b", "c", "d"].map((id) => ({ id, anchor: [150, 100] as [number, number], w: 80, h: 20 }));
    const out = placeLabels(labels, bounds);
    expect(out.map((l) => l.id)).toEqual(["a", "b", "c", "d"]);
    for (const a of out) {
      expect(a.x >= 0 && a.y >= 0 && a.x + a.w <= 300 && a.y + a.h <= 200).toBe(true);
      for (const b of out) if (a !== b) expect(overlaps(a, b)).toBe(false);
    }
    expect(placeLabels(labels, { x: 0, y: 0, w: 90, h: 22 })).toHaveLength(1);
  });
  const pair = (a: string, b: string, before: number, after: number, feasible: boolean) =>
    ({ ids: [a, b] as [string, string], de00_before: before, de00_after: after, feasible });
  const pairs = Array.from({ length: 12 }, (_, i) => pair("x", `y${i}`, 12 - i, i + 1, i % 2 === 0));
  it("draws only confusable pairs, worst first, top N", () => {
    expect(topPairs(pairs, "A")).toEqual([]);
    const b = topPairs(pairs, "B");
    expect(b).toHaveLength(8);
    expect(b[0].de00_before).toBe(1);
    const c = topPairs(pairs, "C", 3);
    expect(c.map((p) => p.de00_after)).toEqual([2, 4, 6]);
    expect(c.every((p) => !p.feasible)).toBe(true);
    expect(partnersOf(pairs, "y1")).toEqual([{ id: "x", before: 11, after: 2, feasible: false }]);
  });
  it("projects, hulls and picks the toy objects without touching the scene", () => {
    const { scene, camera } = buildToyScene(16 / 9);
    scene.updateMatrixWorld(true);
    const bush = scene.children.find((o) => o.userData.sv?.id === "bush") as THREE.Mesh;
    const rect = { x: 0, y: 0, w: 1600, h: 900 };
    const centre = projectPoint(bush.getWorldPosition(new THREE.Vector3()), camera, rect)!;
    const box = meshBox(bush, camera, rect)!;
    expect(centre[0] > box.x && centre[0] < box.x + box.w).toBe(true);
    expect(meshHull(bush, camera, rect).length).toBeGreaterThan(5);
    expect(pickObject(scene, camera, ndcIn(rect, centre[0], centre[1]))).toBe("bush");
    expect(pickObject(scene, camera, [0, 0.99])).toBeNull();
    expect(projectPoint(new THREE.Vector3(0, 4, 20), camera, rect)).toBeNull();
  });
});

describe("shortcuts", () => {
  it("maps view keys and leaves form typing alone", () => {
    expect(["1", "2", "3"].map((k) => viewKey(k, null))).toEqual([{ mode: "split" }, { mode: "wipe" }, { mode: "single" }]);
    expect(["I", "g", "D", "?"].map((k) => viewKey(k, null))).toEqual(
      [{ toggle: "inspector" }, { toggle: "graph" }, { toggle: "diagnostics" }, { toggle: "help" }]);
    expect(viewKey("d", { tagName: "SELECT" })).toBeNull();
    expect(viewKey("1", { tagName: "INPUT", type: "text" })).toBeNull();
    expect(viewKey("Escape", { tagName: "SELECT" })).toEqual({ action: "close" });
    expect(viewKey("x", null)).toBeNull();
    expect(keyAction("ArrowLeft", { tagName: "DIV", role: "slider" })).toBeNull();
  });
});

describe("inspector and diagnostics data", () => {
  const { scene, camera } = buildToyScene(16 / 9);
  const snap = exportSnapshot(scene, camera, { frame: 0, t: 0, observerId: "t0", budgetMs: 40 });
  const car = snap.objects!.find((o) => o.id === "car")!;
  it("recomputes the lit colour with the exporter's formula", () => {
    const same = litAfter(car, snap, { palette_overrides: [{ id: "car", albedo: car.albedo_rgb }] })!;
    same.forEach((v, i) => expect(v).toBeCloseTo(car.lit_color![i], 4));
    const dark = litAfter(car, snap, { palette_overrides: [{ id: "car", albedo: [0, 0, 0] }] })!;
    expect(dark).toEqual([0, 0, 0]);
  });
  it("collects partners, cues and the worst dE00 pair values", () => {
    const d = inspect("car", snap, {
      cues: [{ id: "car", type: "hatch" }],
      confusion_pairs: [{ ids: ["bush", "car"], de00_before: 2, de00_after: 9, feasible: true }],
    }, "allocated")!;
    expect(d).toMatchObject({ cls: "vehicle", state: "moving", cues: ["hatch"], worst: { before: 2, after: 9 }, tier: "allocated" });
    expect(d.partners[0].id).toBe("bush");
    expect(inspect("nope", snap, null, null)).toBeNull();
    expect(inspect("car", snap, null, null)).toMatchObject({ litAfter: null, worst: null, tier: "not reported" });
  });
  it("shows only the posterior fields the API returned and says what is missing", () => {
    const preset = { id: "t0", split: "train", model: "machado", kind: "deutan", severity: 0.8 };
    const p = posteriorOf("t0", [preset], [], null)!;
    expect(p.rows).toEqual([{ label: "severity", mean: 0.8, lo: null, hi: null, max: 1 }]);
    expect(p.notes.join(" ")).toMatch(/no uncertainty.*Threshold values are not reported/);
    expect(posteriorOf("t0", [{ ...preset, k: 1.2, s_L: 1, s_C: 0.9 } as typeof preset], [], null)!.rows).toHaveLength(4);
    const prof = { id: "a", observer_id: "profile:a", n_trials: 12, kind: "protan", severity: 0.6, model: "machado" };
    const bare = posteriorOf("profile:a", [], [prof], null)!;
    expect(bare.rows).toHaveLength(1);
    expect(bare.notes[0]).toMatch(/unavailable/);
    const part = posteriorOf("profile:a", [], [prof], { params: { k: { mean: 1.1, lo: 0.8, hi: 2.5 } } })!;
    expect(part.rows).toEqual([{ label: "threshold scale k", mean: 1.1, lo: 0.8, hi: 2.5, max: 2.5 }]);
    expect(part.notes.join(" ")).toMatch(/Not reported: .*deutan severity/);
    expect(posteriorOf("gone", [], [], null)).toBeNull();
  });
  it("keeps the last full solve across stabiliser-only frames", () => {
    const full = solveInfo({ diagnostics: { solver_tier: "allocated", tier_counts: { exact: 1 }, stage_latency_ms: { total: 4 } } }, null)!;
    expect(solveInfo({ diagnostics: { solver_tier: "cached", stage_latency_ms: {} } }, full)).toBe(full);
    expect(solveInfo(null, null)).toBeNull();
    expect([barPct(5, 10), barPct(20, 10), barPct(1, 0)]).toEqual([50, 100, 0]);
  });
});
