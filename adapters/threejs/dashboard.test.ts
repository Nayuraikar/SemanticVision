import { describe, expect, it } from "vitest";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import {
  DASH_LIGHTINGS, DASH_PRESETS, DashboardSim, SYSTEMS, exportSnapshot, recordDashboard,
  type SHLighting, type Snapshot,
} from "./index";
import { getScene, sceneCards } from "./scenes/registry";

const FIX = (l: string) => path.resolve(__dirname, `../../tests/fixtures/dashboard_${l}.ndjson`);
const hash = (ss: Snapshot[]) => crypto.createHash("sha256").update(ss.map((s) => JSON.stringify(s)).join("\n")).digest("hex");
const obj = (s: Snapshot, id: string) => s.objects!.find((o) => o.id === id)!;

function close(a: unknown, b: unknown, p = ""): void {
  if (typeof a === "number" && typeof b === "number") return void expect(a, p).toBeCloseTo(b, 4);
  if (a && b && typeof a === "object") {
    expect(Object.keys(a as object).sort(), p).toEqual(Object.keys(b as object).sort());
    for (const k of Object.keys(a as object)) close((a as any)[k], (b as any)[k], `${p}.${k}`);
    return;
  }
  expect(a, p).toEqual(b);
}

const alarm = { seed: 1, scenario: "alarm" as const, env: "normal" as const };

describe("instrument-panel dashboard (held-out environment)", () => {
  it("same seed -> identical stream hash; another seed differs; no Math.random / wall clock", async () => {
    const o = { ...alarm, seed: 3, steps: 360, exportEvery: 12 };
    const a = (await recordDashboard(o)).snaps, b = (await recordDashboard(o)).snaps;
    expect(a.length).toBe(30);
    expect(hash(a)).toBe(hash(b));
    expect(hash((await recordDashboard({ ...o, seed: 4 })).snaps)).not.toBe(hash(a));
    for (const f of ["dashboardSim.ts", "scenes/dashboard.ts"]) {
      expect(fs.readFileSync(path.resolve(__dirname, f), "utf8")).not.toMatch(/Math\.random\(|Date\.now|performance\.now|new Date/);
    }
    const sim = new DashboardSim({ ...alarm, seed: 3 });
    while (sim.step < 360) sim.advance(0.037);        // uneven chunks -> same fixed-step state
    const ref = new DashboardSim({ ...alarm, seed: 3 });
    while (ref.step < sim.step) ref.tick();
    const cur = (s: DashboardSim) => exportSnapshot(s.scene, s.camera, { frame: 0, t: s.time, observerId: "x", budgetMs: 1 }).objects;
    expect(cur(sim)).toEqual(cur(ref));
  });

  it("layout: >= 24 annunciator lamps in red/amber/green legends, gauges, status lamps, panels, labels", () => {
    const s = new DashboardSim({ ...alarm, scenario: "quiet" });
    const snap = exportSnapshot(s.scene, s.camera, { frame: 0, t: 0, observerId: "x", budgetMs: 1 });
    const n = (c: string) => snap.objects!.filter((o) => o.class === c).length;
    expect(n("annunciator")).toBeGreaterThanOrEqual(24);
    expect(n("status_lamp")).toBe(SYSTEMS.length);
    expect(n("needle")).toBe(4);
    expect(n("gauge_arc")).toBe(12);
    expect(n("label")).toBeGreaterThanOrEqual(12);
    expect(n("trim") + n("display")).toBeGreaterThanOrEqual(4);
    // neighbouring tiles abut on screen (bbox adjacency), and every object is inside the viewport
    const b = (id: string) => obj(snap, id).bbox;
    expect(b("ann_hyd_1").x - (b("ann_eng_1").x + b("ann_eng_1").w)).toBeLessThanOrEqual(0);
    for (const o of snap.objects!) expect(o.screen_area).toBeGreaterThan(0);
    // quiet: no alarms ever
    for (let i = 0; i < 600; i++) s.tick();
    const later = exportSnapshot(s.scene, s.camera, { frame: 1, t: s.time, observerId: "x", budgetMs: 1 });
    expect(later.objects!.some((o) => o.task_tags!.includes("hazard") || o.state === "red" || o.state === "amber")).toBe(false);
  });

  it("one fault cascades: caution, then warning + related caution, then a second related caution", async () => {
    const { snaps, sim } = await recordDashboard({ ...alarm, steps: 480, exportEvery: 3 });
    const c = sim.cascade!;
    expect(c.caution).toBeLessThan(c.warning);
    expect(c.warning).toBeLessThan(c.spread);
    const at = (step: number) => snaps.find((s) => s.frame === Math.ceil(step / 3) - 1)!;
    const sys = c.system, [r0, r1] = c.related;
    const before = at(c.caution - 3), first = at(c.caution), esc = at(c.warning), spread = at(c.spread);
    expect(before.objects!.some((o) => o.task_tags!.includes("hazard"))).toBe(false);
    expect(obj(first, `ann_${sys}_1`).state).toBe("amber");
    expect(obj(first, `ann_${sys}_0`).state).toBe("off");
    expect(obj(first, `ann_${sys}_3`).state).toBe("off");
    expect(obj(first, "master_caution").state).toBe("amber");
    expect(obj(first, "master_warning").state).toBe("off");
    expect(obj(first, `ann_${r0}_1`).state).toBe("off");
    expect(obj(esc, `ann_${sys}_0`).state).toBe("red");
    expect(obj(esc, `ann_${sys}_0`).emissive_rgb![0]).toBeGreaterThan(0.5);
    expect(obj(esc, "master_warning").state).toBe("red");
    expect(obj(esc, `status_${sys}`).state).toBe("red");
    expect(obj(esc, `ann_${r0}_1`).state).toBe("amber");
    expect(obj(esc, `ann_${r1}_2`).state).toBe("off");
    expect(obj(spread, `ann_${r1}_2`).state).toBe("amber");
    expect(obj(spread, `g_${sys}_needle`).state).toBe("warning");
    for (const id of [`ann_${sys}_0`, `ann_${r0}_1`, "master_warning", `g_${sys}_readout`]) expect(obj(spread, id).task_tags).toContain("hazard");
    const ev = esc.events!.filter((e) => e.type === "state_change" && (e.data as any).to === "red").map((e) => e.object_id);
    expect(ev).toEqual(expect.arrayContaining([`ann_${sys}_0`, "master_warning"]));
    // warnings flash
    const red = snaps.filter((s) => s.frame! > esc.frame! + 5).map((s) => obj(s, `ann_${sys}_0`).emissive_rgb![0]);
    expect(Math.min(...red)).toBe(0);
    expect(Math.max(...red)).toBeGreaterThan(0.5);
    sim.reset();
    expect(sim.step).toBe(0);
    expect(sim.cascade).toEqual(c);
  });

  it("lighting presets differ in SH, exposure and white point; a switch emits lighting_change", () => {
    const L: SHLighting[] = DASH_LIGHTINGS.map((env) => {
      const s = new DashboardSim({ ...alarm, env });
      const l = exportSnapshot(s.scene, s.camera, { frame: 0, t: 0, observerId: "x", budgetMs: 1 }).lighting as SHLighting;
      expect(l.exposure).toBeCloseTo(DASH_PRESETS[env].exposure, 6);
      expect(l.white_point).toEqual(DASH_PRESETS[env].white_point);
      return l;
    });
    const [normal, dim, glare] = L;
    expect(new Set(L.map((l) => JSON.stringify([l.coeffs, l.exposure, l.white_point]))).size).toBe(3);
    expect(glare.coeffs[0][0]).toBeGreaterThan(normal.coeffs[0][0]);
    expect(normal.coeffs[0][0]).toBeGreaterThan(dim.coeffs[0][0]);
    const s = new DashboardSim(alarm);
    s.setLighting("glare");
    const snap = exportSnapshot(s.scene, s.camera, { frame: 0, t: 0, observerId: "x", budgetMs: 1 });
    expect(snap.events!.some((e) => e.type === "lighting_change")).toBe(true);
    expect((snap.lighting as SHLighting).exposure).toBeCloseTo(DASH_PRESETS.glare.exposure, 6);
  });

  it("records one committed fixture per lighting condition (alarm escalation window) and matches it", async () => {
    const c = new DashboardSim(alarm).cascade!;
    const every = 6, first = (Math.floor(c.warning / every) - 1) * every;
    for (const env of DASH_LIGHTINGS) {
      const { snaps } = await recordDashboard({ ...alarm, env, steps: first + 3 * every, exportEvery: every, startStep: first });
      expect(snaps.length).toBe(4);
      expect(snaps.some((s) => s.events!.some((e) => e.object_id === "master_warning" && (e.data as any).to === "red"))).toBe(true);
      const p = FIX(env);
      if (process.env.UPDATE_FIXTURE === "1" || !fs.existsSync(p)) fs.writeFileSync(p, snaps.map((s) => JSON.stringify(s)).join("\n") + "\n");
      const saved = fs.readFileSync(p, "utf8").trim().split("\n").map((l) => JSON.parse(l));
      expect(saved.length).toBe(4);
      saved.forEach((s, i) => close(snaps[i], s, `${env}.${i}`));
    }
  });

  it("registry: dashboard card enabled and builds / steps via the accumulator", () => {
    const card = sceneCards().find((x) => x.id === "dashboard")!;
    expect(card.disabled).toBe(false);
    expect(sceneCards().filter((x) => x.id === "dashboard").length).toBe(1);
    const def = getScene("dashboard")!;
    expect(def.title && def.description && def.tasks.length && def.tags.length).toBeTruthy();
    const { scene, camera } = def.build(16 / 9);
    for (let i = 0; i <= 30; i++) def.step(scene, i * 0.05);
    expect((scene.userData.svDash as DashboardSim).step).toBe(90);
    expect(exportSnapshot(scene, camera, { frame: 0, t: 1.5, observerId: "x", budgetMs: 1 }).objects!.length).toBeGreaterThan(60);
  });
});
