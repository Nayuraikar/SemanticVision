import { describe, expect, it } from "vitest";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import {
  ENVS, FixedStep, PRESETS, TrafficSim, exportSnapshot, findCollisionStep, initPhysics, recordTraffic,
  type SHLighting, type Snapshot,
} from "./index";
import { getScene } from "./scenes/registry";

const FIX = (env: string) => path.resolve(__dirname, `../../tests/fixtures/traffic_${env}.ndjson`);
const hash = (ss: Snapshot[]) => crypto.createHash("sha256").update(ss.map((s) => JSON.stringify(s)).join("\n")).digest("hex");
const obj = (s: Snapshot, id: string) => s.objects!.find((o) => o.id === id);

function close(a: unknown, b: unknown, p = ""): void {
  if (typeof a === "number" && typeof b === "number") return void expect(a, p).toBeCloseTo(b, 4);
  if (a && b && typeof a === "object") {
    expect(Object.keys(a as object).sort(), p).toEqual(Object.keys(b as object).sort());
    for (const k of Object.keys(a as object)) close((a as any)[k], (b as any)[k], `${p}.${k}`);
    return;
  }
  expect(a, p).toEqual(b);
}

describe("traffic intersection (Rapier, fixed step, seeded)", () => {
  it("same seed -> identical snapshot stream; different seed differs", async () => {
    const o = { seed: 3, scenario: "normal" as const, env: "day" as const, steps: 240, exportEvery: 24 };
    const a = (await recordTraffic(o)).snaps, b = (await recordTraffic(o)).snaps;
    expect(a.length).toBe(10);
    expect(hash(a)).toBe(hash(b));
    expect(hash((await recordTraffic({ ...o, seed: 4 })).snaps)).not.toBe(hash(a));
  });

  it("accumulator: uneven elapsed chunks give the same state as whole steps", async () => {
    const fs1 = new FixedStep(1 / 60);
    expect(fs1.advance(0.05, () => {})).toBe(3);
    expect(fs1.advance(0.012, () => {})).toBe(0);     // carried over
    expect(fs1.advance(0.006, () => {})).toBe(1);     // 0.018 >= dt
    expect(fs1.acc).toBeCloseTo(0.018 - 1 / 60, 9);
    await initPhysics();
    const A = new TrafficSim({ seed: 2, scenario: "normal", env: "day" });
    const B = new TrafficSim({ seed: 2, scenario: "normal", env: "day" });
    while (B.step < 180) B.advance(0.037);
    while (A.step < B.step) A.tick();
    const pos = (s: TrafficSim) => [...s.physics!.bodies.entries()].map(([k, b]) => [k, b.translation().x, b.translation().z]);
    expect(pos(B)).toEqual(pos(A));
    A.physics!.free(); B.physics!.free();
  });

  it("scripted collision: exactly one collision event, damaged + hazard on, importance bump, trigger entry", async () => {
    const { snaps, sim } = await recordTraffic({ seed: 1, scenario: "collision", env: "day", steps: 480, exportEvery: 6 });
    const col = snaps.flatMap((s) => (s.events ?? []).filter((e) => e.type === "collision").map((e) => ({ f: s.frame!, e })));
    expect(col.length).toBe(1);
    expect(sim.log.filter((e) => e.kind === "collision").length).toBe(1);
    const { f, e } = col[0];
    const a = e.object_id!, b = (e.data as any).other as string;
    const k = snaps.findIndex((s) => s.frame === f);
    const before = snaps[k - 1], after = snaps[k + 3];
    for (const id of [a, b]) {
      expect(obj(before, id)!.state).not.toBe("damaged");
      expect(obj(before, id)!.task_tags).not.toContain("hazard");
      expect(obj(snaps[k], id)!.state).toBe("damaged");
      expect(obj(after, id)!.state).toBe("damaged");
      expect(obj(after, id)!.task_tags).toEqual(expect.arrayContaining(["hazard", "collision"]));
      expect(obj(after, `${id}_ind`)!.state).toBe("hazard");
    }
    const ev = snaps[k].events!;
    expect(ev.some((x) => x.type === "state_change" && x.object_id === a && (x.data as any).to === "damaged")).toBe(true);
    // hazard indicators: beacon appears and flashes (emissive), smoke marker present
    expect(obj(before, "hazard_beacon")).toBeUndefined();
    const beacon = snaps.slice(k).map((s) => obj(s, "hazard_beacon")!);
    expect(beacon.every((x) => x && x.task_tags!.includes("hazard"))).toBe(true);
    expect(new Set(beacon.map((x) => x.emissive_rgb!.join())).size).toBeGreaterThan(1);
    expect(Math.max(...beacon[0].emissive_rgb!)).toBeGreaterThan(1);
    expect(obj(after, "smoke")).toBeDefined();
    const ind = snaps.slice(k).map((s) => obj(s, `${a}_ind`)!.emissive_rgb![0]);
    expect(Math.max(...ind)).toBeGreaterThan(1);
    expect(Math.min(...ind)).toBe(0);
    // nearby objects got the importance bump (hazard tag) and react
    const near = after.objects!.filter((o) => o.task_tags!.includes("near_collision"));
    expect(near.length).toBeGreaterThan(0);
    for (const o of near) expect(o.task_tags).toContain("hazard");
    // trigger volume entries: pedestrians entering crosswalks, vehicles entering the junction
    const trig = snaps.flatMap((s) => (s.events ?? []).filter((x) => x.type === "trigger_enter"));
    expect(trig.some((x) => x.object_id!.startsWith("ped") && String((x.data as any).trigger).startsWith("cw_"))).toBe(true);
    expect(trig.some((x) => x.object_id === a && (x.data as any).trigger === "box_junction")).toBe(true);
    expect(snaps.some((s) => s.events!.some((x) => x.type === "state_change" && (x.data as any).to === "crossing"))).toBe(true);
    // brake lights: some vehicle brakes (brake light emissive on) before the crash
    expect(snaps.slice(0, k).some((s) => s.objects!.some((o) => o.id.endsWith("_brake") && o.state === "on" && o.emissive_rgb![0] > 1))).toBe(true);
  });

  it("normal flow has no collisions and cycles signals red/green/amber", async () => {
    const { snaps, sim } = await recordTraffic({ seed: 5, scenario: "normal", env: "day", steps: 1400, exportEvery: 700 });
    expect(sim.log.filter((e) => e.kind === "collision").length).toBe(0);
    expect(snaps.length).toBe(2);
    const st = new Set<string>();
    const s2 = await recordTraffic({ seed: 5, scenario: "normal", env: "day", steps: 1400, exportEvery: 60 });
    for (const s of s2.snaps) for (const o of s.objects!) if (o.id === "sig_E_red" || o.id === "sig_E_amber" || o.id === "sig_E_green") st.add(o.state!);
    expect([...st].sort()).toEqual(["amber", "green", "off", "red"]);
  });

  it("each environment preset exports different lighting; emission not scaled by irradiance", async () => {
    await initPhysics();
    const L: SHLighting[] = [];
    for (const env of ENVS) {
      const sim = new TrafficSim({ seed: 1, scenario: "normal", env });
      const s = exportSnapshot(sim.scene, sim.camera, { frame: 0, t: 0, observerId: "x", budgetMs: 1 });
      const l = s.lighting as SHLighting;
      expect(l.exposure).toBeCloseTo(PRESETS[env].exposure, 6);
      expect(l.white_point).toEqual(PRESETS[env].white_point);
      expect(l.fog_density ?? 0).toBeCloseTo(PRESETS[env].fog?.density ?? 0, 6);
      const sky = obj(s, "sky")!;   // albedo 0: lit = emissive * exposure, whatever the irradiance
      for (let c = 0; c < 3; c++) expect(sky.lit_color![c]).toBeCloseTo(Math.min(1, PRESETS[env].sky[c] * l.exposure!), 5);
      L.push(l);
      sim.physics?.free();
    }
    const keys = L.map((l) => JSON.stringify([l.coeffs, l.exposure, l.white_point, l.fog_density ?? null]));
    expect(new Set(keys).size).toBe(ENVS.length);
    expect(L[0].coeffs[0][0]).toBeGreaterThan(L[1].coeffs[0][0]);          // day brighter than night
    expect(L[2].fog_density!).toBeGreaterThan(L[3].fog_density!);          // fog denser than rain
  });

  it("records one committed fixture per environment (collision window) and matches it", async () => {
    const c = await findCollisionStep({ seed: 1, scenario: "collision", env: "day" });
    expect(c).toBeGreaterThan(0);
    const every = 6, first = (Math.floor(c / every) - 1) * every;
    for (const env of ENVS) {
      const { snaps } = await recordTraffic({ seed: 1, scenario: "collision", env, steps: first + 3 * every, exportEvery: every, startStep: first });
      expect(snaps.length).toBe(4);
      expect(snaps.flatMap((s) => s.events!).filter((e) => e.type === "collision").length).toBe(1);
      const p = FIX(env);
      if (process.env.UPDATE_FIXTURE === "1" || !fs.existsSync(p)) fs.writeFileSync(p, snaps.map((s) => JSON.stringify(s)).join("\n") + "\n");
      const saved = fs.readFileSync(p, "utf8").trim().split("\n").map((l) => JSON.parse(l));
      expect(saved.length).toBe(4);
      saved.forEach((s, i) => close(snaps[i], s, `${env}.${i}`));
    }
  });

  it("registry scene builds and steps via the accumulator", async () => {
    await initPhysics();
    const def = getScene("traffic")!;
    const { scene, camera } = def.build(16 / 9);
    for (let i = 0; i <= 30; i++) def.step(scene, i * 0.05);
    const sim = scene.userData.svSim as TrafficSim;
    expect(sim.step).toBe(90);
    const s = exportSnapshot(scene, camera, { frame: 0, t: 1.5, observerId: "x", budgetMs: 1 });
    expect(s.objects!.some((o) => o.class === "light" && o.state === "green")).toBe(true);
    sim.physics?.free();
  });
});
