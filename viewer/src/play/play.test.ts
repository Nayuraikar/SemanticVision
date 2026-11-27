import { describe, expect, it } from "vitest";
import { GROUNDS } from "../../../adapters/threejs/play/movement";
import type { Facts } from "../../../adapters/threejs/play/worlds";
import { DEFAULT_VIEW, type ViewState } from "../sim/view/layout";
import { Beeper, gaugeValueAt, keysFrom, moveKey } from "./input";
import { LOG_LABEL, MISSIONS, addTime, logDocument, missionsFor, score, startRun, viewCondition, viewFor, type Answer } from "./missions";
import { mapScale } from "./PlayHud";

const mission = (id: string) => MISSIONS.find((m) => m.id === id)!;
const FACTS: Facts = {
  hazards: [], alarms: 0, braking: ["veh1"], crash: ["veh2", "hazard_beacon"], redCar: "park_b", signals: { ew: "amber", ns: "red" },
  fault: "hyd", warnings: ["ann_hyd_0"], cautions: ["ann_eng_1"], outGauges: ["fuel"], separation: 3.5,
};
const RIGHT: Record<string, Answer> = {
  brake: { type: "object", id: "veh1" }, signal: { type: "choice", value: "amber" }, crash: { type: "object", id: "hazard_beacon" },
  lineup: { type: "object", id: "park_b" }, fault: { type: "choice", value: "hyd" }, levels: { type: "object", id: "ann_hyd_0" },
  gauge: { type: "choice", value: "fuel" }, separate: { type: "choice", value: "done" },
};

describe("mission scoring", () => {
  it("every mission accepts the right answer and rejects a wrong one", () => {
    for (const m of MISSIONS) {
      expect(m.check(RIGHT[m.id], FACTS), m.id).toBe(true);
      const wrong: Answer = m.choices ? { type: "choice", value: "nope" } : { type: "object", id: "ann_eng_1" };
      expect(m.check(wrong, m.done ? { ...FACTS, separation: 1 } : FACTS), m.id).toBe(false);
    }
    expect(["traffic", "dashboard", "toy"].map((s) => missionsFor(s).length)).toEqual([4, 3, 1]);
  });

  it("attributes time to B or C only when the whole attempt stayed in that view", () => {
    const v = (p: Partial<ViewState>) => viewCondition({ ...DEFAULT_VIEW, ...p });
    expect([v(viewFor("B")), v(viewFor("C")), v({ mode: "single" }), v({}), v({ mode: "wipe" })]).toEqual(["B", "C", "A", "mixed", "mixed"]);
    const run = startRun(mission("signal"));
    addTime(run, "B", 1500); addTime(run, "B", 500);
    const b = score(run, RIGHT.signal, FACTS, "t0");
    expect([b.condition, b.seconds, b.correct]).toEqual(["B", 2, true]);
    addTime(run, "C", 1000);
    const mixed = score(run, { type: "choice", value: "red" }, FACTS, "t0");
    expect([mixed.condition, mixed.correct, mixed.secondsByView.C]).toEqual(["mixed", false, 1]);
    const doc = logDocument([b, mixed, { ...b, condition: "C", correct: false, seconds: 4 }]);
    expect(doc.label).toBe(LOG_LABEL);
    expect(doc.summary).toEqual({ B: { n: 1, correct: 1, meanSeconds: 2 }, C: { n: 1, correct: 0, meanSeconds: 4 } });
  });
});

describe("play input helpers", () => {
  it("maps movement keys and leaves form controls alone", () => {
    expect(moveKey("W", null)).toBe("w");
    expect(moveKey("ArrowLeft", { tagName: "INPUT", type: "range" })).toBeNull();
    expect(moveKey("i", null)).toBeNull();
    expect(keysFrom(new Set(["w", "arrowright", "q", "shift"]))).toEqual({ forward: 1, right: 1, turn: -1, sprint: true });
  });

  it("converts a needle drag to a gauge value and keeps the minimap invertible", () => {
    const at = (deg: number) => gaugeValueAt(Math.cos((deg * Math.PI) / 180), -Math.sin((deg * Math.PI) / 180));
    expect(at(210)).toBeCloseTo(0, 6);
    expect(at(90)).toBeCloseTo(0.5, 6);
    expect(at(-30)).toBeCloseTo(1, 6);
    expect(at(-80)).toBe(1);
    const m = mapScale(GROUNDS.traffic);
    const [x, z] = m.toWorld(...m.toMap(5, -7));
    expect(x).toBeCloseTo(5, 6); expect(z).toBeCloseTo(-7, 6);
  });

  it("alarm beep is muted by default", () => {
    const b = new Beeper();
    expect(b.muted).toBe(true);
    expect(() => b.beep()).not.toThrow();
  });
});
