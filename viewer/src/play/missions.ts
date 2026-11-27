// Mission cards and scoring (pure). A demo task log, not a validated study.
import { GAUGE_SYSTEMS, SYSTEMS } from "../../../adapters/threejs/dashboardSim";
import type { MissionName } from "../../../adapters/threejs/play/actions";
import { SEPARATION_GOAL, type Facts } from "../../../adapters/threejs/play/worlds";
import type { ViewState } from "../sim/view/layout";

export const LOG_LABEL = "demo task log, not a validated study";

export type Answer = { type: "object"; id: string } | { type: "choice"; value: string };
export interface Mission {
  id: MissionName; scene: string; title: string; task: string; hint: string;
  /** multiple choice; without it the player answers by clicking an object (unless `done`) */
  choices?: readonly string[];
  /** answered with a Done button */
  done?: boolean;
  check(a: Answer, f: Facts): boolean;
}

const obj = (a: Answer, ids: string[] | undefined) => a.type === "object" && !!ids?.includes(a.id);
const choice = (a: Answer) => (a.type === "choice" ? a.value : null);

export const MISSIONS: Mission[] = [
  { id: "brake", scene: "traffic", title: "Find the braking car", task: "Click a car whose brake lights are on.",
    hint: "Brake lights sit at the rear of the car.", check: (a, f) => obj(a, f.braking) },
  { id: "signal", scene: "traffic", title: "Read the signal at distance", task: "What colour is the east-west signal?",
    hint: "The heads face traffic on the east-west road. Zoom with the scroll wheel.", choices: ["red", "amber", "green"],
    check: (a, f) => choice(a) === f.signals?.ew },
  { id: "crash", scene: "traffic", title: "Find the crash", task: "Click a crashed car or its hazard lights.",
    hint: "Wait for the impact, then look for flashing amber.", check: (a, f) => obj(a, f.crash) },
  { id: "lineup", scene: "traffic", title: "Red car or brown car?", task: "In the parking bay, click the red car.",
    hint: "Two parked cars: one red, one brown.", check: (a, f) => a.type === "object" && a.id === f.redCar },
  { id: "fault", scene: "dashboard", title: "Which system is faulted?", task: "Name the system that raised the warning.",
    hint: "Wait for the master warning. Cautions spread to other systems.", choices: SYSTEMS,
    check: (a, f) => choice(a) === f.fault },
  { id: "levels", scene: "dashboard", title: "Warning or caution?", task: "Click a lit lamp that is a WARNING, not a caution.",
    hint: "Wait for the alarm to escalate. Warnings and cautions differ in legend colour.", check: (a, f) => obj(a, f.warnings) },
  { id: "gauge", scene: "dashboard", title: "Which gauge is out of range?", task: "Name the gauge that left its normal range.",
    hint: "Compare each needle with the arc under it.", choices: GAUGE_SYSTEMS,
    check: (a, f) => !!f.outGauges?.some((s) => s === choice(a)) },
  { id: "separate", scene: "toy", title: "Separate cone from vegetation", task: `Drag the cone at least ${SEPARATION_GOAL} m from the bush, then press Done.`,
    hint: "The two colours were swapped for close ones. Drag the object you think is the cone.", done: true,
    check: (_a, f) => (f.separation ?? 0) >= SEPARATION_GOAL },
];
export const missionsFor = (scene: string) => MISSIONS.filter((m) => m.scene === scene);

export type Condition = "A" | "B" | "C" | "mixed";
/** What the player is looking at. Split and wipe show several panes at once. */
export function viewCondition(v: ViewState): Condition {
  if (v.mode !== "single") return "mixed";
  if (v.feel >= 0.99) return v.source === "original" ? "B" : "C";
  return v.feel <= 0.01 && v.source === "original" ? "A" : "mixed";
}
export const viewFor = (c: "B" | "C"): Partial<ViewState> => ({ mode: "single", source: c === "B" ? "original" : "corrected", feel: 1 });

export interface MissionRun { mission: Mission; ms: Record<Condition, number> }
export const startRun = (mission: Mission): MissionRun => ({ mission, ms: { A: 0, B: 0, C: 0, mixed: 0 } });
export const addTime = (r: MissionRun, c: Condition, ms: number) => { r.ms[c] += Math.max(0, ms); };
export const elapsedMs = (r: MissionRun) => r.ms.A + r.ms.B + r.ms.C + r.ms.mixed;

export interface MissionRecord {
  mission: string; scene: string; observer: string; correct: boolean; answer: string;
  /** B or C only when the whole attempt was spent in that view */
  condition: Condition; seconds: number; secondsByView: Record<Condition, number>;
}

export function score(r: MissionRun, a: Answer, f: Facts, observer: string): MissionRecord {
  const total = elapsedMs(r);
  const seen = (Object.keys(r.ms) as Condition[]).filter((c) => r.ms[c] > 0);
  const s = (ms: number) => Math.round(ms) / 1000;
  return {
    mission: r.mission.id, scene: r.mission.scene, observer, correct: r.mission.check(a, f),
    answer: a.type === "object" ? a.id : a.value, condition: seen.length === 1 ? seen[0] : "mixed", seconds: s(total),
    secondsByView: { A: s(r.ms.A), B: s(r.ms.B), C: s(r.ms.C), mixed: s(r.ms.mixed) },
  };
}

export interface ConditionSummary { n: number; correct: number; meanSeconds: number | null }
export function summarize(log: MissionRecord[]): Record<"B" | "C", ConditionSummary> {
  const of = (c: "B" | "C"): ConditionSummary => {
    const xs = log.filter((r) => r.condition === c);
    return { n: xs.length, correct: xs.filter((r) => r.correct).length,
      meanSeconds: xs.length ? xs.reduce((t, r) => t + r.seconds, 0) / xs.length : null };
  };
  return { B: of("B"), C: of("C") };
}

/** In memory only; survives scene switches, not reloads. */
export const sessionLog: MissionRecord[] = [];
export const logDocument = (records: MissionRecord[]) =>
  ({ label: LOG_LABEL, views: { B: "how you see it", C: "corrected, as you see it" }, summary: summarize(records), records });
