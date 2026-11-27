// Pure timeline helpers: event extraction from the sim's own log, seek/scrub/playback mapping.
import type { Event } from "../../../adapters/threejs/protocol.gen";

export const STEP_S = 1 / 60;
export const SPEEDS = [0.25, 0.5, 1, 1.5, 2];
export const SCRUB_STEPS = 30;
/** events from the last N ticks of a seek are kept for the next exported snapshot */
export const EVENT_WINDOW = 15;
const STATE_MERGE = 30;

export type MarkerKind = "collision" | "cascade" | "signal" | "state";
export interface Marker { step: number; kind: MarkerKind; label: string; ids: string[]; point?: [number, number, number] }
export interface LogEntry { step: number; events: Event[] }

export const kindOf = (e: Event): MarkerKind | null => {
  if (e.type === "collision") return "collision";
  if (e.type !== "state_change") return null;
  const id = e.object_id ?? "";
  if (id.startsWith("sig_")) return "signal";
  return /^(master_|status_|ann_)/.test(id) ? "cascade" : "state";
};

const LABEL: Record<MarkerKind, string> = { collision: "collision", cascade: "alarm cascade", signal: "signal change", state: "state change" };

/** One marker per kind per step; routine state changes closer than half a second are merged. Sorted by step. */
export function extractMarkers(log: LogEntry[]): Marker[] {
  const out: Marker[] = [];
  const last = new Map<MarkerKind, Marker>();
  for (const { step, events } of [...log].sort((a, b) => a.step - b.step)) {
    for (const e of events) {
      const kind = kindOf(e);
      if (!kind) continue;
      let m = last.get(kind);
      if (!m || step - m.step > (kind === "state" ? STATE_MERGE : 0)) {
        m = { step, kind, label: LABEL[kind], ids: [] };
        last.set(kind, m); out.push(m);
      }
      const data = (e.data ?? {}) as { other?: string; point?: [number, number, number] };
      for (const id of [e.object_id, data.other]) if (id && !m.ids.includes(id)) m.ids.push(id);
      if (kind === "collision" && data.point) m.point = data.point;
    }
  }
  return out.sort((a, b) => a.step - b.step);
}

export const clampStep = (s: number, total: number) => Math.min(total, Math.max(0, Math.round(s)));
export const stepFromFraction = (f: number, total: number) => clampStep(f * total, total);
export const fractionOfStep = (s: number, total: number) => (total > 0 ? clampStep(s, total) / total : 0);
export const scrub = (s: number, dir: -1 | 1, total: number) => clampStep(s + dir * SCRUB_STEPS, total);
export const formatTime = (step: number) => `${(step * STEP_S).toFixed(1)}s`;

/** A sim only runs forwards: going back means a fresh sim ticked from zero. */
export const seekPlan = (cur: number, target: number) =>
  target >= cur ? { rebuild: false, ticks: target - cur } : { rebuild: true, ticks: target };

/** Next marker strictly after `step`, wrapping to the first; null when there are none. */
export const nextMarker = (ms: Marker[], step: number): Marker | null => ms.find((m) => m.step > step) ?? ms[0] ?? null;

/** Most important marker in (from, to], ignoring routine state changes. */
export function crossedMarker(ms: Marker[], from: number, to: number): Marker | null {
  const rank: MarkerKind[] = ["collision", "cascade", "signal"];
  const hit = ms.filter((m) => m.step > from && m.step <= to && m.kind !== "state");
  return hit.sort((a, b) => rank.indexOf(a.kind) - rank.indexOf(b.kind))[0] ?? null;
}

/** Fixed-step playback accumulator; elapsed is capped so a stalled tab does not fast-forward. */
export function advanceClock(acc: number, elapsed: number, speed: number) {
  const total = acc + Math.min(0.25, Math.max(0, elapsed)) * speed;
  const steps = Math.floor(total / STEP_S + 1e-9);
  return { steps, acc: Math.max(0, total - steps * STEP_S) };
}

export type KeyAction = "toggle" | "back" | "forward" | null;
/** Shortcuts stay out of the way of form controls that use the same keys. */
export function keyAction(key: string, target: { tagName?: string; type?: string; role?: string | null } | null): KeyAction {
  const tag = target?.tagName ?? "", type = target?.type ?? "";
  if (tag === "SELECT" || tag === "TEXTAREA" || target?.role === "radio") return null;
  if (tag === "INPUT" && type !== "range" && type !== "checkbox") return null;
  if (key === " ") return tag === "BUTTON" || type === "checkbox" ? null : "toggle";
  return key === "ArrowLeft" ? "back" : key === "ArrowRight" ? "forward" : null;
}
