// Pure rules for decorative effects, waypoint selection and event-follow easing.
import { easeInOutCubic } from "../ui/budget";
import { importanceOf, type OverlayFlags } from "../overlays";
import type { SceneObject } from "../../../adapters/threejs/protocol.gen";

export interface FxToggles { dust: boolean; waypoints: boolean; cinematic: boolean; follow: boolean }
export const DEFAULT_FX: FxToggles = { dust: true, waypoints: true, cinematic: false, follow: false };
export type HudShow = Record<"tier" | "latency" | "stats" | "posterior", boolean>;

/** Names of the visible measurements that force cinematic grading off. */
export function cinematicBlockers(flags: OverlayFlags, hud: HudShow): string[] {
  const on: [boolean, string][] = [
    [flags.labels, "ΔE00 labels"], [flags.graph, "confusion graph"], [flags.heatmap, "importance heatmap"],
    [hud.tier, "solver tier"], [hud.latency, "latency"], [hud.stats, "stats"],
  ];
  return on.filter(([v]) => v).map(([, n]) => n);
}

export const cinematicTooltip = (blockers: string[], performance: boolean) =>
  performance ? "Off in performance mode."
    : blockers.length ? `Off so measured colours stay accurate. Hide: ${blockers.join(", ")}.`
      : "Bloom, vignette and warm grade. Changes on-screen colours.";

export function effectiveFx(t: FxToggles, blockers: string[], performance: boolean, still: boolean) {
  const motion = !performance && !still;
  return {
    cinematic: t.cinematic && !performance && blockers.length === 0,
    dust: t.dust && motion, weather: motion, waypoints: t.waypoints, animate: motion,
  };
}

export interface Waypoint { key: string; position: [number, number, number]; score: number }

/** Parts belong to their parent: veh0_brake -> veh0, sig_E_red -> sig_E, g_eng_needle -> g_eng. */
export const groupKey = (id: string) => {
  const p = id.split("_");
  return p[0] === "sig" || p[0] === "g" ? p.slice(0, 2).join("_") : p[0] === "ann" ? `ann_${p[1]}` : p[0] === "status" ? `ann_${p[1]}` : p[0];
};

/** Top tagged groups by their most important member; hazards outrank signals and warnings. */
export function topGroups(objs: SceneObject[], n = 3): Waypoint[] {
  const best = new Map<string, Waypoint>();
  for (const o of objs) {
    if (!o.position || !o.task_tags?.some((t) => t === "hazard" || t === "warning" || t === "signal")) continue;
    const key = groupKey(o.id), score = importanceOf(o) * (o.task_tags.includes("hazard") ? 2 : 1);
    if (score > (best.get(key)?.score ?? 0)) best.set(key, { key, score, position: o.position as [number, number, number] });
  }
  return [...best.values()].sort((a, b) => b.score - a.score || a.key.localeCompare(b.key)).slice(0, n);
}

export const FOLLOW_IN_S = 0.9, FOLLOW_HOLD_S = 1.8;
/** Camera weight toward an event, `age` seconds after it: ease in, hold, ease out. */
export function followWeight(age: number): number {
  if (age <= 0) return 0;
  if (age < FOLLOW_IN_S) return easeInOutCubic(age / FOLLOW_IN_S);
  const out = age - FOLLOW_IN_S - FOLLOW_HOLD_S;
  return out <= 0 ? 1 : out >= FOLLOW_IN_S ? 0 : 1 - easeInOutCubic(out / FOLLOW_IN_S);
}
