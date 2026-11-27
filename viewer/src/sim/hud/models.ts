// Pure data for the inspector and diagnostics drawers: only values the API actually returns.
import { irradiance, litColor } from "../../../../adapters/threejs/lighting";
import type { Overrides, SceneObject, SHLighting, Snapshot } from "../../../../adapters/threejs/protocol.gen";
import { API, type Stat, type Summary } from "../../calibLib";
import { PROFILE_PREFIX } from "../../appState";
import type { PresetObs, SavedProfile } from "../../screens";
import { partnersOf } from "../view/geometry";

export type Rgb = [number, number, number];

export interface Inspection {
  id: string; cls: string; state: string;
  litBefore: Rgb | null; litAfter: Rgb | null;
  partners: ReturnType<typeof partnersOf>;
  worst: { before: number; after: number } | null;
  tier: string; cues: string[];
}

/** Lit colour after correction: the exporter's own lighting formula applied to the override albedo/emission. */
export function litAfter(o: SceneObject, snap: Snapshot, ov: Overrides): Rgb | null {
  const L = snap.lighting as SHLighting;
  if (!L.coeffs) return null;
  const p = ov.palette_overrides?.find((x) => x.id === o.id);
  if (!p) return (o.lit_color as Rgb) ?? null;
  const E = irradiance(L, (o.normal as Rgb) ?? null, (o.position as Rgb) ?? null, o.shadow_vis ?? null);
  return litColor(p.albedo, (p.emissive ?? o.emissive_rgb ?? [0, 0, 0]) as Rgb, E, L.exposure ?? 1);
}

export function inspect(id: string, snap: Snapshot | null, ov: Overrides | null, tier: string | null): Inspection | null {
  const o = snap?.objects?.find((x) => x.id === id);
  if (!o) return null;
  const partners = partnersOf(ov?.confusion_pairs ?? [], id);
  return {
    id, cls: o.class, state: o.state ?? "default",
    litBefore: (o.lit_color as Rgb) ?? null,
    litAfter: ov ? litAfter(o, snap!, ov) : null,
    partners,
    worst: partners.length
      ? { before: Math.min(...partners.map((p) => p.before)), after: Math.min(...partners.map((p) => p.after)) } : null,
    tier: tier ?? "not reported",
    cues: (ov?.cues ?? []).filter((c) => c.id === id).map((c) => c.type),
  };
}

export interface PosteriorRow { label: string; mean: number; lo: number | null; hi: number | null; max: number }
export interface Posterior {
  type: string; model: string | null; rows: PosteriorRow[];
  probs: [string, number][]; trials: number | null; width: number | null;
  /** plain statements about what the API did not provide */
  notes: string[];
}

const PARAMS: [string, string, number][] = [
  ["e_protan", "protan severity", 1], ["e_deutan", "deutan severity", 1], ["e_tritan", "tritan severity", 1],
  ["k", "threshold scale k", 2], ["s_L", "lightness sensitivity s_L", 2], ["s_C", "chroma sensitivity s_C", 2],
];

export type PresetFull = PresetObs & { k?: number; s_L?: number; s_C?: number };

export function posteriorOf(
  obsId: string, presets: PresetFull[], profiles: SavedProfile[], summary: Partial<Summary> | null,
): Posterior | null {
  const point = (label: string, v: number | undefined, max: number): PosteriorRow[] =>
    v == null ? [] : [{ label, mean: v, lo: null, hi: null, max: Math.max(max, v) }];
  const preset = presets.find((p) => p.id === obsId);
  if (preset) {
    const rows = [...point("severity", preset.severity, 1), ...point("threshold scale k", preset.k, 2),
      ...point("lightness sensitivity s_L", preset.s_L, 2), ...point("chroma sensitivity s_C", preset.s_C, 2)];
    const notes = ["Preset observer: fixed values, so there are no uncertainty ranges."];
    if (preset.k == null) notes.push("Threshold values are not reported for this observer.");
    return { type: preset.kind, model: preset.model, rows, probs: [], trials: null, width: null, notes };
  }
  const prof = profiles.find((p) => p.observer_id === obsId);
  if (!prof) return null;
  const notes: string[] = [];
  const rows: PosteriorRow[] = [];
  const missing: string[] = [];
  for (const [key, label, max] of PARAMS) {
    const s: Stat | undefined = summary?.params?.[key];
    if (s?.mean == null) { missing.push(label); continue; }
    rows.push({ label, mean: s.mean, lo: s.lo ?? null, hi: s.hi ?? null, max: Math.max(max, s.hi ?? s.mean) });
  }
  if (!rows.length) {
    rows.push(...point("severity", prof.severity, 1));
    notes.push("Profile detail is unavailable: only the point type and severity are shown, without uncertainty.");
  } else {
    notes.push("Bars show the posterior mean and 90% interval.");
    if (missing.length) notes.push(`Not reported: ${missing.join(", ")}.`);
  }
  const probs = Object.entries(summary?.type_probs ?? {}).sort((a, b) => b[1] - a[1]);
  if (!probs.length) notes.push("Type probabilities are not reported.");
  return { type: prof.kind, model: prof.model, rows, probs, trials: prof.n_trials ?? null, width: summary?.width ?? null, notes };
}

export async function fetchProfileSummary(observerId: string): Promise<Partial<Summary> | null> {
  const r = await fetch(`${API}/profiles/${encodeURIComponent(observerId.slice(PROFILE_PREFIX.length))}`);
  if (!r.ok) throw new Error(`GET /profiles/{id} ${r.status}`);
  return (await r.json()).summary ?? null;
}
