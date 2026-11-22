// Calibration page helpers (pure; unit-tested). Stimuli arrive as LINEAR sRGB and are encoded only for display.
export const API = "http://localhost:8000";
export const PX_PER_DEG = 40; // nominal; real display geometry is not measured

export type Stimulus = {
  trial_id: number; mode: "same_different" | "slider"; c1: number[]; c2: number[];
  size_deg: number; bg_lum: number; axis: string; swap: boolean;
};
export type Progress = { n: number; max_trials: number; min_trials: number; width: number; target_width: number; done: boolean };
export type Stat = { mean: number; lo: number; hi: number };
export type MapCell = { axis: string; Y: number; delta50: number | null; lo: number | null; hi: number | null; resolvable: number };
export type Summary = {
  n_trials: number; width: number; dominant: string; effective_severity: number[];
  params: Record<string, Stat>; type_probs: Record<string, number>; discrimination_map?: MapCell[];
};

export function linearToSrgb(c: number): number {
  const a = Math.min(1, Math.max(0, c));
  return a <= 0.0031308 ? 12.92 * a : 1.055 * Math.pow(a, 1 / 2.4) - 0.055;
}

export function cssColor(lin: number[]): string {
  const [r, g, b] = lin.map((x) => Math.round(255 * linearToSrgb(x)));
  return `rgb(${r}, ${g}, ${b})`;
}

export const bgCss = (lum: number) => cssColor([lum, lum, lum]);
export const patchPx = (deg: number) => Math.round(deg * PX_PER_DEG);
export const sliderLabel = (v: number) => (v < 0.125 ? "identical" : v < 0.375 ? "barely different" : v < 0.625 ? "different" : v < 0.875 ? "clearly different" : "very different");

export const TYPE_LABEL: Record<string, string> = {
  protan: "reduced long-wavelength (red) sensitivity",
  deutan: "reduced medium-wavelength (green) sensitivity",
  tritan: "reduced short-wavelength (blue) sensitivity",
};
export const AXIS_LABEL: Record<string, string> = {
  protan: "red-axis (L cone)", deutan: "green-axis (M cone)", tritan: "blue-axis (S cone)", lightness: "lightness",
};

export const fmt = (x: number | null | undefined, d = 2) => (x == null ? "n/a" : x.toFixed(d));
export const range = (s: Stat, d = 2) => `${fmt(s.mean, d)} (${fmt(s.lo, d)}–${fmt(s.hi, d)})`;

/** Discrimination map -> rows by axis, columns by lightness level (sorted). */
export function mapGrid(cells: MapCell[]): { levels: number[]; rows: { axis: string; cells: MapCell[] }[] } {
  const levels = [...new Set(cells.map((c) => c.Y))].sort((a, b) => a - b);
  const axes = [...new Set(cells.map((c) => c.axis))];
  return { levels, rows: axes.map((axis) => ({ axis, cells: levels.map((y) => cells.find((c) => c.axis === axis && c.Y === y)!) })) };
}

export function cellText(c: MapCell): string {
  return c.delta50 == null ? "not resolvable" : `${fmt(c.delta50 * 100, 0)}% (${fmt(c.lo == null ? null : c.lo * 100, 0)}–${c.hi == null ? "∞" : fmt(c.hi * 100, 0)})`;
}

/** Plain-language uncertainty from the stop-rule width (normalised 90% interval width). */
export const certainty = (w: number) => (w <= 0.25 ? "high" : w <= 0.4 ? "moderate" : "low");
