// Pure view-mode geometry (CSS px, origin top-left). One shared camera: every tile has the same view aspect.
export type ViewMode = "split" | "wipe" | "single";
export type Pane = "A" | "B" | "C";
export const PANES: Pane[] = ["A", "B", "C"];
export const TILE_GAP = 6;

export const PANE_INFO: Record<Pane, { name: string; hint: string }> = {
  A: { name: "Original", hint: "the scene as it was made" },
  B: { name: "How you see it", hint: "the original through your colour-vision model" },
  C: { name: "Corrected, as you see it", hint: "corrected colours through the same model" },
};

export interface Rect { x: number; y: number; w: number; h: number }
/** view: where the full camera image maps to; clip: the visible part; sim: CVD amount 0..1 */
export interface Tile { pane: Pane; view: Rect; clip: Rect; sim: number }

export interface ViewState {
  mode: ViewMode;
  /** wipe: the two panes compared and the divider position 0..1 */
  left: Pane; right: Pane; divider: number;
  /** single: which colours are shown, and the animated CVD amount */
  source: "original" | "corrected"; feel: number;
}
export const DEFAULT_VIEW: ViewState = { mode: "split", left: "B", right: "C", divider: 0.5, source: "original", feel: 0 };

export const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));
const simOf = (p: Pane) => (p === "A" ? 0 : 1);

export function layoutTiles(s: ViewState, w: number, h: number): Tile[] {
  const full: Rect = { x: 0, y: 0, w, h };
  if (s.mode === "split") {
    const tw = (w - 2 * TILE_GAP) / 3;
    return PANES.map((pane, i) => {
      const r = { x: i * (tw + TILE_GAP), y: 0, w: tw, h };
      return { pane, view: r, clip: { ...r }, sim: simOf(pane) };
    });
  }
  if (s.mode === "wipe") {
    const d = clamp(s.divider, 0, 1) * w;
    return PANES.map((pane) => {
      const clip = pane === s.left ? { x: 0, y: 0, w: d, h }
        : pane === s.right ? { x: d, y: 0, w: w - d, h }
          : { x: pane === "A" ? 0 : w, y: 0, w: 0, h };
      return { pane, view: { ...full }, clip, sim: simOf(pane) };
    });
  }
  const shown: Pane = s.source === "original" ? "A" : "C";
  return PANES.map((pane) => ({
    pane, view: { ...full },
    clip: pane === shown ? { ...full } : { x: pane === "A" ? 0 : w, y: 0, w: 0, h },
    sim: pane === "B" ? 1 : clamp(s.feel, 0, 1),
  }));
}

const mix = (a: number, b: number, k: number) => a + (b - a) * k;
const mixRect = (a: Rect, b: Rect, k: number): Rect =>
  ({ x: mix(a.x, b.x, k), y: mix(a.y, b.y, k), w: mix(a.w, b.w, k), h: mix(a.h, b.h, k) });

export const lerpTiles = (from: Tile[], to: Tile[], k: number): Tile[] =>
  to.map((t, i) => ({ pane: t.pane, view: mixRect(from[i].view, t.view, k), clip: mixRect(from[i].clip, t.clip, k), sim: mix(from[i].sim, t.sim, k) }));

export const isVisible = (t: Tile) => t.clip.w >= 1 && t.clip.h >= 1;

/** Topmost visible tile under a point (later tiles draw on top). */
export function tileAt(tiles: Tile[], x: number, y: number): Tile | null {
  for (let i = tiles.length - 1; i >= 0; i--) {
    const c = tiles[i].clip;
    if (isVisible(tiles[i]) && x >= c.x && x < c.x + c.w && y >= c.y && y < c.y + c.h) return tiles[i];
  }
  return null;
}

export const ndcIn = (view: Rect, x: number, y: number): [number, number] =>
  [((x - view.x) / view.w) * 2 - 1, 1 - ((y - view.y) / view.h) * 2];

/** Narrowest aspect the authored framing is kept for; narrower tiles widen the vertical field of view instead. */
export const SAFE_ASPECT = 1.45;
export const MAX_FOV = 90;
export function fitFov(baseFov: number, aspect: number, safe = SAFE_ASPECT): number {
  if (!(aspect > 0) || aspect >= safe) return baseFov;
  const t = Math.tan((baseFov * Math.PI) / 360) * (safe / aspect);
  return Math.min(MAX_FOV, (Math.atan(t) * 360) / Math.PI);
}
