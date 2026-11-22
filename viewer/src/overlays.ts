// Canvas overlays drawn on top of the 3 panels (never baked into the scene).
import type { ConfusionPair, SceneObject } from "../../adapters/threejs";

export interface OverlayFlags {
  graph: boolean; heatmap: boolean; bboxes: boolean; labels: boolean;
}
export const DEFAULT_FLAGS: OverlayFlags = { graph: true, heatmap: false, bboxes: true, labels: true };

/** DeltaE00 -> colour: red (confusable) -> yellow -> green (clearly separable). */
export function deColor(de: number): string {
  const t = Math.min(1, Math.max(0, de / 20));
  return `hsl(${Math.round(t * 120)}, 90%, 50%)`;
}

/** Rule-based importance placeholder (Phase 8 replaces with learned): tag weight * sqrt(screen area). */
export function importanceOf(o: Pick<SceneObject, "task_tags" | "screen_area">): number {
  const tags = o.task_tags ?? [];
  const tw = tags.includes("hazard") ? 2 : tags.some((t) => t === "warning" || t === "signal") ? 1.5 : 1;
  return tw * Math.sqrt(Math.max(o.screen_area, 0));
}

/** Per-object worst (min) DeltaE00 over its confusion pairs: id -> [before, after]. */
export function perObjectDe(pairs: ConfusionPair[]): Map<string, [number, number]> {
  const m = new Map<string, [number, number]>();
  for (const p of pairs) for (const id of p.ids) {
    const c = m.get(id);
    m.set(id, c ? [Math.min(c[0], p.de00_before), Math.min(c[1], p.de00_after)] : [p.de00_before, p.de00_after]);
  }
  return m;
}

export type PanelKind = "A" | "B" | "C";

export function drawPanel(g: CanvasRenderingContext2D, x0: number, w: number, h: number, kind: PanelKind,
                          objs: SceneObject[], pairs: ConfusionPair[], f: OverlayFlags) {
  g.save();
  g.beginPath(); g.rect(x0, 0, w, h); g.clip();
  const box = (o: SceneObject) => [x0 + o.bbox.x * w, o.bbox.y * h, o.bbox.w * w, o.bbox.h * h] as const;
  const ctr = new Map(objs.map((o) => { const [x, y, bw, bh] = box(o); return [o.id, [x + bw / 2, y + bh / 2] as const]; }));
  if (f.heatmap) {
    const mx = Math.max(1e-6, ...objs.map(importanceOf));
    for (const o of objs) {
      const [x, y, bw, bh] = box(o), r = Math.max(bw, bh) * 1.2 + 10, a = importanceOf(o) / mx;
      const gr = g.createRadialGradient(x + bw / 2, y + bh / 2, 0, x + bw / 2, y + bh / 2, r);
      gr.addColorStop(0, `rgba(255,60,0,${0.55 * a})`); gr.addColorStop(1, "rgba(255,60,0,0)");
      g.fillStyle = gr; g.fillRect(x - r, y - r, bw + 2 * r, bh + 2 * r);
    }
  }
  if (f.bboxes) {
    g.strokeStyle = "rgba(0,255,255,.8)"; g.lineWidth = 1; g.font = "10px monospace"; g.fillStyle = "#0ff";
    for (const o of objs) { const [x, y, bw, bh] = box(o); g.strokeRect(x, y, bw, bh); g.fillText(o.id, x + 2, y + 10); }
  }
  if (f.graph && kind !== "A") {
    g.lineWidth = 2;
    for (const p of pairs) {
      const a = ctr.get(p.ids[0]), b = ctr.get(p.ids[1]);
      if (!a || !b) continue;
      const de = kind === "C" ? p.de00_after : p.de00_before;
      g.strokeStyle = deColor(de); g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.stroke();
      if (kind === "C" && !p.feasible) { g.fillStyle = "#f33"; g.fillText("✗", (a[0] + b[0]) / 2, (a[1] + b[1]) / 2); }
    }
  }
  if (f.labels && kind !== "A") {
    const pe = perObjectDe(pairs);
    g.font = "10px monospace";
    for (const o of objs) {
      const d = pe.get(o.id);
      if (!d) continue;
      const [x, y, bw, bh] = box(o);
      const s = `ΔE ${d[0].toFixed(1)}→${d[1].toFixed(1)}`;
      g.fillStyle = "rgba(0,0,0,.65)"; g.fillRect(x, y + bh - 12, g.measureText(s).width + 4, 12);
      g.fillStyle = deColor(kind === "C" ? d[1] : d[0]); g.fillText(s, x + 2, y + bh - 3);
    }
  }
  g.restore();
}

export function drawAll(cv: HTMLCanvasElement, pw: number, ph: number, objs: SceneObject[], pairs: ConfusionPair[], f: OverlayFlags) {
  const g = cv.getContext("2d");
  if (!g) return;
  g.clearRect(0, 0, cv.width, cv.height);
  (["A", "B", "C"] as PanelKind[]).forEach((k, i) => drawPanel(g, i * pw, pw, ph, k, objs, pairs, f));
}
