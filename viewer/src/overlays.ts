// Canvas overlays over the 3D view (never baked into the scene). Everything is off until the viewer turns it on;
// the hover/selection glow and partner rings are the only things drawn by default.
import type * as THREE from "three";
import { svMeshes, type ConfusionPair, type SceneObject } from "../../adapters/threejs";
import { isVisible, type Tile } from "./sim/view/layout";
import {
  meshBox, meshHull, partnersOf, placeLabels, topPairs, type LabelIn, type Pt,
} from "./sim/view/geometry";

export interface OverlayFlags {
  graph: boolean; heatmap: boolean; bboxes: boolean; labels: boolean;
}
export const DEFAULT_FLAGS: OverlayFlags = { graph: false, heatmap: false, bboxes: false, labels: false };

/** DeltaE00 -> colour: red (confusable) -> yellow -> green (clearly separable). Always paired with text or dashes. */
export function deColor(de: number): string {
  const t = Math.min(1, Math.max(0, de / 20));
  return `hsl(${Math.round(t * 120)}, 90%, 60%)`;
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

export interface OverlayFrame {
  scene: THREE.Object3D; camera: THREE.PerspectiveCamera; tiles: Tile[];
  objects: SceneObject[]; pairs: ConfusionPair[]; flags: OverlayFlags; topN: number;
  hover: string | null; selected: string | null; t: number;
}

const WARM = "#ffd08a";

function pathOf(g: CanvasRenderingContext2D, pts: Pt[]) {
  g.beginPath();
  pts.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y)));
  g.closePath();
}

/** White-on-dark double stroke: readable on any colour, for any colour vision. */
function glow(g: CanvasRenderingContext2D, pts: Pt[], strong: boolean) {
  if (pts.length < 3) return;
  g.lineJoin = "round";
  pathOf(g, pts);
  g.strokeStyle = "rgba(0,0,0,.65)"; g.lineWidth = strong ? 6 : 5; g.stroke();
  g.shadowColor = WARM; g.shadowBlur = 16;
  g.strokeStyle = strong ? "#fff" : WARM; g.lineWidth = strong ? 2.5 : 1.8; g.stroke();
  g.shadowBlur = 0;
}

/** Dashed double ring around a confusing partner. */
function ring(g: CanvasRenderingContext2D, x: number, y: number, r: number, t: number) {
  g.setLineDash([6, 5]); g.lineDashOffset = -t * 14;
  g.shadowColor = WARM; g.shadowBlur = 12;
  for (const [rr, colour, width] of [[r, "rgba(0,0,0,.7)", 5], [r, WARM, 2.5], [r + 6, "#fff", 1]] as const) {
    g.beginPath(); g.arc(x, y, rr, 0, 2 * Math.PI); g.strokeStyle = colour; g.lineWidth = width; g.stroke();
  }
  g.setLineDash([]); g.shadowBlur = 0;
}

export function drawOverlays(g: CanvasRenderingContext2D, f: OverlayFrame) {
  const meshes = new Map(svMeshes(f.scene).map((m) => [m.userData.sv.id as string, m]));
  const partners = f.selected ? partnersOf(f.pairs, f.selected).map((p) => p.id) : [];
  const de = perObjectDe(f.pairs);
  const mx = Math.max(1e-6, ...f.objects.map(importanceOf));
  const anyBox = f.flags.bboxes || f.flags.heatmap || f.flags.graph;
  for (const tile of f.tiles) {
    if (!isVisible(tile)) continue;
    const { clip, view, pane } = tile;
    g.save();
    g.beginPath(); g.rect(clip.x, clip.y, clip.w, clip.h); g.clip();
    const boxes = new Map<string, ReturnType<typeof meshBox>>();
    const boxOf = (id: string) => {
      if (!boxes.has(id)) { const m = meshes.get(id); boxes.set(id, m ? meshBox(m, f.camera, view) : null); }
      return boxes.get(id)!;
    };
    const centre = (id: string): Pt | null => { const b = boxOf(id); return b ? [b.x + b.w / 2, b.y + b.h / 2] : null; };
    if (anyBox && f.flags.heatmap) {
      for (const o of f.objects) {
        const b = boxOf(o.id);
        if (!b) continue;
        const x = b.x + b.w / 2, y = b.y + b.h / 2, r = Math.max(b.w, b.h) * 0.8 + 10, a = importanceOf(o) / mx;
        const gr = g.createRadialGradient(x, y, 0, x, y, r);
        gr.addColorStop(0, `rgba(255,120,0,${0.5 * a})`); gr.addColorStop(1, "rgba(255,120,0,0)");
        g.fillStyle = gr; g.fillRect(x - r, y - r, 2 * r, 2 * r);
      }
    }
    if (f.flags.bboxes) {
      g.strokeStyle = "rgba(255,255,255,.5)"; g.lineWidth = 1;
      for (const o of f.objects) { const b = boxOf(o.id); if (b) g.strokeRect(b.x, b.y, b.w, b.h); }
    }
    if (f.flags.graph) {
      topPairs(f.pairs, pane, f.topN).forEach((p, i) => {
        const a = centre(p.ids[0]), b = centre(p.ids[1]);
        if (!a || !b) return;
        const bend = (i % 2 ? -1 : 1) * 0.18;
        g.beginPath(); g.moveTo(a[0], a[1]);
        g.quadraticCurveTo((a[0] + b[0]) / 2 - (b[1] - a[1]) * bend, (a[1] + b[1]) / 2 + (b[0] - a[0]) * bend, b[0], b[1]);
        g.setLineDash(pane === "C" ? [5, 4] : []); // dashed: still confusable after correction
        g.strokeStyle = deColor(pane === "C" ? p.de00_after : p.de00_before); g.lineWidth = 1.25; g.stroke();
        g.setLineDash([]);
      });
    }
    for (const id of partners) {
      const b = boxOf(id);
      if (b) ring(g, b.x + b.w / 2, b.y + b.h / 2, Math.min(60, Math.max(b.w, b.h) / 2 + 8), f.t);
    }
    for (const id of new Set([f.selected, f.hover])) {
      const m = id ? meshes.get(id) : undefined;
      if (m) glow(g, meshHull(m, f.camera, view), id === f.selected);
    }
    if (f.flags.labels) {
      g.font = "12px ui-monospace, Menlo, monospace";
      const text = new Map<string, string>();
      const wanted: LabelIn[] = [];
      for (const id of new Set([f.hover, f.selected, ...partners])) {
        const at = id ? centre(id) : null;
        if (!id || !at) continue;
        const d = de.get(id);
        const s = d ? `${id}  ΔE00 ${d[0].toFixed(1)} → ${d[1].toFixed(1)}` : id;
        text.set(id, s);
        wanted.push({ id, anchor: at, w: g.measureText(s).width + 12, h: 20 });
      }
      for (const l of placeLabels(wanted, clip)) {
        g.fillStyle = "rgba(10,10,16,.85)"; g.fillRect(l.x, l.y, l.w, l.h);
        g.fillStyle = "#fff"; g.fillText(text.get(l.id)!, l.x + 6, l.y + 14);
      }
    }
    g.restore();
  }
}
