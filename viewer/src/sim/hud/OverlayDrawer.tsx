// Collapsible "Overlays" tray: every measurement overlay is opt-in.
import type { OverlayFlags } from "../../overlays";
import { Switch } from "../../ui/Switch";

const OVERLAYS: [keyof OverlayFlags, string][] = [
  ["bboxes", "Bounding boxes"], ["labels", "ΔE00 labels"], ["graph", "Confusion graph"], ["heatmap", "Importance heatmap"],
];
export const TOP_N_MAX = 30;

export function OverlayDrawer({ open, onOpen, flags, onChange, topN, onTopN }: {
  open: boolean; onOpen(v: boolean): void; flags: OverlayFlags; onChange(f: OverlayFlags): void; topN: number; onTopN(n: number): void;
}) {
  const count = OVERLAYS.filter(([k]) => flags[k]).length;
  return (
    <section className="sv-overlays">
      <button className="sv-pill sv-solidpill" aria-expanded={open} aria-controls="sv-overlay-panel" onClick={() => onOpen(!open)}>
        {`Overlays${count ? ` (${count})` : ""}`}
      </button>
      {open && (
        <div id="sv-overlay-panel" className="sv-glass sv-overlay-panel" role="group" aria-label="overlays">
          {OVERLAYS.map(([k, label]) => (
            <Switch key={k} label={label} checked={flags[k]} onChange={(v) => onChange({ ...flags, [k]: v })} />
          ))}
          {flags.labels && <p className="sv-note">Labels appear for the hovered or selected object and its partners.</p>}
          {flags.graph && (
            <>
              <label className="sv-range">{`Show worst ${topN} confusable pairs`}
                <input type="range" aria-label="graph pairs" min={1} max={TOP_N_MAX} value={topN} onChange={(e) => onTopN(Number(e.target.value))} />
              </label>
              <p className="sv-note">Solid: confusable in the original (view B). Dashed: still below tau after correction (view C).</p>
            </>
          )}
        </div>
      )}
    </section>
  );
}
