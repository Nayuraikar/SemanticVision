// View-mode pill, per-pane plain-language labels, wipe divider and the single-view "Feel the difference" control.
import type { PointerEvent } from "react";
import { PANES, PANE_INFO, clamp, isVisible, layoutTiles, type Pane, type ViewMode, type ViewState } from "../view/layout";

export const MODES: [ViewMode, string][] = [["split", "Split"], ["wipe", "Wipe"], ["single", "Single"]];
const WIPE_STEP = 0.02;

export function ModeBar({ mode, onMode }: { mode: ViewMode; onMode(m: ViewMode): void }) {
  const i = MODES.findIndex(([m]) => m === mode);
  return (
    <div className="sv-dial sv-modes" role="radiogroup" aria-label="view mode">
      <span className="sv-dial-thumb" style={{ width: "calc((100% - 8px) / 3)", transform: `translateX(${100 * i}%)` }} />
      {MODES.map(([m, label], k) => (
        <button key={m} role="radio" aria-checked={m === mode} aria-keyshortcuts={String(k + 1)} title={`${label} (${k + 1})`}
          onClick={() => onMode(m)}>{label}</button>
      ))}
    </div>
  );
}

function Chip({ pane, left, width }: { pane: Pane; left: number; width: number }) {
  return (
    <div className="sv-chip" style={{ left: `${left}%`, width: `${width}%` }}>
      <div><b>{`${pane} · ${PANE_INFO[pane].name}`}</b><small>{PANE_INFO[pane].hint}</small></div>
    </div>
  );
}

/** Labels follow the target layout, expressed in percentages so they need no measured size. */
export function PaneLabels({ view }: { view: ViewState }) {
  if (view.mode === "single") {
    const pane: Pane = view.source === "original" ? "A" : "C";
    const name = view.feel > 0.5 ? (pane === "A" ? "How you see it" : "Corrected, as you see it") : pane === "A" ? "Original" : "Corrected colours";
    return (
      <div className="sv-chips"><div className="sv-chip" style={{ left: "0%", width: "100%" }}>
        <div><b>{name}</b><small>{view.feel > 0.5 ? "through your colour-vision model" : "without the colour-vision simulation"}</small></div>
      </div></div>
    );
  }
  const tiles = layoutTiles(view, 1000, 1).filter(isVisible);
  return (
    <div className="sv-chips">
      {view.mode === "wipe"
        ? [view.left, view.right].map((p, i) => <Chip key={`${p}${i}`} pane={p} left={i * 50} width={50} />)
        : tiles.map((t) => <Chip key={t.pane} pane={t.pane} left={t.clip.x / 10} width={t.clip.w / 10} />)}
    </div>
  );
}

export function WipeDivider({ value, onChange }: { value: number; onChange(v: number): void }) {
  const move = (e: PointerEvent<HTMLDivElement>) => {
    const r = e.currentTarget.parentElement!.getBoundingClientRect();
    if (r.width) onChange(clamp((e.clientX - r.left) / r.width, 0, 1));
  };
  return (
    <div className="sv-divider" role="slider" tabIndex={0} aria-label="comparison divider" aria-valuemin={0} aria-valuemax={100}
      aria-valuenow={Math.round(value * 100)} style={{ left: `${value * 100}%` }}
      onPointerDown={(e) => e.currentTarget.setPointerCapture?.(e.pointerId)}
      onPointerMove={(e) => { if (e.buttons & 1) move(e); }}
      onKeyDown={(e) => {
        const d = e.key === "ArrowLeft" ? -WIPE_STEP : e.key === "ArrowRight" ? WIPE_STEP : 0;
        if (!d) return;
        e.preventDefault(); e.stopPropagation(); // the arrows move the divider here, not the timeline
        onChange(clamp(value + d, 0, 1));
      }}>
      <span aria-hidden="true">◀ ▶</span>
    </div>
  );
}

/** Bar for the mode-specific choices (nothing in split). */
export function ModeOptions({ view, onChange, onFeel, feeling }: {
  view: ViewState; onChange(p: Partial<ViewState>): void; onFeel(): void; feeling: boolean;
}) {
  if (view.mode === "split") return null;
  const pick = (side: "left" | "right", label: string) => (
    <label className="sv-pill sv-solidpill"><span>{label}</span>
      <select aria-label={label} value={view[side]} onChange={(e) => {
        const p = e.target.value as Pane, other = side === "left" ? "right" : "left";
        // choosing the other side's pane swaps them: the two sides always differ
        onChange({ [side]: p, ...(view[other] === p ? { [other]: view[side] } : {}) });
      }}>
        {PANES.map((p) => <option key={p} value={p}>{PANE_INFO[p].name}</option>)}
      </select>
    </label>
  );
  return (
    <div className="sv-modeopts">
      {view.mode === "wipe" ? <>{pick("left", "left side")}{pick("right", "right side")}</> : (
        <>
          <button className="sv-btn sv-btn-primary" aria-pressed={feeling} onClick={onFeel}>Feel the difference</button>
          <label className="sv-pill sv-solidpill"><span>colours</span>
            <select aria-label="colours" value={view.source} onChange={(e) => onChange({ source: e.target.value as ViewState["source"] })}>
              <option value="original">Original</option><option value="corrected">Corrected</option>
            </select>
          </label>
        </>
      )}
    </div>
  );
}
