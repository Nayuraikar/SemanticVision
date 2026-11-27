import { useEffect, useRef } from "react";
import { PANES, PANE_INFO } from "../view/layout";

export const CUES: [string, string][] = [
  ["outline", "A solid rim around the object."],
  ["hatch", "A wireframe pattern drawn over the surface."],
  ["icon", "A symbol floating above the object."],
  ["label", "A short text tag above the object."],
];
export const SHORTCUTS: [string, string][] = [
  ["1 / 2 / 3", "split, wipe, single view"], ["Space", "play or pause"], ["← / →", "step back or forward"],
  ["I", "inspector"], ["G", "confusion graph"], ["D", "diagnostics"], ["?", "this help"], ["Esc", "close panels"],
  ["0", "reset the camera"], ["Drag / scroll", "orbit and zoom"], ["Click", "inspect and fly to an object"],
];

function Glyph({ type }: { type: string }) {
  return (
    <svg width="34" height="34" viewBox="0 0 34 34" aria-hidden="true" fill="none" stroke="currentColor">
      <rect x="7" y="12" width="20" height="18" rx="2" strokeWidth={type === "outline" ? 4 : 1} />
      {type === "hatch" && <path d="M7 20 15 12M7 28 23 12M13 30 27 16M21 30 27 24" />}
      {type === "icon" && <path d="M17 1 23 10H11Z" fill="currentColor" />}
      {type === "label" && <><rect x="5" y="1" width="24" height="9" rx="2" /><path d="M9 5.5h16" /></>}
    </svg>
  );
}

export function Help({ open, onClose }: { open: boolean; onClose(): void }) {
  const close = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (!open) return;
    const before = document.activeElement as HTMLElement | null;
    close.current?.focus();
    return () => before?.focus?.();
  }, [open]);
  if (!open) return null;
  return (
    <div className="sv-help" onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="sv-glass" role="dialog" aria-modal="true" aria-label="help">
        <header><h2>Help</h2><button ref={close} className="sv-pill" onClick={onClose}>Close</button></header>
        <h4>Views</h4>
        <dl className="sv-facts">{PANES.map((p) => [<dt key={`${p}t`}>{p}</dt>, <dd key={`${p}d`}>{`${PANE_INFO[p].name}: ${PANE_INFO[p].hint}`}</dd>])}</dl>
        <h4>Non-colour cues (corrected view)</h4>
        <ul className="sv-cues">{CUES.map(([t, d]) => <li key={t}><Glyph type={t} /><span><b>{t}</b> {d}</span></li>)}</ul>
        <h4>Shortcuts</h4>
        <dl className="sv-facts">{SHORTCUTS.map(([k, d]) => [<dt key={`${k}t`}><kbd>{k}</kbd></dt>, <dd key={`${k}d`}>{d}</dd>])}</dl>
      </div>
    </div>
  );
}
