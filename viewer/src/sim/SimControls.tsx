// Bottom event timeline and the environment dial / effect toggles.
import { SPEEDS, formatTime, fractionOfStep, type Marker } from "./timeline";
import type { FxToggles } from "./fxPolicy";
import { Switch } from "../ui/Switch";

export interface PlayState { step: number; playing: boolean; speed: number; live: boolean }
export interface TimelineProps {
  state: PlayState; total: number; markers: Marker[] | null; follow: boolean;
  onSeek(step: number): void; onToggle(): void; onSpeed(s: number): void; onNext(): void; onLive(): void;
  onFollow(v: boolean): void;
}

export function Timeline(p: TimelineProps) {
  const { step, playing, speed, live } = p.state;
  return (
    <section className="sv-glass sv-timeline" aria-label="event timeline">
      <div className="sv-timeline-row">
        <button className="sv-pill" onClick={p.onToggle} title="space">{playing ? "Pause" : "Play"}</button>
        <button className="sv-pill" onClick={p.onNext} disabled={!p.markers?.length}>Next event</button>
        <button className="sv-pill" aria-pressed={live} onClick={p.onLive}>Live</button>
        <label className="sv-pill"><span>speed</span>
          <select aria-label="speed" value={speed} onChange={(e) => p.onSpeed(Number(e.target.value))}>
            {SPEEDS.map((s) => <option key={s} value={s}>{s}x</option>)}
          </select>
        </label>
        <Switch label="follow events" checked={p.follow} onChange={p.onFollow} />
        {!p.markers && <small className="sv-note">indexing events…</small>}
        <output className="sv-time">{formatTime(step)} / {formatTime(p.total)}</output>
      </div>
      <div className="sv-track">
        <input type="range" aria-label="scrub" min={0} max={p.total} step={1} value={step} onChange={(e) => p.onSeek(Number(e.target.value))} />
        {p.markers?.map((m) => (
          <button key={`${m.kind}${m.step}`} className="sv-marker" data-kind={m.kind} tabIndex={-1}
            style={{ left: `${100 * fractionOfStep(m.step, p.total)}%` }}
            aria-label={`${m.label} at ${formatTime(m.step)}`} title={`${m.label} ${formatTime(m.step)}: ${m.ids.slice(0, 4).join(", ")}`}
            onClick={() => p.onSeek(m.step)} />
        ))}
      </div>
    </section>
  );
}

export function EnvDial({ label, options, value, onChange }: { label: string; options: readonly string[]; value: string; onChange(v: string): void }) {
  const i = Math.max(0, options.indexOf(value));
  return (
    <div className="sv-dial" role="radiogroup" aria-label={label}>
      <span className="sv-dial-thumb" style={{ width: `calc((100% - 8px) / ${options.length})`, transform: `translateX(${100 * i}%)` }} />
      {options.map((o) => (
        <button key={o} role="radio" aria-checked={o === value} onClick={() => { if (o !== value) onChange(o); }}>{o}</button>
      ))}
    </div>
  );
}

export function FxBar({ fx, onChange, cinematicOn, blocked, tooltip, performance }: {
  fx: FxToggles; onChange(v: FxToggles): void; cinematicOn: boolean; blocked: boolean; tooltip: string; performance: boolean;
}) {
  const box = (k: "dust" | "waypoints", label: string, disabled = false) => (
    <Switch label={label} checked={fx[k] && !disabled} disabled={disabled} onChange={(v) => onChange({ ...fx, [k]: v })} />
  );
  return (
    <>
      {box("dust", "dust particles", performance)}
      {box("waypoints", "waypoints")}
      <Switch label="cinematic" title={tooltip} checked={cinematicOn} disabled={blocked}
        onChange={(v) => onChange({ ...fx, cinematic: v })} />
    </>
  );
}
