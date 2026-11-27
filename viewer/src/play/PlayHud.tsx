// Play-mode HUD: Watch|Play pill, mission HUD (replaces the timeline), scene controls, context card, minimap, pause.
import { useEffect, useRef, type MouseEvent } from "react";
import { GAUGE_SYSTEMS, DASH_LIGHTINGS, type DashLighting } from "../../../adapters/threejs/dashboardSim";
import { PALETTE_NAMES, type Action, type CardAction } from "../../../adapters/threejs/play/actions";
import type { Ground } from "../../../adapters/threejs/play/movement";
import type { Facts } from "../../../adapters/threejs/play/worlds";
import { formatTime } from "../sim/timeline";
import { Switch } from "../ui/Switch";
import {
  LOG_LABEL, logDocument, missionsFor, summarize, type Answer, type Mission, type MissionRecord,
} from "./missions";

export const SCRUB_WHY = "Scrubbing is off in Play: the run depends on your inputs, so there is no future to seek to and rewinding would undo your actions. Switch to Watch to scrub.";

export function PlayPill({ play, onChange }: { play: boolean; onChange(v: boolean): void }) {
  return (
    <div className="sv-dial sv-playpill" role="radiogroup" aria-label="mode">
      <span className="sv-dial-thumb" style={{ width: "calc((100% - 8px) / 2)", transform: `translateX(${play ? 100 : 0}%)` }} />
      {[false, true].map((v) => (
        <button key={String(v)} role="radio" aria-checked={v === play} onClick={() => { if (v !== play) onChange(v); }}>{v ? "Play" : "Watch"}</button>
      ))}
    </div>
  );
}

export interface MissionHudProps {
  scene: string; step: number; beacons: Ground["beacons"];
  active: Mission | null; seconds: number; result: MissionRecord | null; records: MissionRecord[];
  muted: boolean; audioOff: boolean;
  onStart(m: Mission, c: "B" | "C"): void; onAnswer(a: Answer): void; onCancel(): void; onCloseResult(): void;
  onFly(id: string): void; onMute(m: boolean): void; onDownload(): void; onPause(): void;
}

export function MissionHud(p: MissionHudProps) {
  const sum = summarize(p.records);
  const row = (c: "B" | "C") => `${c}: ${sum[c].correct}/${sum[c].n} correct${sum[c].meanSeconds === null ? "" : `, mean ${sum[c].meanSeconds!.toFixed(1)} s`}`;
  return (
    <section className="sv-glass sv-timeline sv-missionhud" aria-label="mission HUD">
      <div className="sv-timeline-row">
        <b>Live</b>
        <span title={SCRUB_WHY}><input type="range" aria-label="scrub" disabled value={0} readOnly /></span>
        <button className="sv-pill" onClick={p.onPause} aria-keyshortcuts="Escape">Pause menu</button>
        <Switch label="alarm beep" checked={!p.muted && !p.audioOff} disabled={p.audioOff} onChange={(v) => p.onMute(!v)} />
        {p.beacons.map((b) => <button key={b.id} className="sv-pill" onClick={() => p.onFly(b.id)} title="fly to waypoint">{b.label}</button>)}
        <output className="sv-time">{formatTime(p.step)}</output>
      </div>
      {p.result ? (
        <div className="sv-mission" role="status">
          <b>{p.result.correct ? "✓ Correct" : "✗ Not correct"}</b>
          <span>{`${p.result.seconds.toFixed(1)} s, view ${p.result.condition}`}</span>
          <button className="sv-pill" onClick={p.onCloseResult}>Next mission</button>
        </div>
      ) : p.active ? (
        <div className="sv-mission" role="group" aria-label="active mission">
          <b>{p.active.title}</b>
          <span>{p.active.task}</span>
          <small>{p.active.hint}</small>
          <output>{`${p.seconds.toFixed(1)} s`}</output>
          {p.active.choices?.map((c) => <button key={c} className="sv-btn" onClick={() => p.onAnswer({ type: "choice", value: c })}>{c}</button>)}
          {p.active.done && <button className="sv-btn sv-btn-primary" onClick={() => p.onAnswer({ type: "choice", value: "done" })}>Done</button>}
          <button className="sv-pill" onClick={p.onCancel}>Give up</button>
        </div>
      ) : (
        <div className="sv-mission" role="group" aria-label="missions">
          {missionsFor(p.scene).map((m) => (
            <span key={m.id} className="sv-missioncard">
              <b>{m.title}</b>
              <button className="sv-pill" aria-label={`${m.title}: start in view B`} title="B: how you see it" onClick={() => p.onStart(m, "B")}>Start in B</button>
              <button className="sv-pill" aria-label={`${m.title}: start in view C`} title="C: corrected, as you see it" onClick={() => p.onStart(m, "C")}>Start in C</button>
            </span>
          ))}
        </div>
      )}
      <div className="sv-timeline-row">
        <small className="sv-note">{`${LOG_LABEL} · ${row("B")} · ${row("C")}`}</small>
        <button className="sv-pill" onClick={p.onDownload} disabled={!p.records.length}>Download log</button>
      </div>
    </section>
  );
}

export function downloadLog(records: MissionRecord[]) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(logDocument(records), null, 2)], { type: "application/json" }));
  const a = document.createElement("a");
  a.href = url; a.download = "semanticvision-demo-task-log.json"; a.click();
  URL.revokeObjectURL(url);
}

export function SceneControls({ scene, facts, lighting, onAction, onLighting }: {
  scene: string; facts: Facts | null; lighting: DashLighting; onAction(a: Action): void; onLighting(v: DashLighting): void;
}) {
  if (scene === "traffic") {
    return (
      <div className="sv-glass sv-fxtray" role="group" aria-label="scene actions">
        <button className="sv-btn sv-btn-primary" onClick={() => onAction({ type: "collision" })}>Trigger collision</button>
        <button className="sv-pill" onClick={() => onAction({ type: "signal", axis: "ew" })}>Cycle east-west signal</button>
        <button className="sv-pill" onClick={() => onAction({ type: "signal", axis: "ns" })}>Cycle north-south signal</button>
      </div>
    );
  }
  if (scene === "dashboard") {
    return (
      <div className="sv-glass sv-fxtray sv-gaugetray" role="group" aria-label="scene actions">
        <label className="sv-pill"><span>cabin light</span>
          <input type="range" aria-label="cabin light lever" min={0} max={DASH_LIGHTINGS.length - 1} step={1}
            value={DASH_LIGHTINGS.indexOf(lighting)} aria-valuetext={lighting}
            onChange={(e) => onLighting(DASH_LIGHTINGS[Number(e.target.value)])} />
          <span>{lighting}</span>
        </label>
        {GAUGE_SYSTEMS.map((s) => (
          <label key={s} className="sv-pill"><span>{s}</span>
            <input type="range" aria-label={`${s} gauge`} min={0} max={1} step={0.01} value={facts?.gauges?.[s] ?? 0.5}
              onChange={(e) => onAction({ type: "gauge", system: s, value: Number(e.target.value) })} />
          </label>
        ))}
        <button className="sv-pill" onClick={() => GAUGE_SYSTEMS.forEach((s) => onAction({ type: "gauge", system: s, value: null }))}>Release gauges</button>
      </div>
    );
  }
  return (
    <div className="sv-glass sv-fxtray" role="group" aria-label="scene actions">
      <button className="sv-pill" onClick={() => onAction({ type: "lamp" })}>Toggle signal lamp</button>
      {PALETTE_NAMES.map((n) => <button key={n} className="sv-pill" onClick={() => onAction({ type: "palette", preset: n })}>{`${n} colours`}</button>)}
    </div>
  );
}

/** Big buttons for the clicked object. During a mission it only offers the answer (and never names the object). */
export function ContextCard({ id, actions, answering, onAction, onAnswer, onClose }: {
  id: string; actions: CardAction[]; answering: boolean; onAction(a: Action): void; onAnswer(a: Answer): void; onClose(): void;
}) {
  return (
    <div className="sv-glass sv-context" role="group" aria-label="object actions">
      <b>{answering ? "Selected object" : id}</b>
      {answering
        ? <button className="sv-btn sv-btn-primary" onClick={() => onAnswer({ type: "object", id })}>Answer: this one</button>
        : actions.map((a) => <button key={a.label} className="sv-btn sv-btn-primary" onClick={() => onAction(a.action)}>{a.label}</button>)}
      <button className="sv-pill" onClick={onClose}>Dismiss</button>
    </div>
  );
}

export const MAP_PX = 150;
/** world (x, z) <-> minimap pixels, aspect preserved */
export function mapScale(g: Ground, px = MAP_PX) {
  const k = px / Math.max(g.maxX - g.minX, g.maxZ - g.minZ);
  const ox = (px - k * (g.maxX - g.minX)) / 2, oy = (px - k * (g.maxZ - g.minZ)) / 2;
  // the dashboard panel (z = 0) lies outside the walkable box: pin it to the map edge
  const pin = (v: number) => Math.min(px - 4, Math.max(4, v));
  return {
    toMap: (x: number, z: number): [number, number] => [pin(ox + k * (x - g.minX)), pin(oy + k * (z - g.minZ))],
    toWorld: (u: number, v: number): [number, number] => [(u - ox) / k + g.minX, (v - oy) / k + g.minZ],
  };
}

export function Minimap({ ground, player, hazards, onFly, onWalk }: {
  ground: Ground; player: { x: number; z: number; yaw: number }; hazards: Facts["hazards"];
  onFly(id: string): void; onWalk(x: number, z: number): void;
}) {
  const m = mapScale(ground);
  const [px, py] = m.toMap(player.x, player.z);
  const walk = (e: MouseEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    onWalk(...m.toWorld(e.clientX - r.left, e.clientY - r.top));
  };
  return (
    <div className="sv-glass sv-minimap">
      <svg width={MAP_PX} height={MAP_PX} role="img" aria-label={`minimap, ${hazards.length} hazards`} onClick={walk}>
        {ground.beacons.map((b) => {
          const [x, y] = m.toMap(b.x, b.z);
          return (
            <circle key={b.id} className="sv-map-beacon" cx={x} cy={y} r={5} onClick={(e) => { e.stopPropagation(); onFly(b.id); }}>
              <title>{b.label}</title>
            </circle>
          );
        })}
        {hazards.map((h) => {
          const [x, y] = m.toMap(h.x, h.z);
          return <path key={h.id} className="sv-map-hazard" d={`M${x} ${y - 5}L${x + 5} ${y + 4}H${x - 5}Z`} />;
        })}
        <path className="sv-map-player" d="M0 -8L5 6L0 3L-5 6Z" transform={`translate(${px} ${py}) rotate(${(-player.yaw * 180) / Math.PI})`} />
      </svg>
      <small className="sv-note">○ waypoint · △ hazard</small>
    </div>
  );
}

export function PauseMenu({ onResume, onRestart, onMenu }: { onResume(): void; onRestart(): void; onMenu(): void }) {
  const first = useRef<HTMLButtonElement>(null);
  useEffect(() => { first.current?.focus(); }, []);
  return (
    <div className="sv-help">
      <div className="sv-glass sv-pause" role="dialog" aria-modal="true" aria-label="pause menu">
        <h2>Paused</h2>
        <button ref={first} className="sv-btn sv-btn-primary" onClick={onResume}>Resume</button>
        <button className="sv-btn" onClick={onRestart}>Restart scene</button>
        <button className="sv-btn" onClick={onMenu}>Back to menu</button>
      </div>
    </div>
  );
}
