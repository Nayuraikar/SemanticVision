import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { watchVisibility } from "./ui/budget";
import { useShell } from "./ui/shellContext";
import * as THREE from "three";
import {
  applyOverrides, exportSnapshot, getScene, postCorrect, SCENES, type Overrides, type Snapshot,
} from "../../adapters/threejs";
import { ThreePanels } from "./panels";
import { DEFAULT_FLAGS, drawAll, type OverlayFlags } from "./overlays";
import { Stats, type StatsSnapshot } from "./stats";
import { API } from "./calibLib";
import { PROFILE_PREFIX, go, simPath, useAppState } from "./appState";
import { fetchObservers, fetchProfiles, type PresetObs, type SavedProfile } from "./screens";
import * as trafficScene from "../../adapters/threejs/scenes/traffic";
import { ENVS, type EnvName } from "../../adapters/threejs/environments";
import * as dashboardScene from "../../adapters/threejs/scenes/dashboard";
import { DASH_LIGHTINGS, type DashLighting } from "../../adapters/threejs/dashboardSim";

const PW = 400, PH = 260;
const SOLVE_EVERY = 15; // frames; server additionally caches by snapshot signature

interface Obs { id: string; split?: string; model: string; kind: string; severity: number }

const OVERLAYS: [keyof OverlayFlags, string][] = [
  ["graph", "confusion graph"], ["heatmap", "importance heatmap"], ["bboxes", "bboxes"], ["labels", "ΔE00 labels"],
];

export function App({ sceneId }: { sceneId: string }) {
  const glRef = useRef<HTMLCanvasElement>(null);
  const ovRef = useRef<HTMLCanvasElement>(null);
  const viewRef = useRef<HTMLDivElement>(null);
  const { barSlot, overlaySlot } = useShell();
  const { observerId, setObserver } = useAppState();
  const obsId = observerId ?? "";
  const [presets, setPresets] = useState<PresetObs[]>([]);
  const [profiles, setProfiles] = useState<SavedProfile[]>([]);
  const [flags, setFlags] = useState<OverlayFlags>(DEFAULT_FLAGS);
  const [hud, setHud] = useState({ show: { tier: true, latency: true, stats: true, posterior: true } });
  const [ov, setOv] = useState<Overrides | null>(null);
  const [stats, setStats] = useState<StatsSnapshot | null>(null);
  const [status, setStatus] = useState("idle");
  const [env, setEnvState] = useState<EnvName>(trafficScene.ENV);
  const [lighting, setLightingState] = useState<DashLighting>(dashboardScene.ENV);
  const live = useRef({ obs: null as Obs | null, flags, force: false });
  live.current.flags = flags;

  useEffect(() => {
    fetchObservers().then(setPresets).catch((e) => setStatus(`observers: ${e}`));
    fetchProfiles().then(setProfiles).catch((e) => setStatus(`profiles: ${e}`));
  }, []);

  useEffect(() => {
    const o: Obs | null = obsId.startsWith(PROFILE_PREFIX)
      ? (profiles.find((p) => p.observer_id === obsId) ? { ...profiles.find((p) => p.observer_id === obsId)!, id: obsId } : null)
      : presets.find((x) => x.id === obsId) ?? null;
    live.current.obs = o; live.current.force = true;
    panelsRef.current?.setObserver(o);
  }, [obsId, presets, profiles]);

  const panelsRef = useRef<ThreePanels | null>(null);

  useEffect(() => {
    const def = getScene(sceneId);
    if (!def) { setStatus(`unknown scene ${sceneId}`); return; }
    const panels = new ThreePanels(glRef.current!, PW, PH);
    panelsRef.current = panels;
    panels.setObserver(live.current.obs);
    const stat = new Stats(panels.renderer);
    const { scene, camera } = def.build(PW / PH);
    live.current.force = true;
    const prev = new Map<string, [number, number]>();
    let frame = 0, prevT: number | undefined, raf = 0, busy = false, lastSnap: Snapshot | null = null, lastOv: Overrides | null = null;
    let lastHud = 0;
    let visible = true;
    const unwatch = watchVisibility(viewRef.current!, (v) => { visible = v; });
    const loop = (ms: number) => {
      if (!visible) { raf = requestAnimationFrame(loop); return; } // no GPU work or solves while off-screen
      const t = ms / 1000;
      def.step(scene, t);
      stat.begin();
      panels.render(scene, camera);
      const L = live.current;
      if ((frame % SOLVE_EVERY === 0 || L.force) && !busy && L.obs) {
        busy = true; L.force = false;
        const snap = exportSnapshot(scene, camera, { frame, t, observerId: L.obs.id, budgetMs: 40, prev, prevT });
        lastSnap = snap;
        postCorrect(API, snap)
          .then((o) => { lastOv = o; panels.setOverrides(scene, o, applyOverrides); setOv(o); setStatus("ok"); })
          .catch((e) => setStatus(String(e)))
          .finally(() => { busy = false; });
      }
      if (lastSnap && ovRef.current) drawAll(ovRef.current, PW, PH, lastSnap.objects ?? [], lastOv?.confusion_pairs ?? [], L.flags);
      stat.end(ms);
      if (ms - lastHud > 250) { lastHud = ms; setStats(stat.snapshot()); }
      prevT = t; frame++;
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    const canvas = glRef.current!;
    return () => {
      cancelAnimationFrame(raf); unwatch(); panels.dispose(); panelsRef.current = null;
      // release the context only once the canvas has left the page (it is reused on scene/env switches)
      queueMicrotask(() => { if (!canvas.isConnected) panels.renderer.forceContextLoss?.(); });
    };
  }, [sceneId, env, lighting]);

  const d = ov?.diagnostics;
  const f1 = (x: number | null | undefined) => (x == null ? "-" : x.toFixed(1));
  const optGroup = (label: string, xs: PresetObs[]) => (
    <optgroup label={label}>
      {xs.map((o) => <option key={o.id} value={o.id}>{`${o.id} ${o.model}/${o.kind} ${o.severity.toFixed(2)}`}</option>)}
    </optgroup>
  );
  const barControls = (
    <>
      <label className="sv-pill"><span>scene</span>
        <select aria-label="scene" value={sceneId} onChange={(e) => go(simPath(e.target.value))}>
          {SCENES.map((s) => <option key={s.id} value={s.id}>{s.title}</option>)}
        </select>
      </label>
      {sceneId === trafficScene.id && (
        <label className="sv-pill"><span>env</span>
          <select aria-label="environment" value={env}
            onChange={(e) => { const v = e.target.value as EnvName; trafficScene.setEnv(v); setEnvState(v); }}>
            {ENVS.map((x) => <option key={x} value={x}>{x}</option>)}
          </select>
        </label>
      )}
      {sceneId === dashboardScene.id && (
        <label className="sv-pill"><span>lighting</span>
          <select aria-label="lighting" value={lighting}
            onChange={(e) => { const v = e.target.value as DashLighting; dashboardScene.setEnv(v); setLightingState(v); }}>
            {DASH_LIGHTINGS.map((x) => <option key={x} value={x}>{x}</option>)}
          </select>
        </label>
      )}
      <label className="sv-pill"><span>observer</span>
        <select aria-label="observer" value={obsId} onChange={(e) => setObserver(e.target.value)}>
          {!presets.some((o) => o.id === obsId) && !profiles.some((p) => p.observer_id === obsId) && <option value={obsId}>{obsId}</option>}
          {optGroup("train", presets.filter((o) => o.split === "train"))}{optGroup("held-out", presets.filter((o) => o.split !== "train"))}
          <optgroup label="saved profiles">
            {profiles.map((p) => <option key={p.id} value={p.observer_id}>{`${p.observer_id} ${p.kind} ${p.severity.toFixed(2)}`}</option>)}
          </optgroup>
        </select>
      </label>
      <button className="sv-pill" onClick={() => go("#/calibrate")}>re-run calibration</button>
    </>
  );
  const overlayControls = (
    <>
      {OVERLAYS.map(([k, label]) => (
        <label key={k}>
          <input type="checkbox" checked={flags[k]} onChange={(e) => setFlags({ ...flags, [k]: e.target.checked })} />{label}
        </label>
      ))}
      {(["tier", "latency", "stats", "posterior"] as const).map((k) => (
        <label key={k}>
          <input type="checkbox" checked={hud.show[k]} onChange={(e) => setHud({ show: { ...hud.show, [k]: e.target.checked } })} />{k}
        </label>
      ))}
    </>
  );
  return (
    <div className="sv-sim">
      {barSlot && createPortal(barControls, barSlot)}
      {overlaySlot && createPortal(overlayControls, overlaySlot)}
      <div className="sv-glass sv-stage">
        <div className="sv-stage-labels">
          {["A original", "B simulated CVD (original)", "C corrected → CVD"].map((t) => <div key={t}>{t}</div>)}
        </div>
        <div className="sv-stage-view" ref={viewRef}>
          <canvas ref={glRef} width={PW * 3} height={PH} />
          <canvas ref={ovRef} width={PW * 3} height={PH} style={{ pointerEvents: "none" }} />
        </div>
      </div>
      <div className="sv-hud">
        {hud.show.tier && <div className="sv-glass">solver tier: {d?.solver_tier ?? "-"}<br />status: {status}<br />
          flags: {d?.fallback_flags?.join(",") || "-"}<br />cues: {ov?.cues?.length ?? 0} | pairs: {ov?.confusion_pairs?.length ?? 0}
          {" "}(infeasible {ov?.confusion_pairs?.filter((p) => !p.feasible).length ?? 0})</div>}
        {hud.show.latency && <div className="sv-glass">latency ms<br />
          {Object.entries(d?.stage_latency_ms ?? {}).map(([k, v]) => <div key={k}>{k}: {v.toFixed(2)}</div>)}</div>}
        {hud.show.stats && <div className="sv-glass">fps {f1(stats?.fps)} | frame {f1(stats?.frameMs)} ms<br />
          GPU {stats?.gpuMs == null ? "n/a" : `${stats.gpuMs.toFixed(2)} ms`} | heap {f1(stats?.heapMB)} MB<br />
          calls {stats?.calls ?? "-"} | tris {stats?.triangles ?? "-"}<br />geom {stats?.geometries ?? "-"} | tex {stats?.textures ?? "-"}</div>}
        {hud.show.posterior && <div className="sv-glass" style={{ color: "var(--sv-text-dim)" }}>observer posterior<br />(placeholder — Phase 5)<br />
          selected: {obsId || "-"}</div>}
      </div>
    </div>
  );
}
