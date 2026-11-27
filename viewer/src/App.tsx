import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { watchVisibility } from "./ui/budget";
import { useShell } from "./ui/shellContext";
import * as THREE from "three";
import {
  applyOverrides, exportSnapshot, getScene, postCorrect, SCENES, svMeshes, type Overrides, type Snapshot,
} from "../../adapters/threejs";
import { SimDriver, markerKey, precomputeMarkers } from "./sim/driver";
import {
  STEP_S, advanceClock, crossedMarker, keyAction, nextMarker, scrub, clampStep, type Marker,
} from "./sim/timeline";
import { RELIGHT_MS, startRelight } from "./sim/relight";
import { FxLayer } from "./sim/fxLayer";
import {
  DEFAULT_FX, cinematicBlockers, cinematicTooltip, effectiveFx, followWeight, topGroups, type Waypoint,
} from "./sim/fxPolicy";
import { EnvDial, FxBar, Timeline, type PlayState } from "./sim/SimControls";
import { easeInOutCubic } from "./ui/budget";
import { usePrefs } from "./ui/prefs";
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
  const prefs = usePrefs();
  const [fx, setFx] = useState(DEFAULT_FX);
  const blockers = cinematicBlockers(flags, hud.show);
  const eff = effectiveFx(fx, blockers, prefs.performance, prefs.still);
  const live = useRef({ obs: null as Obs | null, flags, force: false, fx, eff, still: prefs.still });
  Object.assign(live.current, { flags, fx, eff, still: prefs.still });
  const [play, setPlay] = useState<PlayState>({ step: 0, playing: true, speed: 1, live: true });
  const [marks, setMarks] = useState<{ key: string; list: Marker[] } | null>(null);
  const markKey = markerKey(sceneId);
  const markers = marks?.key === markKey ? marks.list : null;
  /** playback commands; the frame loop is the only consumer */
  const ctl = useRef({
    playing: true, speed: 1, live: true, seek: null as number | null, relight: null as string | null,
    step: 0, total: 1, markers: null as Marker[] | null, jumped: null as Marker | null,
  });
  ctl.current.markers = markers;
  const sync = () => {
    const c = ctl.current;
    setPlay({ step: c.step, playing: c.playing, speed: c.speed, live: c.live });
  };
  const command = (patch: Partial<typeof ctl.current>) => { Object.assign(ctl.current, patch); sync(); };
  const seekTo = (step: number) => {
    const s = clampStep(step, ctl.current.total);
    command({ seek: s, step: s, live: false });
  };
  const togglePlay = () => {
    const c = ctl.current;
    if (!c.playing && c.step >= c.total) seekTo(0);
    command({ playing: !c.playing, live: false });
  };
  const jumpNext = () => {
    const c = ctl.current, m = nextMarker(c.markers ?? [], c.step);
    if (m) { seekTo(m.step); c.jumped = m; }
  };
  const goLive = () => command({ live: true, playing: true, speed: 1 });
  const keys = useRef((_: KeyboardEvent) => {});
  keys.current = (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const el = e.target as HTMLInputElement | null;
    const a = keyAction(e.key, el && { tagName: el.tagName, type: el.type, role: el.getAttribute?.("role") });
    if (!a) return;
    e.preventDefault();
    if (a === "toggle") togglePlay();
    else seekTo(scrub(ctl.current.step, a === "back" ? -1 : 1, ctl.current.total));
  };
  useEffect(() => {
    const f = (e: KeyboardEvent) => keys.current(e);
    addEventListener("keydown", f);
    return () => removeEventListener("keydown", f);
  }, []);

  useEffect(() => {
    if (!getScene(sceneId)) return;
    let off = false;
    precomputeMarkers(sceneId, () => off)
      .then((list) => { if (list && !off) setMarks({ key: markKey, list }); })
      .catch((e) => setStatus(`timeline: ${e}`));
    return () => { off = true; };
  }, [sceneId, markKey]);

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
    if (!getScene(sceneId)) { setStatus(`unknown scene ${sceneId}`); return; }
    const panels = new ThreePanels(glRef.current!, PW, PH);
    panelsRef.current = panels;
    panels.setObserver(live.current.obs);
    const stat = new Stats(panels.renderer);
    const driver = new SimDriver(sceneId, PW / PH);
    const layer = new FxLayer();
    const C = ctl.current;
    Object.assign(C, { playing: true, speed: 1, live: true, seek: null, relight: null, jumped: null, step: 0, total: driver.total });
    live.current.force = true;
    const prev = new Map<string, [number, number]>();
    let frame = 0, prevT: number | undefined, raf = 0, busy = false, lastSnap: Snapshot | null = null, lastOv: Overrides | null = null;
    let lastHud = 0, lastMs: number | undefined, acc = 0, epoch = 0;
    let relight: { apply(k: number): void; start: number } | null = null;
    let waypoints: Waypoint[] = [];
    let follow: { point: THREE.Vector3; age: number } | null = null;
    let base = { pos: driver.camera.position.clone(), quat: driver.camera.quaternion.clone() };
    let visible = true;
    const unwatch = watchVisibility(viewRef.current!, (v) => { visible = v; });
    /** drop colour overrides: they belong to another time or environment */
    const invalidate = () => {
      epoch++; lastOv = null; lastSnap = null; waypoints = []; prev.clear(); prevT = undefined;
      panels.setOverrides(driver.scene, {} as Overrides, applyOverrides);
      setOv(null);
      live.current.force = true;
    };
    const rebased = () => { base = { pos: driver.camera.position.clone(), quat: driver.camera.quaternion.clone() }; };
    const followTo = (m: Marker) => {
      const mesh = svMeshes(driver.scene).find((x) => m.ids.includes(x.userData.sv.id as string));
      const point = m.point ? new THREE.Vector3(...m.point) : mesh?.getWorldPosition(new THREE.Vector3());
      if (point) follow = { point, age: 0 };
    };
    const moveCamera = (dt: number) => {
      const cam = driver.camera, L = live.current;
      let w = 0;
      if (follow && L.fx.follow) {
        follow.age += dt;
        w = L.still ? (followWeight(follow.age) > 0 ? 1 : 0) : followWeight(follow.age);
        if (follow.age > 1 && w === 0) follow = null;
      } else follow = null;
      cam.position.copy(base.pos); cam.quaternion.copy(base.quat);
      if (!follow || w === 0) return;
      cam.position.lerp(follow.point, 0.3 * w);
      const aim = new THREE.Matrix4().lookAt(cam.position, follow.point, cam.up);
      cam.quaternion.slerp(new THREE.Quaternion().setFromRotationMatrix(aim), 0.8 * w);
    };
    const advance = (dt: number) => {
      if (C.seek !== null && driver.ready) {
        const target = C.seek;
        C.seek = null; acc = 0;
        if (target < driver.step) { relight?.apply(1); relight = null; }
        if (driver.seek(target)) rebased();
        invalidate();
        if (C.jumped && live.current.fx.follow) followTo(C.jumped);
        C.jumped = null;
        return;
      }
      if (!C.playing || !driver.ready) return;
      const c = advanceClock(acc, dt, C.speed);
      acc = c.acc;
      const from = driver.step;
      driver.run(Math.min(c.steps, driver.total - from));
      const hit = live.current.fx.follow ? crossedMarker(C.markers ?? [], from, driver.step) : null;
      if (hit) followTo(hit);
      if (driver.step < driver.total) return;
      if (C.live) { relight?.apply(1); relight = null; driver.seek(0); rebased(); invalidate(); }
      else C.playing = false;
    };
    const loop = (ms: number) => {
      if (!visible) { lastMs = undefined; raf = requestAnimationFrame(loop); return; } // no GPU work or solves while off-screen
      const t = ms / 1000;
      const dt = lastMs === undefined ? 0 : Math.min(0.1, (ms - lastMs) / 1000);
      lastMs = ms;
      advance(dt);
      C.step = C.seek ?? driver.step;
      const L = live.current;
      if (C.relight !== null && driver.ready) {
        relight?.apply(1);
        invalidate(); // before the switch, so authored colours are restored first
        relight = { apply: startRelight(driver.scene, C.relight), start: ms };
        C.relight = null;
      }
      if (relight) {
        const x = L.still ? 1 : Math.min(1, (ms - relight.start) / RELIGHT_MS);
        relight.apply(easeInOutCubic(x));
        if (x >= 1) { relight = null; L.force = true; }
      }
      moveCamera(dt);
      const { scene, camera } = driver;
      if (L.eff.animate || L.eff.waypoints) {
        layer.update(camera, {
          dust: L.eff.dust, weather: L.eff.weather ? (scene.userData.svEnv?.name ?? null) : null,
          waypoints: L.eff.waypoints ? waypoints : [], animate: L.eff.animate, dt: L.eff.animate ? dt : 0, t,
        });
      }
      stat.begin();
      panels.render(scene, camera, {
        overlay: L.eff.animate || L.eff.waypoints ? layer.scene : null, cinematic: L.eff.cinematic,
      });
      if (!relight && (frame % SOLVE_EVERY === 0 || L.force) && !busy && L.obs) {
        busy = true; L.force = false;
        const at = epoch;
        const snap = exportSnapshot(scene, camera, { frame, t, observerId: L.obs.id, budgetMs: 40, prev, prevT });
        prevT = t;
        lastSnap = snap;
        waypoints = topGroups(snap.objects ?? []);
        postCorrect(API, snap)
          .then((o) => {
            if (at !== epoch) return; // solved for a state we have since left
            lastOv = o; panels.setOverrides(driver.scene, o, applyOverrides); setOv(o); setStatus("ok");
          })
          .catch((e) => setStatus(String(e)))
          .finally(() => { busy = false; });
      }
      if (ovRef.current) drawAll(ovRef.current, PW, PH, lastSnap?.objects ?? [], lastOv?.confusion_pairs ?? [], L.flags);
      stat.end(ms);
      if (ms - lastHud > 250) { lastHud = ms; setStats(stat.snapshot()); }
      if (frame % 6 === 0) sync();
      frame++;
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    const canvas = glRef.current!;
    return () => {
      cancelAnimationFrame(raf); unwatch(); epoch++; layer.dispose(); driver.dispose(); panels.dispose(); panelsRef.current = null;
      // release the context only once the canvas has left the page (it is reused on scene switches)
      queueMicrotask(() => { if (!canvas.isConnected) panels.renderer.forceContextLoss?.(); });
    };
  }, [sceneId]);

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
      <div className="sv-envbar">
        {sceneId === trafficScene.id && (
          <EnvDial label="environment" options={ENVS} value={env}
            onChange={(v) => { trafficScene.setEnv(v as EnvName); setEnvState(v as EnvName); ctl.current.relight = v; }} />
        )}
        {sceneId === dashboardScene.id && (
          <EnvDial label="lighting" options={DASH_LIGHTINGS} value={lighting}
            onChange={(v) => { dashboardScene.setEnv(v as DashLighting); setLightingState(v as DashLighting); ctl.current.relight = v; }} />
        )}
        <FxBar fx={fx} onChange={setFx} cinematicOn={eff.cinematic} blocked={prefs.performance || blockers.length > 0}
          tooltip={cinematicTooltip(blockers, prefs.performance)} performance={prefs.performance || prefs.still} />
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
      <Timeline state={play} total={ctl.current.total} markers={markers} follow={fx.follow}
        onSeek={seekTo} onToggle={togglePlay} onSpeed={(speed) => command({ speed, live: false })} onNext={jumpNext}
        onLive={goLive} onFollow={(v) => setFx({ ...fx, follow: v })} />
    </div>
  );
}
