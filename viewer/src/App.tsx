import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import { createPortal } from "react-dom";
import { cappedPixelRatio, easeInOutCubic, watchVisibility } from "./ui/budget";
import { useShell } from "./ui/shellContext";
import * as THREE from "three";
import {
  applyOverrides, exportSnapshot, getScene, postCorrect, SCENES, svMeshes, type Overrides, type Snapshot,
} from "../../adapters/threejs";
import { SimDriver, markerKey, precomputeMarkers } from "./sim/driver";
import {
  advanceClock,crossedMarker, keyAction, nextMarker, scrub, clampStep, type Marker,
} from "./sim/timeline";
import { RELIGHT_MS, startRelight } from "./sim/relight";
import { FxLayer } from "./sim/fxLayer";
import {
  DEFAULT_FX, cinematicBlockers, cinematicTooltip, effectiveFx, followWeight, topGroups, type Waypoint,
} from "./sim/fxPolicy";
import { EnvDial, FxBar, Timeline, type PlayState } from "./sim/SimControls";
import {
  DEFAULT_VIEW, fitFov, isVisible, layoutTiles, lerpTiles, ndcIn, tileAt, type Tile, type ViewMode, type ViewState,
} from "./sim/view/layout";
import { OrbitRig } from "./sim/view/orbit";
import { DEFAULT_TOP_N, pickObject, projectPoint } from "./sim/view/geometry";
import { viewKey } from "./sim/view/keys";
import { ModeBar, ModeOptions, PaneLabels, WipeDivider } from "./sim/hud/ViewControls";
import { OverlayDrawer } from "./sim/hud/OverlayDrawer";
import { Drawer } from "./sim/hud/Drawer";
import { Inspector } from "./sim/hud/Inspector";
import { Diagnostics, solveInfo, type SolveInfo } from "./sim/hud/Diagnostics";
import { Help } from "./sim/hud/Help";
import { fetchProfileSummary, inspect, posteriorOf } from "./sim/hud/models";
import { usePrefs } from "./ui/prefs";
import { ThreePanels } from "./panels";
import { DEFAULT_FLAGS, drawOverlays, perObjectDe, type OverlayFlags } from "./overlays";
import { Stats, type StatsSnapshot } from "./stats";
import { buildCvdUniforms } from "./cvd";
import { API, type Summary } from "./calibLib";
import { PROFILE_PREFIX, go, simPath, useAppState } from "./appState";
import { fetchObservers, fetchProfiles, type PresetObs, type SavedProfile } from "./screens";
import * as trafficScene from "../../adapters/threejs/scenes/traffic";
import { ENVS, type EnvName } from "../../adapters/threejs/environments";
import * as dashboardScene from "../../adapters/threejs/scenes/dashboard";
import { DASH_LIGHTINGS, type DashLighting } from "../../adapters/threejs/dashboardSim";
import { initPhysics, physicsReady } from "../../adapters/threejs/physics";
import { PlaySession, type PlaySceneId } from "../../adapters/threejs/play/session";
import { actionsFor, type Action } from "../../adapters/threejs/play/actions";
import type { Facts } from "../../adapters/threejs/play/worlds";
import { Beeper, gaugeValueAt, keysFrom, moveKey } from "./play/input";
import {
  addTime, elapsedMs, score, sessionLog, startRun, viewCondition, viewFor,
  type Answer, type Mission, type MissionRecord, type MissionRun,
} from "./play/missions";
import { ContextCard, Minimap, MissionHud, PauseMenu, PlayPill, SceneControls, downloadLog } from "./play/PlayHud";

const SOLVE_EVERY = 15; // frames; server additionally caches by snapshot signature
/** used until the view element has a measurable size */
const FALLBACK_SIZE = { w: 1200, h: 700 };
const MODE_MS = 480, FEEL_MS = 1400, DRAG_PX = 4;
const PLAY_SEED = 1, BEACON_PX = 28, LOOK_RATE = 0.004;
/** clicking these walks there instead of selecting */
const WALKABLE = ["grass", "road", "sidewalk", "crosswalk", "road_marking", "sky", "panel"];
const DRAG_OBJECTS = ["car", "bush", "cone", "lamp"];
const GAUGE_PART = /^g_(eng|hyd|elec|fuel)_(needle|face|readout|arc_\w+)$/;
interface PlayHudState { facts: Facts; step: number; seconds: number; player: { x: number; z: number; yaw: number } }

interface Obs { id: string; split?: string; model: string; kind: string; severity: number }
type Side = "inspector" | "diagnostics" | null;

export function App({ sceneId }: { sceneId: string }) {
  const glRef = useRef<HTMLCanvasElement>(null);
  const ovRef = useRef<HTMLCanvasElement>(null);
  const viewRef = useRef<HTMLDivElement>(null);
  const cardRef = useRef<HTMLDivElement>(null);
  const { barSlot } = useShell();
  const { observerId, setObserver } = useAppState();
  const obsId = observerId ?? "";
  const [presets, setPresets] = useState<PresetObs[]>([]);
  const [profiles, setProfiles] = useState<SavedProfile[]>([]);
  const [flags, setFlags] = useState<OverlayFlags>(DEFAULT_FLAGS);
  const [topN, setTopN] = useState(DEFAULT_TOP_N);
  const [trayOpen, setTrayOpen] = useState(false);
  const [side, setSide] = useState<Side>(null);
  const [help, setHelp] = useState(false);
  const [view, setView] = useState<ViewState>(DEFAULT_VIEW);
  const [feeling, setFeeling] = useState(false);
  const [hover, setHover] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [ov, setOv] = useState<Overrides | null>(null);
  const [snapState, setSnapState] = useState<Snapshot | null>(null);
  const [solve, setSolve] = useState<SolveInfo | null>(null);
  const [stats, setStats] = useState<StatsSnapshot | null>(null);
  const [summary, setSummary] = useState<{ id: string; data: Partial<Summary> | null; error: string } | null>(null);
  const [status, setStatus] = useState("idle");
  const [env, setEnvState] = useState<EnvName>(trafficScene.ENV);
  const [lighting, setLightingState] = useState<DashLighting>(dashboardScene.ENV);
  const prefs = usePrefs();
  const [playMode, setPlayMode] = useState(false);
  const [paused, setPaused] = useState(false);
  const [restarts, setRestarts] = useState(0);
  const [card, setCard] = useState<string | null>(null);
  const [mission, setMission] = useState<Mission | null>(null);
  const [result, setResult] = useState<MissionRecord | null>(null);
  const [records, setRecords] = useState<MissionRecord[]>(() => [...sessionLog]);
  const [muted, setMuted] = useState(true);
  const [hud, setHud] = useState<PlayHudState | null>(null);
  const sessionRef = useRef<PlaySession | null>(null);
  const runRef = useRef<MissionRun | null>(null);
  const beeper = useRef(new Beeper());
  const held = useRef(new Set<string>());
  const sentKeys = useRef("");
  const act = (a: Action) => sessionRef.current?.push(a);
  const sendKeys = () => {
    const k = keysFrom(held.current), text = JSON.stringify(k);
    if (text === sentKeys.current) return;
    sentKeys.current = text;
    act({ type: "keys", ...k });
  };
  const releaseKeys = () => { held.current.clear(); sendKeys(); };
  const switchMode = (v: boolean) => {
    if (!v || physicsReady()) setPlayMode(v);
    else void initPhysics().then(() => setPlayMode(true));
  };
  const [fx, setFx] = useState(DEFAULT_FX);
  const diagOpen = side === "diagnostics";
  const blockers = cinematicBlockers(flags, { tier: diagOpen, latency: diagOpen, stats: diagOpen, posterior: diagOpen });
  const eff = effectiveFx(fx, blockers, prefs.performance, prefs.still);
  const live = useRef({
    obs: null as Obs | null, flags, topN, force: false, fx, eff, still: prefs.still, performance: prefs.performance,
    view, hover, selected, feelGoal: null as number | null, paused,
  });
  Object.assign(live.current, { flags, topN, fx, eff, still: prefs.still, performance: prefs.performance, view, hover, selected, paused });
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

  /** scene-side actions, installed by the frame-loop effect */
  const rigRef = useRef<{
    focus(id: string): void; reset(): void; pick(x: number, y: number): string | null; orbit: OrbitRig;
    /** play mode: pointer -> ground point, object lookup and screen positions */
    onPlane(x: number, y: number, normal: "y" | "z", at: number): THREE.Vector3 | null;
    mesh(id: string): THREE.Mesh | undefined;
    screenOf(p: THREE.Vector3, x: number, y: number): [number, number] | null;
  } | null>(null);
  const panelsRef = useRef<ThreePanels | null>(null);

  const changeView = (patch: Partial<ViewState>) => setView((v) => ({ ...v, ...patch }));
  const setMode = (mode: ViewMode) => {
    live.current.feelGoal = null; setFeeling(false);
    changeView({ mode, feel: 0 });
  };
  const toggleFeel = () => {
    const on = !feeling;
    setFeeling(on);
    if (live.current.still) { live.current.feelGoal = null; changeView({ feel: on ? 1 : 0 }); }
    else live.current.feelGoal = on ? 1 : 0;
  };
  const openSide = (s: Exclude<Side, null>) => setSide((cur) => (cur === s ? null : s));
  const select = (id: string | null) => {
    setSelected(id);
    if (!id) return;
    setSide("inspector");
    rigRef.current?.focus(id);
  };

  // ---- missions ----
  const startMission = (m: Mission, c: "B" | "C") => {
    live.current.feelGoal = null; setFeeling(true); changeView(viewFor(c));
    act({ type: "mission", name: m.id });
    runRef.current = startRun(m);
    setMission(m); setResult(null); setCard(null); setSelected(null); setSide(null);
  };
  const endMission = () => {
    runRef.current = null; setMission(null); setCard(null);
    act({ type: "mission", name: "end" });
  };
  const answer = (a: Answer) => {
    const run = runRef.current, s = sessionRef.current;
    if (!run || !s) return;
    const rec = score(run, a, s.facts(), obsId);
    sessionLog.push(rec);
    setRecords([...sessionLog]); setResult(rec);
    endMission();
  };

  const keys = useRef((_: KeyboardEvent) => {});
  keys.current = (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const el = e.target as HTMLInputElement | null;
    const target = el && { tagName: el.tagName, type: el.type, role: el.getAttribute?.("role") };
    if (playMode) {
      // Escape = pause menu; WASD / arrows / Q / E / Shift move; timeline keys do nothing
      if (e.key === "Escape") {
        e.preventDefault();
        if (help) setHelp(false);
        else { releaseKeys(); setPaused((p) => !p); }
        return;
      }
      if (paused) return;
      const k = moveKey(e.key, target);
      if (k) { e.preventDefault(); held.current.add(k); sendKeys(); return; }
    }
    const v = viewKey(e.key, target);
    if (v) {
      e.preventDefault();
      if ("mode" in v) setMode(v.mode);
      else if ("toggle" in v) {
        if (v.toggle === "help") setHelp((h) => !h);
        else if (v.toggle === "graph") setFlags((f) => ({ ...f, graph: !f.graph }));
        else openSide(v.toggle);
      } else if (v.action === "reset") rigRef.current?.reset();
      else if (help) setHelp(false);
      else if (side || trayOpen) { setSide(null); setTrayOpen(false); }
      else { setSelected(null); rigRef.current?.reset(); }
      return;
    }
    if (help || playMode) return;
    const a = keyAction(e.key, target);
    if (!a) return;
    e.preventDefault();
    if (a === "toggle") togglePlay();
    else seekTo(scrub(ctl.current.step, a === "back" ? -1 : 1, ctl.current.total));
  };
  useEffect(() => {
    const f = (e: KeyboardEvent) => keys.current(e);
    const up = (e: KeyboardEvent) => { if (held.current.delete(e.key.toLowerCase())) sendKeys(); };
    const blur = () => releaseKeys();
    addEventListener("keydown", f); addEventListener("keyup", up); addEventListener("blur", blur);
    return () => {
      removeEventListener("keydown", f); removeEventListener("keyup", up); removeEventListener("blur", blur);
      beeper.current.dispose();
    };
  }, []);

  // a new session (mode, scene or restart) starts with a clean HUD
  useEffect(() => {
    setPaused(false); setCard(null); setMission(null); setResult(null); setHud(null);
    runRef.current = null; held.current.clear(); sentKeys.current = "";
  }, [sceneId, playMode, restarts]);

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

  // full posterior (intervals) only when the drawer that shows it is open
  useEffect(() => {
    if (!diagOpen || !obsId.startsWith(PROFILE_PREFIX)) return;
    let off = false;
    fetchProfileSummary(obsId)
      .then((data) => { if (!off) setSummary({ id: obsId, data, error: "" }); })
      .catch((e) => { if (!off) setSummary({ id: obsId, data: null, error: String(e) }); });
    return () => { off = true; };
  }, [diagOpen, obsId]);

  useEffect(() => {
    const o: Obs | null = obsId.startsWith(PROFILE_PREFIX)
      ? (profiles.find((p) => p.observer_id === obsId) ? { ...profiles.find((p) => p.observer_id === obsId)!, id: obsId } : null)
      : presets.find((x) => x.id === obsId) ?? null;
    live.current.obs = o; live.current.force = true;
    panelsRef.current?.setObserver(o);
  }, [obsId, presets, profiles]);

  useEffect(() => {
    setHover(null); setSelected(null); setSnapState(null); setSolve(null);
    if (!getScene(sceneId)) { setStatus(`unknown scene ${sceneId}`); return; }
    const panels = new ThreePanels(glRef.current!);
    panelsRef.current = panels;
    panels.setObserver(live.current.obs);
    const stat = new Stats(panels.renderer);
    // Watch: the scripted driver. Play: a separate session driven by the input queue.
    const session = playMode ? new PlaySession({
      sceneId: sceneId as PlaySceneId, seed: PLAY_SEED, aspect: 16 / 9,
      env: sceneId === trafficScene.id ? trafficScene.ENV : sceneId === dashboardScene.id ? dashboardScene.ENV : undefined,
    }) : null;
    sessionRef.current = session;
    const driver = session ? null : new SimDriver(sceneId, 16 / 9);
    const src = (session ?? driver)!;
    const beacons: Waypoint[] = session?.ground.beacons.map((b) => ({
      key: b.id, score: 1, position: [b.x, sceneId === dashboardScene.id ? -0.35 : 0.15, b.z],
    })) ?? [];
    let alarms = 0;
    const layer = new FxLayer();
    const orbit = new OrbitRig();
    /** the one camera every view shares: the sim's authored camera plus orbit, with a per-tile aspect */
    const cam = new THREE.PerspectiveCamera();
    const C = ctl.current;
    Object.assign(C, { playing: true, speed: 1, live: true, seek: null, relight: null, jumped: null, step: 0, total: driver?.total ?? 1 });
    live.current.force = true;
    setOv(null);
    const prev = new Map<string, [number, number]>();
    let frame = 0, prevT: number | undefined, raf = 0, busy = false, lastSnap: Snapshot | null = null, lastOv: Overrides | null = null;
    let lastHud = 0, lastMs: number | undefined, acc = 0, epoch = 0;
    let relight: { apply(k: number): void; start: number } | null = null;
    let waypoints: Waypoint[] = [];
    let follow: { point: THREE.Vector3; age: number } | null = null;
    let base = { pos: src.camera.position.clone(), quat: src.camera.quaternion.clone() };
    let visible = true;
    const unwatch = watchVisibility(viewRef.current!, (v) => { visible = v; });
    const size = { ...FALLBACK_SIZE, ratio: 0 };
    const fit = () => {
      const el = viewRef.current;
      const w = el?.clientWidth || FALLBACK_SIZE.w, h = el?.clientHeight || FALLBACK_SIZE.h;
      const ratio = cappedPixelRatio(globalThis.devicePixelRatio, live.current.performance);
      if (w === size.w && h === size.h && ratio === size.ratio) return;
      Object.assign(size, { w, h, ratio });
      panels.resize(w, h, ratio);
      const o = ovRef.current;
      if (o) { o.width = Math.round(w * ratio); o.height = Math.round(h * ratio); }
      live.current.force = true;
    };
    let target: Tile[] = [], shown: Tile[] = [], from: Tile[] | null = null, since = 0, mode = live.current.view.mode;
    /** current tiles, cross-fading from the previous layout after a mode change */
    const tilesAt = (ms: number) => {
      const v = live.current.view;
      target = layoutTiles(v, size.w, size.h);
      if (v.mode !== mode) { mode = v.mode; from = shown.length && !live.current.still ? shown : null; since = ms; live.current.force = true; }
      const k = from ? Math.min(1, (ms - since) / MODE_MS) : 1;
      shown = from && k < 1 ? lerpTiles(from, target, easeInOutCubic(k)) : target;
      if (k >= 1) from = null;
      return shown;
    };
    const aimCamera = (tiles: Tile[], dt: number) => {
      const authored = src.camera, v = (tiles.find(isVisible) ?? tiles[0]).view;
      cam.aspect = v.w / Math.max(1, v.h);
      if (session) {
        cam.fov = fitFov(session.pose(cam), cam.aspect);
        cam.updateProjectionMatrix();
        return;
      }
      orbit.update(dt, live.current.still);
      cam.near = authored.near; cam.far = authored.far;
      cam.fov = fitFov(authored.fov, cam.aspect);
      cam.updateProjectionMatrix();
      orbit.apply(authored, cam);
    };
    const meshOf = (id: string) => svMeshes(src.scene).find((m) => m.userData.sv.id === id);
    rigRef.current = {
      orbit,
      mesh: meshOf,
      onPlane: (x, y, normal, at) => {
        const tile = tileAt(shown, x, y);
        if (!tile) return null;
        const ray = new THREE.Raycaster();
        ray.setFromCamera(new THREE.Vector2(...ndcIn(tile.view, x, y)), cam);
        const n = new THREE.Vector3(0, normal === "y" ? 1 : 0, normal === "z" ? 1 : 0);
        return ray.ray.intersectPlane(new THREE.Plane(n, -at), new THREE.Vector3());
      },
      screenOf: (p, x, y) => {
        const tile = tileAt(shown, x, y);
        return tile ? projectPoint(p, cam, tile.view) : null;
      },
      focus: (id) => {
        const m = session ? undefined : meshOf(id);
        if (m) orbit.focusOn(new THREE.Box3().setFromObject(m).getCenter(new THREE.Vector3()));
      },
      reset: () => orbit.reset(),
      pick: (x, y) => {
        const tile = tileAt(shown, x, y);
        return tile ? pickObject(src.scene, cam, ndcIn(tile.view, x, y)) : null;
      },
    };
    /** drop colour overrides: they belong to another time or environment */
    const invalidate = () => {
      epoch++; lastOv = null; lastSnap = null; waypoints = []; prev.clear(); prevT = undefined;
      panels.setOverrides(src.scene, {} as Overrides, applyOverrides);
      setOv(null); setSnapState(null);
      live.current.force = true;
    };
    const rebased = () => { base = { pos: src.camera.position.clone(), quat: src.camera.quaternion.clone() }; };
    const followTo = (m: Marker) => {
      const mesh = svMeshes(src.scene).find((x) => m.ids.includes(x.userData.sv.id as string));
      const point = m.point ? new THREE.Vector3(...m.point) : mesh?.getWorldPosition(new THREE.Vector3());
      if (point) follow = { point, age: 0 };
    };
    const moveCamera = (dt: number) => {
      if (session) return;
      const authored = src.camera, L = live.current;
      let w = 0;
      if (follow && L.fx.follow) {
        follow.age += dt;
        w = L.still ? (followWeight(follow.age) > 0 ? 1 : 0) : followWeight(follow.age);
        if (follow.age > 1 && w === 0) follow = null;
      } else follow = null;
      authored.position.copy(base.pos); authored.quaternion.copy(base.quat);
      if (!follow || w === 0) return;
      authored.position.lerp(follow.point, 0.3 * w);
      const aim = new THREE.Matrix4().lookAt(authored.position, follow.point, authored.up);
      authored.quaternion.slerp(new THREE.Quaternion().setFromRotationMatrix(aim), 0.8 * w);
    };
    const playAdvance = (s: PlaySession, dt: number) => {
      if (live.current.paused) return;
      const c = advanceClock(acc, dt, 1);
      acc = c.acc;
      for (let i = 0; i < c.steps; i++) {
        if (s.willInvalidate()) invalidate(); // restore authored colours before they change
        s.tick();
      }
      const run = runRef.current;
      if (run) addTime(run, viewCondition(live.current.view), dt * 1000);
      const now = s.facts().alarms;
      if (now > alarms && !live.current.performance) beeper.current.beep();
      alarms = now;
    };
    const advance = (dt: number) => {
      if (session) return playAdvance(session, dt);
      if (!driver) return;
      if (C.seek !== null && driver.ready) {
        const goal = C.seek;
        C.seek = null; acc = 0;
        if (goal < driver.step) { relight?.apply(1); relight = null; }
        if (driver.seek(goal)) rebased();
        invalidate();
        if (C.jumped && live.current.fx.follow) followTo(C.jumped);
        C.jumped = null;
        return;
      }
      if (!C.playing || !driver.ready) return;
      const c = advanceClock(acc, dt, C.speed);
      acc = c.acc;
      const start = driver.step;
      driver.run(Math.min(c.steps, driver.total - start));
      const hit = live.current.fx.follow ? crossedMarker(C.markers ?? [], start, driver.step) : null;
      if (hit) followTo(hit);
      if (driver.step < driver.total) return;
      if (C.live) { relight?.apply(1); relight = null; driver.seek(0); rebased(); invalidate(); }
      else C.playing = false;
    };
    const feelStep = (dt: number) => {
      const L = live.current, goal = L.feelGoal;
      if (goal === null) return;
      const d = (dt * 1000) / FEEL_MS;
      const next = goal > L.view.feel ? Math.min(goal, L.view.feel + d) : Math.max(goal, L.view.feel - d);
      if (next === goal) L.feelGoal = null;
      L.view = { ...L.view, feel: next };
      setView((v) => ({ ...v, feel: next }));
    };
    /** the hover card follows the hovered object in the first view that shows it */
    const placeCard = (tiles: Tile[]) => {
      const card = cardRef.current, id = live.current.hover;
      const m = card && id ? meshOf(id) : undefined;
      if (!card || !m) return;
      const p = new THREE.Box3().setFromObject(m).getCenter(new THREE.Vector3());
      for (const t of tiles) {
        const at = isVisible(t) ? projectPoint(p, cam, t.view) : null;
        if (!at || tileAt(tiles, at[0], at[1]) !== t) continue;
        card.style.transform = `translate(${Math.min(size.w - 230, at[0] + 18)}px, ${Math.max(8, at[1] - 90)}px)`;
        return;
      }
    };
    const loop = (ms: number) => {
      if (!visible) { lastMs = undefined; raf = requestAnimationFrame(loop); return; } // no GPU work or solves while off-screen
      const t = ms / 1000;
      const dt = lastMs === undefined ? 0 : Math.min(0.1, (ms - lastMs) / 1000);
      lastMs = ms;
      fit();
      advance(dt);
      C.step = C.seek ?? src.step;
      const L = live.current;
      if (C.relight !== null && driver?.ready) {
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
      feelStep(dt);
      moveCamera(dt);
      const { scene } = src;
      const tiles = tilesAt(ms);
      aimCamera(tiles, dt);
      if (L.eff.animate || L.eff.waypoints) {
        layer.update(cam, {
          dust: L.eff.dust, weather: L.eff.weather ? (scene.userData.svEnv?.name ?? null) : null,
          waypoints: !L.eff.waypoints ? [] : session ? (L.performance ? [] : beacons) : waypoints,
          animate: L.eff.animate, dt: L.eff.animate ? dt : 0, t, max: session ? beacons.length : undefined,
        });
      }
      const settled = (target.find(isVisible) ?? target[0]).view.w;
      stat.begin();
      panels.render(scene, cam, tiles, settled, {
        overlay: L.eff.animate || L.eff.waypoints ? layer.scene : null, cinematic: L.eff.cinematic,
      });
      if (!relight && !from && (frame % SOLVE_EVERY === 0 || L.force) && !busy && L.obs) {
        busy = true; L.force = false;
        const at = epoch;
        const snap = exportSnapshot(scene, cam, { frame, t, observerId: L.obs.id, budgetMs: 40, prev, prevT });
        prevT = t;
        lastSnap = snap;
        waypoints = topGroups(snap.objects ?? []);
        postCorrect(API, snap)
          .then((o) => {
            if (at !== epoch) return; // solved for a state we have since left
            lastOv = o; panels.setOverrides(src.scene, o, applyOverrides);
            setOv(o); setSnapState(snap); setSolve((p) => solveInfo(o, p)); setStatus("ok");
          })
          .catch((e) => setStatus(String(e)))
          .finally(() => { busy = false; });
      }
      const g = ovRef.current?.getContext("2d");
      if (g) {
        g.setTransform(size.ratio, 0, 0, size.ratio, 0, 0);
        g.clearRect(0, 0, size.w, size.h);
        drawOverlays(g, {
          scene, camera: cam, tiles, objects: lastSnap?.objects ?? [], pairs: lastOv?.confusion_pairs ?? [],
          flags: L.flags, topN: L.topN, hover: L.hover, selected: L.selected, t: L.still ? 0 : t,
        });
      }
      placeCard(tiles);
      stat.end(ms);
      if (ms - lastHud > 250) { lastHud = ms; setStats(stat.snapshot()); }
      if (frame % 6 === 0) {
        sync();
        if (session) setHud({
          facts: session.facts(), step: session.step, seconds: runRef.current ? elapsedMs(runRef.current) / 1000 : 0,
          player: { x: session.player.x, z: session.player.z, yaw: session.player.yaw },
        });
      }
      frame++;
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    const canvas = glRef.current!, surface = ovRef.current!;
    const wheel = (e: WheelEvent) => {
      e.preventDefault();
      const factor = Math.exp(e.deltaY * 0.0012);
      if (!session) orbit.zoomBy(factor);
      else if (!live.current.paused) session.push({ type: "zoom", factor });
    };
    surface.addEventListener("wheel", wheel, { passive: false });
    return () => {
      cancelAnimationFrame(raf); unwatch(); surface.removeEventListener("wheel", wheel); rigRef.current = null;
      epoch++; layer.dispose(); src.dispose(); sessionRef.current = null; panels.dispose(); panelsRef.current = null;
      // release the context only once the canvas has left the page (it is reused on scene switches)
      queueMicrotask(() => { if (!canvas.isConnected) panels.renderer.forceContextLoss?.(); });
    };
  }, [sceneId, playMode, restarts]);

  // ---- pointer: drag orbits, hover glows, click inspects and flies to the object ----
  const drag = useRef<{ x: number; y: number; moved: boolean; kind: "look" | "gauge" | "object"; id: string | null } | null>(null);
  /** what a drag starting on `id` manipulates in play mode */
  const dragKind = (id: string | null) => {
    if (!playMode || !id || (mission && !mission.done)) return "look" as const;
    if (sceneId === dashboardScene.id && GAUGE_PART.test(id)) return "gauge" as const;
    return sceneId === "toy" && DRAG_OBJECTS.includes(id) ? "object" as const : "look" as const;
  };
  const playDrag = (d: NonNullable<typeof drag.current>, dx: number, dy: number, x: number, y: number) => {
    const rig = rigRef.current!;
    if (d.kind === "gauge") {
      const system = GAUGE_PART.exec(d.id!)![1] as "eng" | "hyd" | "elec" | "fuel";
      const face = rig.mesh(`g_${system}_face`), c = face && rig.screenOf(face.position, x, y);
      if (c) act({ type: "gauge", system, value: gaugeValueAt(x - c[0], y - c[1]) });
    } else if (d.kind === "object") {
      const m = rig.mesh(d.id!), p = m && rig.onPlane(x, y, "y", m.position.y);
      if (p) act({ type: "drag", id: d.id!, x: p.x, z: p.z });
    } else act({ type: "look", yaw: dx * LOOK_RATE, pitch: dy * LOOK_RATE });
  };
  const playClick = (x: number, y: number) => {
    const rig = rigRef.current, s = sessionRef.current;
    if (!rig || !s) return;
    const beacon = prefs.performance || !fx.waypoints ? undefined : s.ground.beacons.find((b) => {
      const at = rig.screenOf(new THREE.Vector3(b.x, sceneId === dashboardScene.id ? -0.35 : 0.15, b.z), x, y);
      return !!at && Math.hypot(at[0] - x, at[1] - y) < BEACON_PX;
    });
    if (beacon) { act({ type: "fly", id: beacon.id }); return; }
    const id = rig.pick(x, y);
    const cls = id ? (rig.mesh(id)?.userData.sv.cls as string | undefined) : undefined;
    if (!id || WALKABLE.includes(cls ?? "")) {
      setCard(null); setSelected(null);
      // the instrument panel has no floor: step sideways towards the clicked spot
      if (sceneId === dashboardScene.id) {
        const p = rig.onPlane(x, y, "z", 0);
        if (p) act({ type: "walk", x: p.x, z: s.player.z });
      } else {
        const p = rig.onPlane(x, y, "y", 0);
        if (p) act({ type: "walk", x: p.x, z: p.z });
      }
      return;
    }
    setCard(id);
    if (mission) { setSelected(id); return; } // during a mission a click only picks an answer
    select(id);
    const primary = actionsFor(sceneId, id)[0];
    if (primary) act(primary.action);
  };
  const local = (e: ReactPointerEvent<Element>) => {
    const r = e.currentTarget.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top] as const;
  };
  const lastPick = useRef(0);
  const pointerDown = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    if (e.button !== 0 || paused) return;
    const id = playMode ? rigRef.current?.pick(...local(e)) ?? null : null;
    drag.current = { x: e.clientX, y: e.clientY, moved: false, kind: dragKind(id), id };
    e.currentTarget.setPointerCapture?.(e.pointerId);
  };
  const pointerMove = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    const d = drag.current;
    if (d) {
      const dx = e.clientX - d.x, dy = e.clientY - d.y;
      if (!d.moved && Math.hypot(dx, dy) < DRAG_PX) return;
      d.moved = true; d.x = e.clientX; d.y = e.clientY;
      if (playMode) { if (rigRef.current && !paused) playDrag(d, dx, dy, ...local(e)); }
      else rigRef.current?.orbit.rotate(-dx * 0.005, dy * 0.004);
      setHover(null);
      return;
    }
    if (e.timeStamp - lastPick.current < 60) return;
    lastPick.current = e.timeStamp;
    const [x, y] = local(e);
    setHover(rigRef.current?.pick(x, y) ?? null);
  };
  const pointerUp = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    const d = drag.current;
    drag.current = null;
    if (!d || d.moved) return;
    const [x, y] = local(e);
    if (playMode) { playClick(x, y); return; }
    const id = rigRef.current?.pick(x, y) ?? null;
    select(id);
    if (!id) rigRef.current?.reset();
  };

  const objects = snapState?.objects ?? [];
  const ids = objects.map((o) => o.id).sort((a, b) => a.localeCompare(b));
  const observer: Obs | null = presets.find((p) => p.id === obsId)
    ?? (profiles.find((p) => p.observer_id === obsId) ? { ...profiles.find((p) => p.observer_id === obsId)!, id: obsId } : null);
  // during a mission object details would give the answer away
  const inspection = side === "inspector" && selected && !mission ? inspect(selected, snapState, ov, solve?.tier ?? null) : null;
  const mySummary = summary?.id === obsId ? summary : null;
  const hovered = hover ? objects.find((o) => o.id === hover) : undefined;
  const hoveredDe = hover ? perObjectDe(ov?.confusion_pairs ?? []).get(hover) : undefined;

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
  const dial = sceneId === trafficScene.id ? (
    <EnvDial label="environment" options={ENVS} value={env}
      onChange={(v) => {
        trafficScene.setEnv(v as EnvName); setEnvState(v as EnvName);
        if (playMode) act({ type: "env", env: v as EnvName });
        else ctl.current.relight = v;
      }} />
  ) : sceneId === dashboardScene.id && !playMode ? (
    <EnvDial label="lighting" options={DASH_LIGHTINGS} value={lighting}
      onChange={(v) => { dashboardScene.setEnv(v as DashLighting); setLightingState(v as DashLighting); ctl.current.relight = v; }} />
  ) : null;
  return (
    <div className="sv-sim" data-side={side ?? "none"} data-mode={playMode ? "play" : "watch"}>
      {barSlot && createPortal(barControls, barSlot)}
      <div className="sv-view" ref={viewRef}>
        <canvas ref={glRef} aria-hidden="true" />
        <canvas ref={ovRef} className="sv-view-input" data-hover={!!hover} role="img"
          aria-label={playMode
            ? "3D scene, play mode. WASD or arrows move, Q and E turn, drag to look, scroll to zoom, click to walk or act, Escape pauses."
            : "3D scene. Drag to orbit, scroll to zoom, click an object to inspect it. Use the inspector to pick objects by keyboard."}
          onPointerDown={pointerDown} onPointerMove={pointerMove} onPointerUp={pointerUp}
          onPointerCancel={() => { drag.current = null; }} onPointerLeave={() => setHover(null)}
          onDoubleClick={() => rigRef.current?.reset()} />
        <PaneLabels view={view} />
        {view.mode === "wipe" && <WipeDivider value={view.divider} onChange={(divider) => changeView({ divider })} />}
        {hover && hover !== selected && !mission && (
          <div ref={cardRef} className="sv-glass sv-hovercard" role="status">
            <b>{hover}</b>
            {hovered && <small>{`${hovered.class} · ${hovered.state ?? "default"}`}</small>}
            <small>{hoveredDe ? `ΔE00 ${hoveredDe[0].toFixed(1)} → ${hoveredDe[1].toFixed(1)}` : "no confusable pair reported"}</small>
          </div>
        )}
      </div>

      <div className="sv-topleft">
        <OverlayDrawer open={trayOpen} onOpen={setTrayOpen} flags={flags} onChange={setFlags} topN={topN} onTopN={setTopN} />
        <PlayPill play={playMode} onChange={switchMode} />
      </div>
      {playMode && card && !mission?.choices && (
        <ContextCard id={card} actions={actionsFor(sceneId, card)} answering={!!mission && !mission.done && !mission.choices}
          onAction={act} onAnswer={answer} onClose={() => setCard(null)} />
      )}
      {playMode && hud && !prefs.performance && sessionRef.current && (
        <Minimap ground={sessionRef.current.ground} player={hud.player} hazards={mission ? [] : hud.facts.hazards}
          onFly={(id) => act({ type: "fly", id })} onWalk={(x, z) => act({ type: "walk", x, z })} />
      )}
      {playMode && paused && (
        <PauseMenu onResume={() => setPaused(false)} onRestart={() => { setPaused(false); setRestarts((n) => n + 1); }}
          onMenu={() => go("#/scenes")} />
      )}
      <ModeBar mode={view.mode} onMode={setMode} />
      <div className="sv-topright">
        <button className="sv-pill sv-solidpill" aria-expanded={side === "inspector"} aria-controls="sv-inspector" aria-keyshortcuts="I"
          onClick={() => openSide("inspector")}>Inspector</button>
        <button className="sv-pill sv-solidpill" aria-expanded={diagOpen} aria-controls="sv-diagnostics" aria-keyshortcuts="D"
          onClick={() => openSide("diagnostics")}>Diagnostics</button>
        <button className="sv-pill sv-solidpill" aria-expanded={help} aria-label="help" aria-keyshortcuts="?" onClick={() => setHelp(!help)}>?</button>
      </div>
      {status !== "ok" && status !== "idle" && <p className="sv-glass sv-alert" role="alert">{status}</p>}

      <div className="sv-dock">
        <ModeOptions view={view} onChange={(p) => {
          if (p.source) { live.current.feelGoal = null; setFeeling(false); changeView({ ...p, feel: 0 }); } else changeView(p);
        }} onFeel={toggleFeel} feeling={feeling} />
        <div className="sv-trays">
          {/* hidden during missions: gauge sliders would show the answer and actions could change it */}
          {playMode && (!mission || mission.done) && (
            <SceneControls scene={sceneId} facts={hud?.facts ?? null} lighting={lighting} onAction={act}
              onLighting={(v) => { dashboardScene.setEnv(v); setLightingState(v); act({ type: "lighting", env: v }); }} />
          )}
          {dial ?? <span />}
          <div className="sv-glass sv-fxtray" role="group" aria-label="effects">
            <FxBar fx={fx} onChange={setFx} cinematicOn={eff.cinematic} blocked={prefs.performance || blockers.length > 0}
              tooltip={cinematicTooltip(blockers, prefs.performance)} performance={prefs.performance || prefs.still} />
          </div>
        </div>
        {playMode ? (
          <MissionHud scene={sceneId} step={hud?.step ?? 0} beacons={sessionRef.current?.ground.beacons ?? []}
            active={mission} seconds={hud?.seconds ?? 0} result={result} records={records}
            muted={muted} audioOff={prefs.performance}
            onStart={startMission} onAnswer={answer} onCancel={endMission} onCloseResult={() => setResult(null)}
            onFly={(id) => act({ type: "fly", id })}
            onMute={(m) => { beeper.current.setMuted(m); setMuted(m); }}
            onDownload={() => downloadLog(records)} onPause={() => { releaseKeys(); setPaused(true); }} />
        ) : (
          <Timeline state={play} total={ctl.current.total} markers={markers} follow={fx.follow}
            onSeek={seekTo} onToggle={togglePlay} onSpeed={(speed) => command({ speed, live: false })} onNext={jumpNext}
            onLive={goLive} onFollow={(v) => setFx({ ...fx, follow: v })} />
        )}
      </div>

      <Drawer id="sv-inspector" title="Inspector" open={side === "inspector"} onClose={() => setSide(null)}>
        <Inspector data={inspection} selected={selected} ids={ids} cvd={observer ? buildCvdUniforms(observer) : null} onSelect={select} />
      </Drawer>
      <Drawer id="sv-diagnostics" title="Diagnostics" open={diagOpen} onClose={() => setSide(null)}>
        <Diagnostics ov={ov} solve={solve} stats={stats} status={status} observerId={obsId}
          posterior={posteriorOf(obsId, presets, profiles, mySummary?.data ?? null)} posteriorError={mySummary?.error ?? ""} />
      </Drawer>
      <Help open={help} onClose={() => setHelp(false)} />
    </div>
  );
}
