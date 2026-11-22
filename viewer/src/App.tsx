import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import {
  applyOverrides, buildToyScene, exportSnapshot, postCorrect, stepToyScene, type Overrides, type Snapshot,
} from "../../adapters/threejs";
import { ThreePanels } from "./panels";
import { DEFAULT_FLAGS, drawAll, type OverlayFlags } from "./overlays";
import { Stats, type StatsSnapshot } from "./stats";

const API = "http://localhost:8000";
const PW = 400, PH = 260;
const SOLVE_EVERY = 15; // frames; server additionally caches by snapshot signature

interface Obs { id: string; split: string; model: string; kind: string; severity: number }

const OVERLAYS: [keyof OverlayFlags, string][] = [
  ["graph", "confusion graph"], ["heatmap", "importance heatmap"], ["bboxes", "bboxes"], ["labels", "ΔE00 labels"],
];

export function App() {
  const glRef = useRef<HTMLCanvasElement>(null);
  const ovRef = useRef<HTMLCanvasElement>(null);
  const [observers, setObservers] = useState<{ train: Obs[]; heldout: Obs[] }>({ train: [], heldout: [] });
  const [obsId, setObsId] = useState("");
  const [flags, setFlags] = useState<OverlayFlags>(DEFAULT_FLAGS);
  const [hud, setHud] = useState({ show: { tier: true, latency: true, stats: true, posterior: true } });
  const [ov, setOv] = useState<Overrides | null>(null);
  const [stats, setStats] = useState<StatsSnapshot | null>(null);
  const [status, setStatus] = useState("idle");
  const live = useRef({ obs: null as Obs | null, flags, force: false });
  live.current.flags = flags;

  useEffect(() => {
    fetch(`${API}/observers`).then((r) => r.json()).then((o) => {
      setObservers(o);
      const first = o.train[0] ?? o.heldout[0];
      if (first) setObsId(first.id);
    }).catch((e) => setStatus(`observers: ${e}`));
  }, []);

  useEffect(() => {
    const o = [...observers.train, ...observers.heldout].find((x) => x.id === obsId) ?? null;
    live.current.obs = o; live.current.force = true;
    panelsRef.current?.setObserver(o);
  }, [obsId, observers]);

  const panelsRef = useRef<ThreePanels | null>(null);

  useEffect(() => {
    const panels = new ThreePanels(glRef.current!, PW, PH);
    panelsRef.current = panels;
    panels.setObserver(live.current.obs);
    const stat = new Stats(panels.renderer);
    const { scene, camera } = buildToyScene(PW / PH);
    const prev = new Map<string, [number, number]>();
    let frame = 0, prevT: number | undefined, raf = 0, busy = false, lastSnap: Snapshot | null = null, lastOv: Overrides | null = null;
    let lastHud = 0;
    const loop = (ms: number) => {
      const t = ms / 1000;
      stepToyScene(scene, t);
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
    return () => { cancelAnimationFrame(raf); panels.dispose(); panelsRef.current = null; };
  }, []);

  const d = ov?.diagnostics;
  const f1 = (x: number | null | undefined) => (x == null ? "-" : x.toFixed(1));
  const box: React.CSSProperties = { border: "1px solid #444", padding: 6, minWidth: 190, background: "#181818" };
  const optGroup = (label: string, xs: Obs[]) => (
    <optgroup label={label}>
      {xs.map((o) => <option key={o.id} value={o.id}>{`${o.id} ${o.model}/${o.kind} ${o.severity.toFixed(2)}`}</option>)}
    </optgroup>
  );
  return (
    <div style={{ padding: 12 }}>
      <div style={{ marginBottom: 6 }}>
        observer:{" "}
        <select value={obsId} onChange={(e) => setObsId(e.target.value)}>
          {optGroup("train", observers.train)}{optGroup("held-out", observers.heldout)}
        </select>{" "}
        {OVERLAYS.map(([k, label]) => (
          <label key={k} style={{ marginLeft: 8 }}>
            <input type="checkbox" checked={flags[k]} onChange={(e) => setFlags({ ...flags, [k]: e.target.checked })} />{label}
          </label>
        ))}
        {(["tier", "latency", "stats", "posterior"] as const).map((k) => (
          <label key={k} style={{ marginLeft: 8 }}>
            <input type="checkbox" checked={hud.show[k]} onChange={(e) => setHud({ show: { ...hud.show, [k]: e.target.checked } })} />{k}
          </label>
        ))}
      </div>
      <div style={{ display: "flex", width: PW * 3, color: "#aaa" }}>
        {["A original", "B simulated CVD (original)", "C corrected → CVD"].map((t) => <div key={t} style={{ width: PW }}>{t}</div>)}
      </div>
      <div style={{ position: "relative", width: PW * 3, height: PH }}>
        <canvas ref={glRef} width={PW * 3} height={PH} style={{ width: PW * 3, height: PH, display: "block" }} />
        <canvas ref={ovRef} width={PW * 3} height={PH} style={{ position: "absolute", left: 0, top: 0, pointerEvents: "none" }} />
      </div>
      <div style={{ display: "flex", gap: 8, marginTop: 8, flexWrap: "wrap" }}>
        {hud.show.tier && <div style={box}>solver tier: {d?.solver_tier ?? "-"}<br />status: {status}<br />
          flags: {d?.fallback_flags?.join(",") || "-"}<br />cues: {ov?.cues?.length ?? 0} | pairs: {ov?.confusion_pairs?.length ?? 0}
          {" "}(infeasible {ov?.confusion_pairs?.filter((p) => !p.feasible).length ?? 0})</div>}
        {hud.show.latency && <div style={box}>latency ms<br />
          {Object.entries(d?.stage_latency_ms ?? {}).map(([k, v]) => <div key={k}>{k}: {v.toFixed(2)}</div>)}</div>}
        {hud.show.stats && <div style={box}>fps {f1(stats?.fps)} | frame {f1(stats?.frameMs)} ms<br />
          GPU {stats?.gpuMs == null ? "n/a" : `${stats.gpuMs.toFixed(2)} ms`} | heap {f1(stats?.heapMB)} MB<br />
          calls {stats?.calls ?? "-"} | tris {stats?.triangles ?? "-"}<br />geom {stats?.geometries ?? "-"} | tex {stats?.textures ?? "-"}</div>}
        {hud.show.posterior && <div style={{ ...box, color: "#777" }}>observer posterior<br />(placeholder — Phase 5)<br />
          selected: {obsId || "-"}</div>}
      </div>
    </div>
  );
}
