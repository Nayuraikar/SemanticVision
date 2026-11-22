import { useEffect, useState } from "react";
import { sceneCards } from "../../adapters/threejs/scenes/registry";
import { API } from "./calibLib";
import { go, simPath, useAppState } from "./appState";

export interface PresetObs { id: string; split: string; model: string; kind: string; severity: number }
export interface SavedProfile { id: string; observer_id: string; n_trials: number; kind: string; severity: number; model: string }

export async function fetchObservers(): Promise<PresetObs[]> {
  const r = await fetch(`${API}/observers`);
  if (!r.ok) throw new Error(`GET /observers ${r.status}`);
  const o = await r.json();
  return [...(o.train ?? []), ...(o.heldout ?? [])];
}
export async function fetchProfiles(): Promise<SavedProfile[]> {
  const r = await fetch(`${API}/profiles`);
  if (!r.ok) throw new Error(`GET /profiles ${r.status}`);
  return r.json();
}

const page: React.CSSProperties = { maxWidth: 820, margin: "0 auto", padding: 20, font: "14px system-ui, sans-serif", lineHeight: 1.5 };
const btn: React.CSSProperties = { padding: "8px 18px", fontSize: 14, marginRight: 8, cursor: "pointer" };
const card: React.CSSProperties = { border: "1px solid #444", padding: 10, background: "#181818" };

export function Welcome() {
  return (
    <div style={page}>
      <h1 style={{ marginTop: 0 }}>SemanticVision</h1>
      <p>
        This is an accessibility research tool. It simulates how a scene may look to people with different colour vision
        and tests automatic colour corrections. It is <b>not a diagnosis</b> and not a medical device.
      </p>
      <button style={btn} onClick={() => go("#/observer")}>Continue</button>
    </div>
  );
}

export function ObserverStep() {
  const { observerId, setObserver } = useAppState();
  const [presets, setPresets] = useState<PresetObs[]>([]);
  const [profiles, setProfiles] = useState<SavedProfile[]>([]);
  const [err, setErr] = useState("");
  useEffect(() => {
    fetchObservers().then(setPresets).catch((e) => setErr(String(e)));
    fetchProfiles().then(setProfiles).catch((e) => setErr(String(e)));
  }, []);
  const choose = (id: string) => { setObserver(id); go("#/scenes"); };
  return (
    <div style={page}>
      <h2 style={{ marginTop: 0 }}>Choose an observer</h2>
      {err && <p style={{ color: "#f88" }}>{err}</p>}
      <h3>Calibrate</h3>
      <p>Answer a few colour-matching questions to estimate your own parameters.</p>
      <button style={btn} onClick={() => go("#/calibrate")}>Run calibration</button>
      <h3>Use a preset observer</h3>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
        {presets.map((o) => (
          <button key={o.id} style={{ ...btn, margin: 0 }} aria-pressed={observerId === o.id} onClick={() => choose(o.id)}>
            {`preset ${o.id}: ${o.kind} ${o.severity.toFixed(2)} (${o.split})`}
          </button>
        ))}
        {!presets.length && <span style={{ opacity: 0.6 }}>no presets available</span>}
      </div>
      <h3>Load saved profile</h3>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
        {profiles.map((p) => (
          <button key={p.id} style={{ ...btn, margin: 0 }} aria-pressed={observerId === p.observer_id} onClick={() => choose(p.observer_id)}>
            {`profile ${p.id}: ${p.kind} ${p.severity.toFixed(2)}, ${p.n_trials} trials`}
          </button>
        ))}
        {!profiles.length && <span style={{ opacity: 0.6 }}>no saved profiles</span>}
      </div>
    </div>
  );
}

export function ScenePicker() {
  const { setScene } = useAppState();
  return (
    <div style={page}>
      <h2 style={{ marginTop: 0 }}>Choose a scene</h2>
      <p><a href="#/observer" style={{ color: "#8cf" }}>change observer</a></p>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(240px, 1fr))", gap: 10 }}>
        {sceneCards().map((c) => (
          <div key={c.id} style={{ ...card, opacity: c.disabled ? 0.5 : 1 }} data-testid={`scene-${c.id}`}>
            <h3 style={{ margin: "0 0 4px" }}>{c.title}</h3>
            <p style={{ margin: "0 0 6px" }}>{c.description}</p>
            <p style={{ margin: "0 0 8px", opacity: 0.7 }}>{c.tags.join(" · ")}</p>
            {c.disabled
              ? <button style={btn} disabled>Coming soon</button>
              : <button style={btn} onClick={() => { setScene(c.id); go(simPath(c.id)); }}>Open</button>}
          </div>
        ))}
      </div>
    </div>
  );
}
