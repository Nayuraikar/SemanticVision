import { useEffect, useRef, useState } from "react";
import { sceneCards } from "../../adapters/threejs/scenes/registry";
import { API } from "./calibLib";
import { go, simPath, useAppState } from "./appState";
import { GlassCard } from "./ui/GlassCard";
import { useShell } from "./ui/shellContext";

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

export function Welcome() {
  return (
    <div className="sv-hero">
      <h1>SemanticVision</h1>
      <p className="sv-glass">
        This is an accessibility research tool. It simulates how a scene may look to people with different colour vision
        and tests automatic colour corrections. It is <b>not a diagnosis</b> and not a medical device.
      </p>
      <button className="sv-btn sv-btn-primary" onClick={() => go("#/observer")}>Continue</button>
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
    <div className="sv-page">
      <h2 className="sv-title">Choose an observer</h2>
      <p className="sv-lead">The observer is the model of colour vision that the scene is simulated and corrected for.</p>
      {err && <p className="sv-error">{err}</p>}
      <div className="sv-grid">
        <GlassCard>
          <span className="sv-badge">Recommended</span>
          <h3>Calibrate (recommended)</h3>
          <p>Answer a few colour-matching questions to estimate your own parameters. The result fits the correction to your answers instead of a generic type.</p>
          <button className="sv-btn sv-btn-primary" onClick={() => go("#/calibrate")}>Run calibration</button>
        </GlassCard>
        <GlassCard>
          <h3>Preset observer</h3>
          <p>A ready-made example of one colour-vision type. Strength runs from 0 (typical) to 1 (strongest). Good for exploring.</p>
          <div className="sv-choices">
            {presets.map((o) => (
              <button key={o.id} className="sv-btn" aria-pressed={observerId === o.id} onClick={() => choose(o.id)}>
                <span>{`preset ${o.id}`}</span><small>{`${o.kind} ${o.severity.toFixed(2)} (${o.split})`}</small>
              </button>
            ))}
            {!presets.length && <p>no presets available</p>}
          </div>
        </GlassCard>
        <GlassCard>
          <h3>Saved profile</h3>
          <p>Reuse the result of a calibration you finished earlier.</p>
          <div className="sv-choices">
            {profiles.map((p) => (
              <button key={p.id} className="sv-btn" aria-pressed={observerId === p.observer_id} onClick={() => choose(p.observer_id)}>
                <span>{`profile ${p.id}`}</span><small>{`${p.kind} ${p.severity.toFixed(2)}, ${p.n_trials} trials`}</small>
              </button>
            ))}
            {!profiles.length && <p>no saved profiles</p>}
          </div>
        </GlassCard>
      </div>
    </div>
  );
}

export function ScenePicker() {
  const { setScene } = useAppState();
  const { flyTo, thumbs } = useShell();
  const [flying, setFlying] = useState(false);
  const mounted = useRef(true);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  const open = (id: string) => {
    if (flying) return;
    setScene(id);
    const flight = flyTo(id);
    if (!flight) { go(simPath(id)); return; }
    setFlying(true);
    // skip if the user left the picker mid-flight
    flight.then(() => { if (mounted.current && location.hash === "#/scenes") go(simPath(id)); });
  };
  return (
    <div className={`sv-page${flying ? " sv-flying" : ""}`}>
      <h2 className="sv-title">Choose a scene to explore</h2>
      <p className="sv-lead"><a href="#/observer">change observer</a></p>
      <div className="sv-grid">
        {sceneCards().map((c) => (
          <GlassCard key={c.id} disabled={c.disabled} data-testid={`scene-${c.id}`}>
            <div className="sv-thumb" aria-hidden="true" style={thumbs[c.id] ? { backgroundImage: `url(${thumbs[c.id]})` } : undefined} />
            <h3>{c.title}</h3>
            <p>{c.description}</p>
            <span className="sv-tags">{c.tags.map((t) => <span key={t} className="sv-tag">{t}</span>)}</span>
            {c.tasks.length > 0 && <ul>{c.tasks.map((t) => <li key={t}>{t}</li>)}</ul>}
            {c.disabled
              ? <button className="sv-btn" disabled>Coming soon</button>
              : <button className="sv-btn sv-btn-primary" onClick={() => open(c.id)}>Open</button>}
          </GlassCard>
        ))}
      </div>
    </div>
  );
}
