import { useState } from "react";
import {
  API, AXIS_LABEL, TYPE_LABEL, bgCss, cellText, certainty, cssColor, fmt, mapGrid, patchPx, range, sliderLabel,
  type Progress, type Stimulus, type Summary,
} from "./calibLib";

const COLOURS = ["red", "green", "brown", "orange", "pink", "purple", "yellow", "turquoise"];
const NOTE = "These are accessibility parameters estimated from your answers. They are not a medical diagnosis.";
type Phase = "intro" | "trial" | "summary";

async function call(path: string, init?: RequestInit) {
  const r = await fetch(`${API}${path}`, { headers: { "content-type": "application/json" }, ...init });
  if (!r.ok) throw new Error(`${path}: ${r.status} ${await r.text()}`);
  return r.json();
}

export function Calibration({ onFinish, onBack }: { onFinish?: (profileId: string) => void; onBack?: () => void } = {}) {
  const [phase, setPhase] = useState<Phase>("intro");
  const [guess, setGuess] = useState("");
  const [struggle, setStruggle] = useState<string[]>([]);
  const [pid, setPid] = useState("");
  const [stim, setStim] = useState<Stimulus | null>(null);
  const [prog, setProg] = useState<Progress | null>(null);
  const [slider, setSlider] = useState(0);
  const [profile, setProfile] = useState<{ summary: Summary } | null>(null);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);

  const guard = async (f: () => Promise<void>) => {
    setBusy(true); setErr("");
    try { await f(); } catch (e) { setErr(String(e)); } finally { setBusy(false); }
  };

  const finish = (id: string) => guard(async () => {
    setProfile(await call(`/profiles/${id}/finish`, { method: "POST" }));
    setPhase("summary");
  });

  const next = (id: string) => guard(async () => {
    const j = await call(`/profiles/${id}/next`);
    setProg(j.progress);
    if (j.done) { await finish(id); return; }
    setStim(j.stimulus); setSlider(0);
  });

  const start = () => guard(async () => {
    const self_report = guess || struggle.length ? { type_guess: guess || null, struggle_colors: struggle } : null;
    const j = await call("/profiles", { method: "POST", body: JSON.stringify({ self_report }) });
    setPid(j.id); setPhase("trial");
    await next(j.id);
  });

  const answer = (response: string | number) => guard(async () => {
    if (!stim) return;
    const j = await call(`/profiles/${pid}/answer`, { method: "POST", body: JSON.stringify({ trial_id: stim.trial_id, response }) });
    setProg(j.progress);
    await next(pid);
  });

  const page: React.CSSProperties = { maxWidth: 760, margin: "0 auto", padding: 20, font: "14px system-ui, sans-serif", lineHeight: 1.5 };
  const btn: React.CSSProperties = { padding: "8px 18px", fontSize: 14, marginRight: 8, cursor: "pointer" };

  return (
    <div style={page}>
      <h2 style={{ marginTop: 0 }}>Colour discrimination calibration</h2>
      <p style={{ opacity: 0.8 }}>{NOTE} <a href="#/observer" style={{ color: "#8cf" }} onClick={(e) => { if (onBack) { e.preventDefault(); onBack(); } }}>back</a></p>
      {err && <p style={{ color: "#f88" }}>{err}</p>}

      {phase === "intro" && (
        <div>
          <p>Optional: tell us what you already know. This only sets the starting guess; your answers decide the result.</p>
          <label>Type, if you have been told one:{" "}
            <select value={guess} onChange={(e) => setGuess(e.target.value)}>
              <option value="">no answer</option><option value="protan">protan (red-weak)</option>
              <option value="deutan">deutan (green-weak)</option><option value="tritan">tritan (blue-weak)</option>
              <option value="unsure">not sure</option>
            </select>
          </label>
          <p>Colours you often struggle to tell apart:</p>
          <div>{COLOURS.map((c) => (
            <label key={c} style={{ marginRight: 14 }}>
              <input type="checkbox" checked={struggle.includes(c)}
                onChange={(e) => setStruggle(e.target.checked ? [...struggle, c] : struggle.filter((x) => x !== c))} /> {c}
            </label>))}
          </div>
          <p>Use a normal-brightness screen, avoid glare and keep colour filters (night mode) off.</p>
          <button style={btn} disabled={busy} onClick={start}>Start</button>
        </div>
      )}

      {phase === "trial" && stim && prog && (
        <div>
          <progress value={prog.n} max={prog.max_trials} style={{ width: "100%" }} />
          <p>Trial {prog.n + 1} of at most {prog.max_trials} · uncertainty {fmt(prog.width)} (stops at {fmt(prog.target_width)}, minimum {prog.min_trials} trials)</p>
          <div style={{ background: bgCss(stim.bg_lum), padding: 40, display: "flex", justifyContent: "center", gap: 6 }} data-testid="stimulus">
            {(stim.swap ? [stim.c2, stim.c1] : [stim.c1, stim.c2]).map((c, i) => (
              <div key={i} style={{ width: patchPx(stim.size_deg), height: patchPx(stim.size_deg), background: cssColor(c) }} />))}
          </div>
          {stim.mode === "same_different" ? (
            <p>Do the two patches look the same colour?{" "}
              <button style={btn} disabled={busy} onClick={() => answer("same")}>Same</button>
              <button style={btn} disabled={busy} onClick={() => answer("different")}>Different</button></p>
          ) : (
            <p>How different do they look?<br />
              <input type="range" min={0} max={1} step={0.25} value={slider} style={{ width: 320 }}
                onChange={(e) => setSlider(Number(e.target.value))} /> {sliderLabel(slider)}<br />
              <button style={btn} disabled={busy} onClick={() => answer(slider)}>Submit</button></p>
          )}
          <button style={btn} disabled={busy || prog.n === 0} onClick={() => finish(pid)}>Finish now</button>
        </div>
      )}

      {phase === "summary" && profile && <SummaryView s={profile.summary} pid={pid} />}
      {phase === "summary" && profile && onFinish && <p><button style={btn} onClick={() => onFinish(pid)}>Use this profile and choose a scene</button></p>}
    </div>
  );
}

function SummaryView({ s, pid }: { s: Summary; pid: string }) {
  const p = s.params;
  const g = s.discrimination_map ? mapGrid(s.discrimination_map) : null;
  const cell: React.CSSProperties = { border: "1px solid #444", padding: "4px 8px", textAlign: "left" };
  return (
    <div>
      <h3>Your profile</h3>
      <p>{NOTE} Profile id <code>profile:{pid}</code> can be used as the observer in the viewer. Based on {s.n_trials} trials; confidence {certainty(s.width)} (uncertainty {fmt(s.width)}).</p>
      <h4>Colour-vision type and strength</h4>
      <ul>
        {(["protan", "deutan", "tritan"] as const).map((k, i) => (
          <li key={k}>{TYPE_LABEL[k]}: strength {range({ ...p[`e_${k}`] })}{s.dominant === k ? " ← most likely" : ""} · chance this is the main pattern {fmt(100 * (s.type_probs[k] ?? 0), 0)}%</li>))}
        <li>no meaningful difference from typical: {fmt(100 * (s.type_probs.normal ?? 0), 0)}%</li>
      </ul>
      <h4>Sensitivity</h4>
      <ul>
        <li>overall threshold scale (1 = typical, higher = needs bigger differences): {range(p.k)}</li>
        <li>lightness sensitivity {range(p.s_L)} · chroma sensitivity {range(p.s_C)}</li>
        <li>background dependence {range(p.bg_gain)}</li>
      </ul>
      {g && (<>
        <h4>Discrimination map</h4>
        <p>Smallest change you can reliably tell apart (relative modulation, median and 90% range) for a 2° patch, by direction and brightness.</p>
        <table style={{ borderCollapse: "collapse" }}>
          <thead><tr><th style={cell}>direction \ brightness</th>{g.levels.map((y) => <th key={y} style={cell}>{fmt(y * 100, 0)}%</th>)}</tr></thead>
          <tbody>{g.rows.map((r) => (
            <tr key={r.axis}><th style={cell}>{AXIS_LABEL[r.axis] ?? r.axis}</th>{r.cells.map((c) => <td key={c.Y} style={cell}>{cellText(c)}</td>)}</tr>))}
          </tbody>
        </table>
      </>)}
    </div>
  );
}
