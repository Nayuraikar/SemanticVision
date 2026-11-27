import type { Overrides } from "../../../../adapters/threejs/protocol.gen";
import type { StatsSnapshot } from "../../stats";
import type { Posterior, PosteriorRow } from "./models";

/** What the last full solve reported (stabiliser-only frames carry no tiers or latency). */
export interface SolveInfo { tier: string; tierCounts: Record<string, number>; latency: Record<string, number> }

export function solveInfo(ov: Overrides | null, prev: SolveInfo | null): SolveInfo | null {
  const d = ov?.diagnostics;
  if (!d) return prev;
  const latency = d.stage_latency_ms ?? {};
  if (!Object.keys(latency).length) return prev;
  return { tier: d.solver_tier ?? "not reported", tierCounts: d.tier_counts ?? {}, latency };
}

export const barPct = (v: number, max: number) => (max > 0 ? Math.min(100, Math.max(0, (100 * v) / max)) : 0);

function Bars({ rows, unit }: { rows: [string, number][]; unit: string }) {
  const max = Math.max(0, ...rows.map((r) => r[1]));
  return (
    <ul className="sv-bars">
      {rows.map(([k, v]) => (
        <li key={k}>
          <span>{k}</span>
          <span className="sv-bar-track" aria-hidden="true"><span style={{ width: `${barPct(v, max)}%` }} /></span>
          <span>{unit ? `${v.toFixed(1)} ${unit}` : v}</span>
        </li>
      ))}
    </ul>
  );
}

function Interval({ r }: { r: PosteriorRow }) {
  const range = r.lo != null && r.hi != null;
  const text = range ? `${r.mean.toFixed(2)} (${r.lo!.toFixed(2)}–${r.hi!.toFixed(2)})` : r.mean.toFixed(2);
  return (
    <li>
      <span>{r.label}</span><span>{text}</span>
      <span className="sv-interval" role="img" aria-label={`${r.label} ${text}`}>
        {range && <span className="sv-interval-range" style={{ left: `${barPct(r.lo!, r.max)}%`, width: `${barPct(r.hi! - r.lo!, r.max)}%` }} />}
        <span className="sv-interval-mean" style={{ left: `${barPct(r.mean, r.max)}%` }} />
      </span>
    </li>
  );
}

export function PosteriorCard({ p, error }: { p: Posterior | null; error: string }) {
  if (!p) return <p className="sv-note">No observer data returned for this observer.</p>;
  return (
    <>
      <dl className="sv-facts">
        <dt>type</dt><dd>{p.type}</dd>
        <dt>model</dt><dd>{p.model ?? "not reported"}</dd>
        {p.trials != null && <><dt>trials</dt><dd>{p.trials}</dd></>}
        {p.width != null && <><dt>interval width</dt><dd>{p.width.toFixed(2)}</dd></>}
      </dl>
      <ul className="sv-intervals">{p.rows.map((r) => <Interval key={r.label} r={r} />)}</ul>
      {p.probs.length > 0 && (
        <p className="sv-note">type probability: {p.probs.map(([k, v]) => `${k} ${(100 * v).toFixed(0)}%`).join(", ")}</p>
      )}
      {error && <p className="sv-note">{error}</p>}
      {p.notes.map((n) => <p key={n} className="sv-note">{n}</p>)}
    </>
  );
}

export function Diagnostics({ ov, solve, stats, status, observerId, posterior, posteriorError }: {
  ov: Overrides | null; solve: SolveInfo | null; stats: StatsSnapshot | null; status: string;
  observerId: string; posterior: Posterior | null; posteriorError: string;
}) {
  const f1 = (x: number | null | undefined) => (x == null ? "n/a" : x.toFixed(1));
  const pairs = ov?.confusion_pairs ?? [];
  const tiers = Object.entries(solve?.tierCounts ?? {});
  return (
    <>
      <h4>Solver</h4>
      <dl className="sv-facts">
        <dt>status</dt><dd>{status}</dd>
        <dt>tier (last solve)</dt><dd>{solve?.tier ?? "-"}</dd>
        <dt>flags</dt><dd>{ov?.diagnostics?.fallback_flags?.join(", ") || "none"}</dd>
        <dt>cues</dt><dd>{ov?.cues?.length ?? 0}</dd>
        <dt>pairs</dt><dd>{`${pairs.length} (${pairs.filter((p) => !p.feasible).length} still below tau)`}</dd>
      </dl>
      <h4>Tier counts</h4>
      {tiers.length ? <Bars rows={tiers} unit="" /> : <p className="sv-note">No tier counts reported yet.</p>}
      <h4>Stage latency</h4>
      {solve ? <Bars rows={Object.entries(solve.latency)} unit="ms" /> : <p className="sv-note">No latency reported yet.</p>}
      <h4>Rendering</h4>
      <dl className="sv-facts">
        <dt>fps</dt><dd>{f1(stats?.fps)}</dd>
        <dt>frame</dt><dd>{f1(stats?.frameMs)} ms</dd>
        <dt>GPU</dt><dd>{stats?.gpuMs == null ? "n/a (no timer query)" : `${stats.gpuMs.toFixed(2)} ms`}</dd>
        <dt>heap</dt><dd>{stats?.heapMB == null ? "n/a" : `${stats.heapMB.toFixed(1)} MB`}</dd>
        <dt>draw calls</dt><dd>{stats?.calls ?? "-"}</dd>
        <dt>triangles</dt><dd>{stats?.triangles ?? "-"}</dd>
        <dt>geometries / textures</dt><dd>{`${stats?.geometries ?? "-"} / ${stats?.textures ?? "-"}`}</dd>
      </dl>
      <h4>Observer posterior</h4>
      <p className="sv-note">selected: {observerId || "none"}</p>
      <PosteriorCard p={posterior} error={posteriorError} />
    </>
  );
}
