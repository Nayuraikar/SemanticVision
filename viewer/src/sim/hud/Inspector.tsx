import { cssColor } from "../../calibLib";
import { simulateCpu, type CvdUniforms } from "../../cvd";
import type { Inspection, Rgb } from "./models";

function Swatch({ label, rgb }: { label: string; rgb: Rgb | number[] | null }) {
  const css = rgb ? cssColor(rgb as number[]) : null;
  return (
    <figure className="sv-swatch">
      <span style={css ? { background: css } : undefined} role="img" aria-label={`${label}: ${css ?? "not reported"}`} />
      <figcaption>{label}<small>{css ?? "not reported"}</small></figcaption>
    </figure>
  );
}

export function Inspector({ data, selected, ids, cvd, onSelect }: {
  data: Inspection | null; selected: string | null; ids: string[]; cvd: CvdUniforms | null; onSelect(id: string | null): void;
}) {
  const seen = (c: Rgb | null) => (c && cvd ? simulateCpu(cvd, c) : null);
  return (
    <>
      <label className="sv-pill sv-wide"><span>object</span>
        <select aria-label="inspect object" value={selected ?? ""} onChange={(e) => onSelect(e.target.value || null)}>
          <option value="">none</option>
          {selected && !ids.includes(selected) && <option value={selected}>{selected}</option>}
          {ids.map((id) => <option key={id} value={id}>{id}</option>)}
        </select>
      </label>
      {!selected && <p className="sv-note">Click an object in the view, or pick one here.</p>}
      {selected && !data && <p className="sv-note">This object is not in view in the latest snapshot.</p>}
      {data && (
        <>
          <dl className="sv-facts">
            <dt>class</dt><dd>{data.cls}</dd>
            <dt>state</dt><dd>{data.state}</dd>
            <dt>cue type</dt><dd>{data.cues.join(", ") || "none"}</dd>
            <dt>solve tier</dt><dd>{data.tier}</dd>
          </dl>
          <p className="sv-note">The tier is for the whole solve; the API does not report it per object.</p>
          <h4>Lit colour</h4>
          <div className="sv-swatches">
            <Swatch label="before" rgb={data.litBefore} />
            <Swatch label="after" rgb={data.litAfter} />
            <Swatch label="before, as you see it" rgb={seen(data.litBefore)} />
            <Swatch label="after, as you see it" rgb={seen(data.litAfter)} />
          </div>
          <h4>Lowest ΔE00 to a confusing partner</h4>
          <p className="sv-big">{data.worst ? `${data.worst.before.toFixed(1)} → ${data.worst.after.toFixed(1)}` : "no confusable pair reported"}</p>
          {data.partners.length > 0 && (
            <table className="sv-table">
              <thead><tr><th>partner</th><th>before</th><th>after</th><th>result</th></tr></thead>
              <tbody>
                {data.partners.map((p) => (
                  <tr key={p.id}>
                    <td><button className="sv-link" onClick={() => onSelect(p.id)}>{p.id}</button></td>
                    <td>{p.before.toFixed(1)}</td><td>{p.after.toFixed(1)}</td>
                    <td>{p.feasible ? "✓ separated" : "✗ still below tau"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </>
      )}
    </>
  );
}
