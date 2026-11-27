// Styled toggle switch. State is shown by thumb position and a tick, not by colour alone.
export function Switch({ label, checked, onChange, disabled, title }: {
  label: string; checked: boolean; onChange(v: boolean): void; disabled?: boolean; title?: string;
}) {
  return (
    <button type="button" role="switch" className="sv-switch" aria-checked={checked} aria-label={label}
      disabled={disabled} title={title} onClick={() => onChange(!checked)}>
      <span className="sv-switch-track" aria-hidden="true"><span className="sv-switch-thumb">{checked ? "✓" : ""}</span></span>
      <span>{label}</span>
    </button>
  );
}
