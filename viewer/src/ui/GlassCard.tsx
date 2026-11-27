// Glass card with pointer parallax and glow (CSS variables only, no re-render on move).
import type { HTMLAttributes, PointerEvent } from "react";
import { usePrefs } from "./prefs";

const TILT_DEG = 5;

export function GlassCard({ disabled, className = "", ...rest }: HTMLAttributes<HTMLDivElement> & { disabled?: boolean }) {
  const { still, performance } = usePrefs();
  const move = (e: PointerEvent<HTMLDivElement>) => {
    if (disabled || still || performance) return;
    const el = e.currentTarget, r = el.getBoundingClientRect();
    if (!r.width || !r.height) return;
    const x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
    el.style.setProperty("--mx", `${(x * 100).toFixed(1)}%`);
    el.style.setProperty("--my", `${(y * 100).toFixed(1)}%`);
    el.style.setProperty("--rx", `${((0.5 - y) * TILT_DEG).toFixed(2)}deg`);
    el.style.setProperty("--ry", `${((x - 0.5) * TILT_DEG).toFixed(2)}deg`);
  };
  const leave = (e: PointerEvent<HTMLDivElement>) => {
    for (const v of ["--rx", "--ry"]) e.currentTarget.style.removeProperty(v);
  };
  return <div {...rest} className={`sv-glass sv-card ${className}`} aria-disabled={disabled || undefined} onPointerMove={move} onPointerLeave={leave} />;
}
