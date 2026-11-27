// Keyboard / pointer helpers for play mode, plus the optional alarm beep.
import { ARC_SPAN, ARC_START } from "../../../adapters/threejs/dashboardSim";
import type { Keys } from "../../../adapters/threejs/play/movement";

const CODES = {
  forward: ["w", "arrowup"], back: ["s", "arrowdown"], left: ["a", "arrowleft"], right: ["d", "arrowright"],
  turnL: ["q"], turnR: ["e"],
};
const ALL = [...Object.values(CODES).flat(), "shift"];

/** Lower-cased key if play movement owns it; form controls keep their own keys. */
export function moveKey(key: string, target: { tagName?: string; type?: string; role?: string | null } | null): string | null {
  const tag = target?.tagName ?? "";
  if (tag === "SELECT" || tag === "TEXTAREA" || tag === "INPUT" || target?.role === "slider" || target?.role === "radio") return null;
  const k = key.toLowerCase();
  return ALL.includes(k) ? k : null;
}

export function keysFrom(held: ReadonlySet<string>): Keys {
  const on = (xs: string[]) => (xs.some((k) => held.has(k)) ? 1 : 0);
  return {
    forward: on(CODES.forward) - on(CODES.back), right: on(CODES.right) - on(CODES.left),
    turn: on(CODES.turnR) - on(CODES.turnL), sprint: held.has("shift"),
  };
}

/** Gauge value for a pointer at (dx, dy) screen pixels from the dial centre (y down). */
export function gaugeValueAt(dx: number, dy: number): number {
  let a = Math.atan2(-dy, dx);
  if (a < -Math.PI / 2) a += 2 * Math.PI;
  return Math.min(1, Math.max(0, (ARC_START - a) / ARC_SPAN));
}

/** Short UI beep through WebAudio; no assets. The context is created on the first unmuted beep. */
export class Beeper {
  muted = true;
  private ctx: AudioContext | null = null;
  setMuted(m: boolean) { this.muted = m; if (!m) this.beep(); }
  beep() {
    if (this.muted || typeof AudioContext === "undefined") return;
    const ctx = (this.ctx ??= new AudioContext());
    const osc = ctx.createOscillator(), gain = ctx.createGain();
    osc.frequency.value = 880;
    gain.gain.setValueAtTime(0.08, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.18);
    osc.connect(gain).connect(ctx.destination);
    osc.start(); osc.stop(ctx.currentTime + 0.2);
  }
  dispose() { void this.ctx?.close(); this.ctx = null; }
}
