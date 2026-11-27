// Step-indexed player input queue. Inputs are applied at the start of their fixed step, in push order,
// so a recorded log replays deterministically. No wall-clock time or Math.random.
import type { Action } from "./actions";

export interface Input { step: number; seq: number; action: Action }

export class InputQueue {
  private items: Input[];
  private cursor = 0;
  private seq: number;
  constructor(readonly seed: number, log: readonly Input[] = []) {
    this.items = log.map((i) => ({ ...i, action: { ...i.action } })).sort((a, b) => a.step - b.step || a.seq - b.seq);
    this.seq = this.items.reduce((m, i) => Math.max(m, i.seq + 1), 0);
  }

  /** Inputs must not be older than anything already queued or applied. */
  push(step: number, action: Action): void {
    const last = this.items[this.items.length - 1];
    if (last && step < last.step) throw new Error(`input for step ${step} is older than queued step ${last.step}`);
    this.items.push({ step, seq: this.seq++, action: { ...action } });
  }

  /** Not yet applied inputs due at or before `step`. */
  pending(step: number): Input[] {
    let k = this.cursor;
    while (k < this.items.length && this.items[k].step <= step) k++;
    return this.items.slice(this.cursor, k);
  }

  take(step: number): Input[] {
    const out = this.pending(step);
    this.cursor += out.length;
    return out;
  }

  log(): Input[] { return this.items.map((i) => ({ ...i, action: { ...i.action } })); }
}
