// Seeded RNG (mulberry32) for deterministic scenarios. Never use Math.random inside the sim.
export class Rng {
  private s: number;
  constructor(seed: number) {
    this.s = (seed >>> 0) ^ 0x9e3779b9;
    for (let i = 0; i < 4; i++) this.next();
  }
  next(): number {
    let t = (this.s = (this.s + 0x6d2b79f5) >>> 0);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  uniform(a: number, b: number): number { return a + (b - a) * this.next(); }
  int(n: number): number { return Math.floor(this.next() * n); }
  pick<T>(xs: readonly T[]): T { return xs[this.int(xs.length)]; }
}
