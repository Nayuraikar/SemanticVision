// Headless recorder CLI (no WebGL). Run from viewer/:
//   npx vite-node ../adapters/threejs/record.ts -- --scenario collision --env day --seed 1 --steps 600 --every 2 [--out F]
// Writes snapshot NDJSON (default bench/logs/traffic/<scenario>_<env>_s<seed>.ndjson) and prints a JSON summary.
import fs from "node:fs";
import path from "node:path";
import { recordTraffic, type Scenario } from "./trafficSim";
import type { EnvName } from "./environments";

function arg(name: string, def: string): string {
  const a = process.argv;
  const i = a.indexOf(`--${name}`);
  return i >= 0 && i + 1 < a.length ? a[i + 1] : def;
}

async function main() {
  const scenario = arg("scenario", "collision") as Scenario;
  const env = arg("env", "day") as EnvName;
  const seed = Number(arg("seed", "1"));
  const steps = Number(arg("steps", "600"));
  const every = Number(arg("every", "2"));
  const root = path.resolve(__dirname, "../..");
  const out = path.resolve(arg("out", path.join(root, "bench/logs/traffic", `${scenario}_${env}_s${seed}.ndjson`)));
  fs.mkdirSync(path.dirname(out), { recursive: true });
  const fd = fs.openSync(out, "w");
  let n = 0;
  const { sim } = await recordTraffic({ seed, scenario, env, steps, exportEvery: every, budgetMs: Number(arg("budget", "40")) },
    (s) => { fs.writeSync(fd, JSON.stringify(s) + "\n"); n++; });
  fs.closeSync(fd);
  const col = sim.log.filter((e) => e.kind === "collision");
  console.log(JSON.stringify({ out, frames: n, collisions: col.map((e) => [e.a, e.b, e.step]), triggers: sim.log.length - col.length }));
}

main().catch((e) => { console.error(e); process.exit(1); });
