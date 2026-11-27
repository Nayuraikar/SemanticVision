// Headless recorder CLI (no WebGL). Run from viewer/:
//   npx vite-node ../adapters/threejs/record.ts -- --scenario collision --env day --seed 1 --steps 600 --every 2 [--out F]
// --scene dashboard: --scenario alarm|quiet, --env normal|dim|glare.
// Writes snapshot NDJSON (default bench/logs/<scene>/<scenario>_<env>_s<seed>.ndjson) and prints a JSON summary.
import fs from "node:fs";
import path from "node:path";
import { recordTraffic, type Scenario } from "./trafficSim";
import { recordDashboard, type DashLighting, type DashScenario } from "./dashboardSim";
import type { EnvName } from "./environments";

function arg(name: string, def: string): string {
  const a = process.argv;
  const i = a.indexOf(`--${name}`);
  return i >= 0 && i + 1 < a.length ? a[i + 1] : def;
}

async function main() {
  const scene = arg("scene", "traffic");
  const dash = scene === "dashboard";
  const scenario = arg("scenario", dash ? "alarm" : "collision");
  const env = arg("env", dash ? "normal" : "day");
  const seed = Number(arg("seed", "1"));
  const steps = Number(arg("steps", "600"));
  const every = Number(arg("every", "2"));
  const root = path.resolve(__dirname, "../..");
  const out = path.resolve(arg("out", path.join(root, "bench/logs", scene, `${scenario}_${env}_s${seed}.ndjson`)));
  fs.mkdirSync(path.dirname(out), { recursive: true });
  const fd = fs.openSync(out, "w");
  let n = 0;
  const common = { seed, steps, exportEvery: every, budgetMs: Number(arg("budget", "40")) };
  const sink = (s: unknown) => { fs.writeSync(fd, JSON.stringify(s) + "\n"); n++; };
  if (dash) {
    const { sim } = await recordDashboard({ ...common, scenario: scenario as DashScenario, env: env as DashLighting }, sink);
    fs.closeSync(fd);
    console.log(JSON.stringify({ out, frames: n, cascade: sim.cascade }));
    return;
  }
  const { sim } = await recordTraffic({ ...common, scenario: scenario as Scenario, env: env as EnvName }, sink);
  fs.closeSync(fd);
  const col = sim.log.filter((e) => e.kind === "collision");
  console.log(JSON.stringify({ out, frames: n, collisions: col.map((e) => [e.a, e.b, e.step]), triggers: sim.log.length - col.length }));
}

main().catch((e) => { console.error(e); process.exit(1); });
