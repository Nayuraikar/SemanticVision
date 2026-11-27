import type { Overrides, Snapshot } from "./protocol.gen";

export * from "./protocol.gen";
export * from "./snapshot";
export * from "./lighting";
export * from "./overrides";
export * from "./gt";
export * from "./toyScene";

export async function postCorrect(baseUrl: string, snap: Snapshot): Promise<Overrides> {
  const r = await fetch(`${baseUrl}/correct`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(snap),
  });
  if (!r.ok) throw new Error(`POST /correct ${r.status}`);
  return (await r.json()) as Overrides;
}
export * from "./scenes/registry";
export * from "./physics";
export * from "./rng";
export * from "./environments";
export * from "./trafficSim";
export * from "./dashboardSim";
