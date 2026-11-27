// Player actions and the pure object -> action mapping. Actions are viewer-side events; snapshots are unchanged.
import { SYSTEMS, type System, type DashLighting } from "../dashboardSim";
import type { EnvName } from "../environments";

export type Axis = "ew" | "ns";
export type PaletteName = "default" | "lookalike" | "swapped";
export const PALETTE_NAMES: PaletteName[] = ["default", "lookalike", "swapped"];
export type MissionName = "brake" | "signal" | "crash" | "lineup" | "fault" | "levels" | "gauge" | "separate" | "end";

export type Action =
  // movement (all scenes)
  | { type: "keys"; forward: number; right: number; turn: number; sprint: boolean }
  | { type: "look"; yaw: number; pitch: number }
  | { type: "zoom"; factor: number }
  | { type: "walk"; x: number; z: number }
  | { type: "fly"; id: string }
  // traffic
  | { type: "signal"; axis: Axis }
  | { type: "brake"; id: string }
  | { type: "collision" }
  | { type: "cross"; id: string }
  | { type: "env"; env: EnvName }
  // dashboard
  | { type: "fault"; system: System }
  | { type: "ack"; level: "warning" | "caution" }
  | { type: "gauge"; system: System; value: number | null }
  | { type: "lighting"; env: DashLighting }
  // toy
  | { type: "lamp" }
  | { type: "palette"; preset: PaletteName }
  | { type: "drag"; id: string; x: number; z: number }
  // seeded mission setup
  | { type: "mission"; name: MissionName };

/** Actions that change authored colours: the viewer must drop its colour overrides before they are applied. */
export const invalidatesOverrides = (a: Action) =>
  a.type === "env" || a.type === "lighting" || a.type === "palette" || a.type === "collision" || a.type === "fault" ||
  a.type === "mission";

export interface CardAction { label: string; action: Action }

/** Actions offered for a clicked object. The first one is also performed by the click itself (outside missions). */
export function actionsFor(sceneId: string, id: string): CardAction[] {
  if (sceneId === "traffic") {
    const sig = /^sig_([EWSN])_(red|amber|green)$/.exec(id);
    if (sig) {
      const axis: Axis = sig[1] === "E" || sig[1] === "W" ? "ew" : "ns";
      return [{ label: `Cycle ${axis === "ew" ? "east-west" : "north-south"} signal`, action: { type: "signal", axis } }];
    }
    const veh = /^(veh\d+)(_brake|_ind|_ind_r)?$/.exec(id);
    if (veh) return [{ label: "Make this car brake", action: { type: "brake", id: veh[1] } }];
    const ped = /^(ped\d+)(_head)?$/.exec(id);
    if (ped) return [{ label: "Tell pedestrian to cross", action: { type: "cross", id: ped[1] } }];
    return [];
  }
  if (sceneId === "dashboard") {
    const master = /^master_(warning|caution)$/.exec(id);
    if (master) {
      const level = master[1] as "warning" | "caution";
      return [{ label: `Acknowledge master ${level}`, action: { type: "ack", level } }];
    }
    const lamp = /^(?:ann|status)_([a-z]+)(?:_\d)?$/.exec(id);
    const system = SYSTEMS.find((s) => s === lamp?.[1]);
    return system ? [{ label: `Inject ${system.toUpperCase()} fault`, action: { type: "fault", system } }] : [];
  }
  if (id === "lamp") return [{ label: "Toggle signal lamp", action: { type: "lamp" } }];
  return [];
}
