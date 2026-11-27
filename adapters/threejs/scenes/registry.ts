import * as toy from "./toy";
import * as traffic from "./traffic";
import type { PlannedScene, SceneCard, SceneDef } from "./types";

export type { PlannedScene, SceneCard, SceneDef } from "./types";

export const SCENES: SceneDef[] = [toy, traffic];

const planned = (id: string, title: string, description: string, tags: string[]): PlannedScene => ({
  id, title, description, tasks: [], tags,
});

export const PLANNED: PlannedScene[] = [
  planned("dashboard", "Dashboard", "Status dashboard with colour-coded gauges and alerts.", ["ui", "status"]),
  planned("cockpit", "Cockpit", "Flight deck annunciators and warning lights.", ["aviation", "annunciator"]),
  planned("icu", "ICU", "Bedside monitors and alarm indicators.", ["medical", "alarms"]),
  planned("construction", "Construction site", "Hazard markings, machinery and safety gear.", ["industrial", "hazard"]),
  planned("rail-airport", "Rail / airport", "Signals, platform and apron lighting.", ["transport", "signals"]),
];

export const getScene = (id: string): SceneDef | undefined => SCENES.find((s) => s.id === id);

export function sceneCards(): SceneCard[] {
  const live = SCENES.map(({ id, title, description, tasks, tags }) => ({ id, title, description, tasks, tags, disabled: false }));
  return [...live, ...PLANNED.map((p) => ({ ...p, disabled: true }))];
}
