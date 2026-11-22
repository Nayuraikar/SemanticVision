// Registry wrapper around the existing toy scene (behaviour unchanged).
import { buildToyScene, stepToyScene } from "../toyScene";

export const id = "toy";
export const title = "Toy scene";
export const description = "Car, bush, cone and a red/green signal lamp: the minimal confusable-colour scene.";
export const tasks = ["Tell the signal state (red/green)", "Spot the hazard car", "Separate cone from vegetation"];
export const tags = ["demo", "signals", "emissive"];

export function build(aspect = 16 / 9) {
  return buildToyScene(aspect);
}

export function step(scene: Parameters<typeof stepToyScene>[0], t: number) {
  stepToyScene(scene, t);
}
