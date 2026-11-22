import type * as THREE from "three";

export interface SceneDef {
  id: string;
  title: string;
  description: string;
  build(aspect?: number): { scene: THREE.Scene; camera: THREE.PerspectiveCamera };
  step(scene: THREE.Scene, t: number): void;
  tasks: string[];
  tags: string[];
}

export type PlannedScene = Pick<SceneDef, "id" | "title" | "description" | "tasks" | "tags">;

export interface SceneCard extends PlannedScene {
  disabled: boolean;
}
