import { createContext, useContext } from "react";

export interface ShellValue {
  /** top-bar and settings-drawer mount points that screens portal their controls into */
  barSlot: HTMLElement | null;
  overlaySlot: HTMLElement | null;
  /** camera flight into a scene; null when unavailable (no WebGL / reduced motion) so callers navigate at once */
  flyTo(sceneId: string): Promise<void> | null;
  thumbs: Record<string, string>;
}

export const ShellContext = createContext<ShellValue>({ barSlot: null, overlaySlot: null, flyTo: () => null, thumbs: {} });
export const useShell = () => useContext(ShellContext);
