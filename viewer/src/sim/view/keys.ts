// View and panel shortcuts (playback keys stay in timeline.keyAction).
import type { ViewMode } from "./layout";

export type ViewKey =
  | { mode: ViewMode } | { toggle: "inspector" | "graph" | "diagnostics" | "help" } | { action: "close" | "reset" } | null;

export function viewKey(key: string, target: { tagName?: string; type?: string } | null): ViewKey {
  if (key === "Escape") return { action: "close" };
  const tag = target?.tagName ?? "", type = target?.type ?? "";
  // typing and select type-ahead keep their keys
  if (tag === "SELECT" || tag === "TEXTAREA" || (tag === "INPUT" && type !== "range" && type !== "checkbox")) return null;
  switch (key.toLowerCase()) {
    case "1": return { mode: "split" };
    case "2": return { mode: "wipe" };
    case "3": return { mode: "single" };
    case "0": return { action: "reset" };
    case "i": return { toggle: "inspector" };
    case "g": return { toggle: "graph" };
    case "d": return { toggle: "diagnostics" };
    case "?": return { toggle: "help" };
    default: return null;
  }
}
