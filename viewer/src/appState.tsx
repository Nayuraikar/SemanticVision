import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

export const OBS_KEY = "sv.observer";
export const PROFILE_PREFIX = "profile:";

export function loadSavedObserver(): string | null {
  try { return localStorage.getItem(OBS_KEY) || null; } catch { return null; }
}
export function saveObserver(id: string | null) {
  try { id ? localStorage.setItem(OBS_KEY, id) : localStorage.removeItem(OBS_KEY); } catch { /* storage unavailable */ }
}

export type Route =
  | { name: "welcome" | "observer" | "calibrate" | "scenes" }
  | { name: "sim"; sceneId: string };

export function parseHash(h: string): Route | null {
  const p = h.replace(/^#\/?/, "").split("/").filter(Boolean);
  if (p.length === 1 && ["welcome", "observer", "calibrate", "scenes"].includes(p[0])) return { name: p[0] as "welcome" };
  if (p.length === 2 && p[0] === "sim") return { name: "sim", sceneId: decodeURIComponent(p[1]) };
  return null;
}

export const go = (path: string) => { location.hash = path; };
export const simPath = (sceneId: string) => `#/sim/${encodeURIComponent(sceneId)}`;

export function useHash(): string {
  const [hash, setHash] = useState(location.hash);
  useEffect(() => {
    const f = () => setHash(location.hash);
    addEventListener("hashchange", f);
    return () => removeEventListener("hashchange", f);
  }, []);
  return hash;
}

export interface AppState {
  observerId: string | null; // preset id or "profile:<id>"
  sceneId: string | null;
  setObserver(id: string | null): void;
  setScene(id: string | null): void;
}

const Ctx = createContext<AppState | null>(null);

export function AppStateProvider({ children }: { children: ReactNode }) {
  const [observerId, setObs] = useState<string | null>(loadSavedObserver);
  const [sceneId, setSceneId] = useState<string | null>(null);
  const setObserver = useCallback((id: string | null) => { setObs(id); saveObserver(id); }, []);
  const v = useMemo(() => ({ observerId, sceneId, setObserver, setScene: setSceneId }), [observerId, sceneId, setObserver]);
  return <Ctx.Provider value={v}>{children}</Ctx.Provider>;
}

export function useAppState(): AppState {
  const c = useContext(Ctx);
  if (!c) throw new Error("AppStateProvider missing");
  return c;
}
