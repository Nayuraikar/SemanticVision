// Per-viewer UI preferences: theme, reduced motion (OS setting always wins when on), performance mode.
import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

export const PREFS_KEY = "sv.prefs";
export interface Prefs { theme: "dark" | "light"; reduceMotion: boolean; performance: boolean }
export const DEFAULT_PREFS: Prefs = { theme: "dark", reduceMotion: false, performance: false };

export function loadPrefs(): Prefs {
  try {
    const p = JSON.parse(localStorage.getItem(PREFS_KEY) || "{}");
    return { theme: p.theme === "light" ? "light" : "dark", reduceMotion: p.reduceMotion === true, performance: p.performance === true };
  } catch { return DEFAULT_PREFS; }
}

const QUERY = "(prefers-reduced-motion: reduce)";
const media = () => (typeof matchMedia === "function" ? matchMedia(QUERY) : null);

interface PrefsValue extends Prefs {
  /** effective: user toggle OR OS preference */
  still: boolean;
  set(p: Partial<Prefs>): void;
}

const Ctx = createContext<PrefsValue>({ ...DEFAULT_PREFS, still: false, set: () => {} });
export const usePrefs = () => useContext(Ctx);

export function PrefsProvider({ children }: { children: ReactNode }) {
  const [prefs, setPrefs] = useState<Prefs>(loadPrefs);
  const [osStill, setOsStill] = useState(() => media()?.matches ?? false);
  useEffect(() => {
    const m = media();
    if (!m?.addEventListener) return;
    const f = () => setOsStill(m.matches);
    m.addEventListener("change", f);
    return () => m.removeEventListener("change", f);
  }, []);
  const still = prefs.reduceMotion || osStill;
  useEffect(() => {
    const d = document.documentElement.dataset;
    d.theme = prefs.theme; d.motion = still ? "reduced" : "full"; d.perf = String(prefs.performance);
    try { localStorage.setItem(PREFS_KEY, JSON.stringify(prefs)); } catch { /* storage unavailable */ }
  }, [prefs, still]);
  const value = useMemo(() => ({ ...prefs, still, set: (p: Partial<Prefs>) => setPrefs((c) => ({ ...c, ...p })) }), [prefs, still]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}
