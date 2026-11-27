// Viewer shell: the live 3D background stays mounted; every screen is layered over it.
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import type { Route } from "../appState";
import type { BackdropEngine } from "./backdropEngine";
import { hasWebGL, watchVisibility } from "./budget";
import { usePrefs } from "./prefs";
import { ShellContext } from "./shellContext";
import { TopBar } from "./TopBar";

export type BackdropMode = "hero" | "menu" | "picker" | "sim" | "off";

export function backdropMode(route: Route | null): BackdropMode {
  switch (route?.name) {
    case "observer": return "menu";
    case "scenes": return "picker";
    case "sim": return "sim";
    case "calibrate": return "off";
    default: return "hero";
  }
}

export function Shell({ route, children }: { route: Route | null; children: ReactNode }) {
  const { still, performance } = usePrefs();
  const mode = backdropMode(route);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [engine, setEngine] = useState<BackdropEngine | null>(null);
  const [fallback, setFallback] = useState(() => !hasWebGL());
  const [visible, setVisible] = useState(true);
  const [flying, setFlying] = useState(false);
  const [thumbs, setThumbs] = useState<Record<string, string>>({});
  const [barSlot, setBarSlot] = useState<HTMLElement | null>(null);

  useEffect(() => {
    if (!hasWebGL() || !canvas.current) return;
    const el = canvas.current;
    let live = true, made: BackdropEngine | null = null;
    // lazy: environments without WebGL never load the engine
    import("./backdropEngine").then(({ BackdropEngine }) => {
      if (!live) return;
      made = new BackdropEngine(el);
      setEngine(made);
      made.thumbnails().then((t) => { if (live) setThumbs(t); }).catch(() => {});
    }).catch(() => { if (live) setFallback(true); });
    const lost = () => setFallback(true);
    const resize = () => made?.resize();
    el.addEventListener("webglcontextlost", lost);
    addEventListener("resize", resize);
    const unwatch = watchVisibility(el, setVisible);
    return () => {
      live = false; unwatch(); removeEventListener("resize", resize); el.removeEventListener("webglcontextlost", lost);
      made?.dispose(); setEngine(null);
    };
  }, []);

  // paused behind the simulation (its panels are the live view) and during calibration
  const active = visible && mode !== "sim" && mode !== "off";
  useEffect(() => { engine?.configure({ active, still, performance }); }, [engine, active, still, performance]);
  useEffect(() => {
    if (mode === "sim") return;
    setFlying(false);
    engine?.home();
  }, [engine, mode]);

  const flyTo = useCallback((sceneId: string) => {
    if (!engine || fallback || still) return null;
    setFlying(true);
    return engine.flyTo(sceneId);
  }, [engine, fallback, still]);

  const value = useMemo(() => ({ barSlot, flyTo, thumbs }), [barSlot, flyTo, thumbs]);
  return (
    <ShellContext.Provider value={value}>
      <div className="sv-backdrop" data-testid="backdrop" aria-hidden="true"
        data-mode={flying && mode === "picker" ? "hero" : mode} data-fallback={fallback}>
        <canvas ref={canvas} />
      </div>
      <TopBar route={route} barSlot={setBarSlot} />
      <main className="sv-main">{children}</main>
    </ShellContext.Provider>
  );
}
