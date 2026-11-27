// Persistent top bar (breadcrumb, screen controls, back pill) and the settings drawer.
import { useEffect, useState } from "react";
import { getScene } from "../../../adapters/threejs/scenes/registry";
import { go, type Route } from "../appState";
import { usePrefs } from "./prefs";
import { Switch } from "./Switch";

export function breadcrumb(route: Route | null): string[] {
  if (!route) return ["Menu"];
  if ("sceneId" in route) return ["Menu", "Scenes", getScene(route.sceneId)?.title ?? route.sceneId];
  const trail = { welcome: [], observer: ["Observer"], calibrate: ["Observer", "Calibration"], scenes: ["Scenes"] };
  return ["Menu", ...trail[route.name]];
}

/** "Menu" is the scene picker from inside a scene, otherwise the welcome screen. */
export const menuPath = (route: Route | null) => (route?.name === "sim" ? "#/scenes" : "#/welcome");

interface Props {
  route: Route | null;
  barSlot: (el: HTMLElement | null) => void;
}

export function TopBar({ route, barSlot }: Props) {
  const prefs = usePrefs();
  const [open, setOpen] = useState(false);
  useEffect(() => {
    if (!open) return;
    const esc = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    addEventListener("keydown", esc);
    return () => removeEventListener("keydown", esc);
  }, [open]);
  const crumbs = breadcrumb(route);
  return (
    <>
      <header className="sv-bar sv-glass">
        <nav aria-label="breadcrumb">
          <ol className="sv-crumbs">
            {crumbs.map((c, i) => <li key={c} aria-current={i === crumbs.length - 1 ? "page" : undefined}>{c}</li>)}
          </ol>
        </nav>
        <div className="sv-slot" ref={barSlot} />
        {route && route.name !== "welcome" && (
          <button className="sv-pill" onClick={() => go(menuPath(route))}>Back to menu</button>
        )}
        <button className="sv-pill" aria-expanded={open} aria-controls="sv-drawer" onClick={() => setOpen(!open)}>Settings</button>
      </header>
      <aside id="sv-drawer" className="sv-drawer sv-glass" data-open={open} aria-label="settings">
        <h2>Settings</h2>
        <h4>Appearance</h4>
        <Switch label="Light theme" checked={prefs.theme === "light"} onChange={(v) => prefs.set({ theme: v ? "light" : "dark" })} />
        <Switch label="Reduce motion" checked={prefs.reduceMotion} onChange={(v) => prefs.set({ reduceMotion: v })} />
        <Switch label="Performance mode" checked={prefs.performance} onChange={(v) => prefs.set({ performance: v })} />
        <p><button className="sv-pill" onClick={() => setOpen(false)}>Close</button></p>
      </aside>
    </>
  );
}
