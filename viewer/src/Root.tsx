import { useEffect, useState } from "react";
import { App } from "./App";
import { Calibration } from "./Calibration";
import { AppStateProvider, PROFILE_PREFIX, go, loadSavedObserver, parseHash, saveObserver, useAppState, useHash, type Route } from "./appState";
import { ObserverStep, ScenePicker, Welcome, fetchObservers, fetchProfiles } from "./screens";
import { PrefsProvider } from "./ui/prefs";
import { ScreenTransition } from "./ui/ScreenTransition";
import { Shell } from "./ui/Shell";

async function savedStillExists(id: string): Promise<boolean> {
  try {
    if (id.startsWith(PROFILE_PREFIX)) return (await fetchProfiles()).some((p) => p.observer_id === id);
    return (await fetchObservers()).some((o) => o.id === id);
  } catch { return false; }
}

function Router() {
  const hash = useHash();
  const { observerId, setObserver, setScene } = useAppState();
  const route = parseHash(hash);
  // entry points (no hash / #/welcome) resolve the saved observer once, then skip ahead or show the welcome page
  const [booting, setBooting] = useState(() => !route || route.name === "welcome");

  useEffect(() => {
    if (!booting) return;
    let live = true;
    const saved = loadSavedObserver();
    (saved ? savedStillExists(saved) : Promise.resolve(false)).then((ok) => {
      if (!live) return;
      if (ok) { setObserver(saved); go("#/scenes"); }
      else { if (saved) { setObserver(null); saveObserver(null); } go("#/welcome"); }
      setBooting(false);
    });
    return () => { live = false; };
  }, [booting, setObserver]);

  const needsObserver = route && (route.name === "scenes" || route.name === "sim") && !observerId;
  // read location (not the possibly stale hash state) so a navigation issued this tick is not overridden
  useEffect(() => {
    if (booting) return;
    const cur = parseHash(location.hash);
    if (!cur || ((cur.name === "scenes" || cur.name === "sim") && !observerId)) go("#/welcome");
  }, [booting, route, needsObserver, hash, observerId]);
  useEffect(() => { if (route?.name === "sim") setScene(route.sceneId); }, [route && route.name === "sim" ? route.sceneId : null, setScene]);

  const ready = !booting && !!route && !needsObserver;
  const screenFor = (r: Route) => {
    switch (r.name) {
      case "welcome": return <Welcome />;
      case "observer": return <ObserverStep />;
      case "calibrate":
        return <Calibration onBack={() => go("#/observer")} onFinish={(pid) => { setObserver(`${PROFILE_PREFIX}${pid}`); go("#/scenes"); }} />;
      case "scenes": return <ScenePicker />;
      case "sim": return <App sceneId={r.sceneId} />;
    }
  };
  // the shell (and its live background) stays mounted across every route
  return (
    <Shell route={ready ? route : null}>
      {ready && <ScreenTransition screenKey={route.name}>{screenFor(route)}</ScreenTransition>}
    </Shell>
  );
}

export function Root() {
  return <PrefsProvider><AppStateProvider><Router /></AppStateProvider></PrefsProvider>;
}
