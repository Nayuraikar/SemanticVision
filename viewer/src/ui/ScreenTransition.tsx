// Fade + slight scale/blur between screens: the outgoing screen stays until its exit finishes.
import { useEffect, useRef, useState, type ReactNode } from "react";
import { usePrefs } from "./prefs";

export const EXIT_MS = 200; // matches --sv-t-med

export function ScreenTransition({ screenKey, children }: { screenKey: string; children: ReactNode }) {
  const { still } = usePrefs();
  const [shown, setShown] = useState(screenKey);
  const leaving = !still && shown !== screenKey;
  const last = useRef<{ key: string; node: ReactNode }>({ key: screenKey, node: children });
  if (!leaving) last.current = { key: screenKey, node: children };
  useEffect(() => {
    if (shown === screenKey) return;
    if (still) { setShown(screenKey); return; }
    const h = setTimeout(() => setShown(screenKey), EXIT_MS);
    return () => clearTimeout(h);
  }, [shown, screenKey, still]);
  return (
    <div key={last.current.key} data-screen={last.current.key} className={`sv-screen ${leaving ? "sv-leave" : "sv-enter"}`}>
      {last.current.node}
    </div>
  );
}
