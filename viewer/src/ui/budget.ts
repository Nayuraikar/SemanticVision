// WebGL budget helpers (8 GB target). The shell owns ONE renderer; the sim panels are the only other live context.

export const MAX_PIXEL_RATIO = 1.5;

export const cappedPixelRatio = (dpr: number | undefined, performance: boolean) =>
  performance ? 1 : Math.min(dpr || 1, MAX_PIXEL_RATIO);

/** Cheap check that creates no context; real context failure is caught where the renderer is constructed. */
export const hasWebGL = () => typeof WebGL2RenderingContext !== "undefined";

/** Calls cb(true|false) when "tab visible AND element on screen" changes. Returns an unsubscribe function. */
export function watchVisibility(el: Element, cb: (visible: boolean) => void): () => void {
  let onScreen = true;
  const emit = () => cb(onScreen && document.visibilityState !== "hidden");
  document.addEventListener("visibilitychange", emit);
  const io = typeof IntersectionObserver === "function"
    ? new IntersectionObserver((es) => { onScreen = es[es.length - 1].isIntersecting; emit(); })
    : null;
  io?.observe(el);
  return () => { document.removeEventListener("visibilitychange", emit); io?.disconnect(); };
}

export const easeInOutCubic = (x: number) => (x < 0.5 ? 4 * x * x * x : 1 - (-2 * x + 2) ** 3 / 2);
