// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AppStateProvider } from "../appState";
import { ScenePicker } from "../screens";
import { cappedPixelRatio, easeInOutCubic, watchVisibility } from "./budget";
import { GlassCard } from "./GlassCard";
import { DEFAULT_PREFS, PREFS_KEY, PrefsProvider, loadPrefs } from "./prefs";
import { EXIT_MS, ScreenTransition } from "./ScreenTransition";
import { Shell, backdropMode } from "./Shell";
import { ShellContext } from "./shellContext";
import { breadcrumb, menuPath } from "./TopBar";

let store: Map<string, string>;
beforeEach(() => {
  store = new Map();
  vi.stubGlobal("localStorage", {
    getItem: (k: string) => store.get(k) ?? null, setItem: (k: string, v: string) => void store.set(k, v), removeItem: (k: string) => void store.delete(k),
  });
  location.hash = "";
});
afterEach(() => { cleanup(); vi.useRealTimers(); vi.unstubAllGlobals(); });

const reducedMotionOs = () => vi.stubGlobal("matchMedia", () => ({ matches: true, addEventListener() {}, removeEventListener() {} }));

describe("budget", () => {
  it("caps the pixel ratio at 1.5, 1 in performance mode", () => {
    expect(cappedPixelRatio(3, false)).toBe(1.5);
    expect(cappedPixelRatio(1.25, false)).toBe(1.25);
    expect(cappedPixelRatio(undefined, false)).toBe(1);
    expect(cappedPixelRatio(2, true)).toBe(1);
  });
  it("easing starts at 0, ends at 1, is monotonic", () => {
    const xs = Array.from({ length: 11 }, (_, i) => easeInOutCubic(i / 10));
    expect([xs[0], xs[10]]).toEqual([0, 1]);
    expect(xs.every((x, i) => !i || x >= xs[i - 1])).toBe(true);
  });
  it("reports hidden tabs and unsubscribes", () => {
    const cb = vi.fn();
    const state = vi.spyOn(document, "visibilityState", "get").mockReturnValue("hidden");
    const off = watchVisibility(document.body, cb);
    document.dispatchEvent(new Event("visibilitychange"));
    expect(cb).toHaveBeenLastCalledWith(false);
    state.mockReturnValue("visible");
    document.dispatchEvent(new Event("visibilitychange"));
    expect(cb).toHaveBeenLastCalledWith(true);
    off();
    document.dispatchEvent(new Event("visibilitychange"));
    expect(cb).toHaveBeenCalledTimes(2);
    state.mockRestore();
  });
});

describe("prefs and shell", () => {
  it("falls back to defaults on bad storage", () => {
    store.set(PREFS_KEY, "{nope");
    expect(loadPrefs()).toEqual(DEFAULT_PREFS);
  });

  it("settings drawer sets theme, motion and performance, and persists them", () => {
    render(<PrefsProvider><Shell route={{ name: "scenes" }}>x</Shell></PrefsProvider>);
    expect(document.documentElement.dataset.theme).toBe("dark");
    const toggle = screen.getByText("Settings", { selector: "button" });
    fireEvent.click(toggle);
    expect(toggle.getAttribute("aria-expanded")).toBe("true");
    for (const name of ["Light theme", "Reduce motion", "Performance mode"]) fireEvent.click(screen.getByLabelText(name));
    expect(document.documentElement.dataset).toMatchObject({ theme: "light", motion: "reduced", perf: "true" });
    expect(loadPrefs()).toEqual({ theme: "light", reduceMotion: true, performance: true });
    fireEvent.keyDown(window, { key: "Escape" });
    expect(toggle.getAttribute("aria-expanded")).toBe("false");
  });

  it("uses the gradient fallback without WebGL", () => {
    render(<Shell route={null}>x</Shell>);
    expect(screen.getByTestId("backdrop").dataset.fallback).toBe("true");
  });

  it("maps routes to backdrop modes, breadcrumbs and the menu target", () => {
    expect(backdropMode(null)).toBe("hero");
    expect(backdropMode({ name: "scenes" })).toBe("picker");
    expect(backdropMode({ name: "calibrate" })).toBe("off");
    expect(breadcrumb({ name: "sim", sceneId: "toy" })).toEqual(["Menu", "Scenes", "Toy scene"]);
    expect(menuPath({ name: "sim", sceneId: "toy" })).toBe("#/scenes");
    expect(menuPath({ name: "scenes" })).toBe("#/welcome");
  });

  it("shows the back pill everywhere but welcome", () => {
    const { rerender } = render(<Shell route={{ name: "welcome" }}>x</Shell>);
    expect(screen.queryByText("Back to menu")).toBeNull();
    rerender(<Shell route={{ name: "sim", sceneId: "toy" }}>x</Shell>);
    fireEvent.click(screen.getByText("Back to menu"));
    expect(location.hash).toBe("#/scenes");
  });
});

describe("ScreenTransition", () => {
  const tree = (k: string) => <PrefsProvider><ScreenTransition screenKey={k}><p>{`screen ${k}`}</p></ScreenTransition></PrefsProvider>;
  it("keeps the old screen during its exit, then swaps", () => {
    vi.useFakeTimers();
    const { rerender } = render(tree("a"));
    rerender(tree("b"));
    expect(screen.getByText("screen a").parentElement!.className).toContain("sv-leave");
    expect(screen.queryByText("screen b")).toBeNull();
    act(() => { vi.advanceTimersByTime(EXIT_MS); });
    expect(screen.getByText("screen b").parentElement!.className).toContain("sv-enter");
    expect(screen.queryByText("screen a")).toBeNull();
  });
  it("swaps immediately when the OS asks for reduced motion", () => {
    reducedMotionOs();
    const { rerender } = render(tree("a"));
    rerender(tree("b"));
    expect(screen.getByText("screen b")).toBeTruthy();
    expect(screen.queryByText("screen a")).toBeNull();
  });
});

describe("GlassCard", () => {
  const hover = () => {
    const el = screen.getByText("card");
    el.getBoundingClientRect = () => ({ left: 0, top: 0, width: 100, height: 100 }) as DOMRect;
    const e = new MouseEvent("pointermove", { bubbles: true, clientX: 100, clientY: 0 });
    fireEvent(el, e);
    return el;
  };
  it("tilts and glows toward the pointer", () => {
    render(<GlassCard>card</GlassCard>);
    const el = hover();
    expect(el.style.getPropertyValue("--mx")).toBe("100.0%");
    expect(el.style.getPropertyValue("--ry")).toBe("2.50deg");
  });
  it("stays flat with reduced motion or when disabled", () => {
    reducedMotionOs();
    render(<PrefsProvider><GlassCard>card</GlassCard></PrefsProvider>);
    expect(hover().style.getPropertyValue("--ry")).toBe("");
    cleanup();
    vi.unstubAllGlobals();
    render(<GlassCard disabled>card</GlassCard>);
    expect(hover().getAttribute("aria-disabled")).toBe("true");
    expect(hover().style.getPropertyValue("--ry")).toBe("");
  });
});

describe("ScenePicker fly-in", () => {
  const picker = (flyTo: (id: string) => Promise<void> | null) => (
    <ShellContext.Provider value={{ barSlot: null, overlaySlot: null, flyTo, thumbs: { toy: "data:image/jpeg;base64,AA" } }}>
      <AppStateProvider><ScenePicker /></AppStateProvider>
    </ShellContext.Provider>
  );
  const openToy = () => fireEvent.click(screen.getByText("Toy scene").parentElement!.querySelector("button")!);

  it("navigates only after the camera arrives, and shows tasks and thumbnails", async () => {
    location.hash = "#/scenes";
    let arrive!: () => void;
    const flyTo = vi.fn(() => new Promise<void>((r) => { arrive = r; }));
    render(picker(flyTo));
    expect(screen.getByText("Spot the hazard car")).toBeTruthy();
    expect((screen.getByTestId("scene-toy").querySelector(".sv-thumb") as HTMLElement).style.backgroundImage).toContain("data:image/jpeg");
    openToy();
    openToy();
    expect(flyTo).toHaveBeenCalledTimes(1);
    expect(location.hash).toBe("#/scenes");
    await act(async () => { arrive(); });
    expect(location.hash).toBe("#/sim/toy");
  });

  it("does not navigate if the user left mid-flight", async () => {
    location.hash = "#/scenes";
    let arrive!: () => void;
    render(picker(() => new Promise<void>((r) => { arrive = r; })));
    openToy();
    location.hash = "#/welcome";
    await act(async () => { arrive(); });
    expect(location.hash).toBe("#/welcome");
  });

  it("navigates at once when no flight is available", () => {
    render(picker(() => null));
    openToy();
    expect(location.hash).toBe("#/sim/toy");
  });
});
