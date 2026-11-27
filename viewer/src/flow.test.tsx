// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("./panels", () => ({
  ThreePanels: class {
    renderer = {};
    setObserver() {}
    setOverrides() {}
    render() {}
    dispose() {}
  },
}));
vi.mock("./stats", () => ({
  Stats: class {
    begin() {}
    end() {}
    snapshot() { return { fps: 0, frameMs: 0, gpuMs: null, heapMB: 0, calls: 0, triangles: 0, geometries: 0, textures: 0 }; }
  },
}));
vi.mock("./overlays", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./overlays")>()),
  DEFAULT_FLAGS: { graph: true, heatmap: true, bboxes: true, labels: true },
  drawAll: () => {},
}));

import { Root } from "./Root";
import { OBS_KEY } from "./appState";
import * as dashboardScene from "../../adapters/threejs/scenes/dashboard";

const stat = (m: number) => ({ mean: m, lo: m - 0.1, hi: m + 0.1 });
const SUMMARY = {
  n_trials: 12, width: 0.3, dominant: "deutan", effective_severity: [0, 0.8, 0],
  params: { e_protan: stat(0.1), e_deutan: stat(0.8), e_tritan: stat(0), k: stat(1), s_L: stat(1), s_C: stat(1), bg_gain: stat(0) },
  type_probs: { protan: 0.1, deutan: 0.8, tritan: 0.05, normal: 0.05 },
};
const PRESETS = {
  train: [{ id: "t0", split: "train", model: "machado", kind: "deutan", severity: 0.8 }],
  heldout: [{ id: "h0", split: "heldout", model: "brettel", kind: "protan", severity: 0.3 }],
};
const profileRow = (id: string) => ({ id, observer_id: `profile:${id}`, status: "complete", n_trials: 12, kind: "deutan", severity: 0.8, model: "machado" });

let profiles: ReturnType<typeof profileRow>[];
let calls: { url: string; method: string; body: any }[];

const json = (b: unknown, status = 200) => ({ ok: status < 400, status, json: async () => b, text: async () => JSON.stringify(b) });

// Node's experimental global localStorage shadows jsdom's; use an in-memory Storage.
function memStorage(): Storage {
  const m = new Map<string, string>();
  return {
    get length() { return m.size; }, clear: () => m.clear(), key: (i) => [...m.keys()][i] ?? null,
    getItem: (k) => m.get(k) ?? null, setItem: (k, v) => void m.set(k, String(v)), removeItem: (k) => void m.delete(k),
  };
}

beforeEach(() => {
  profiles = []; calls = [];
  vi.stubGlobal("localStorage", memStorage());
  location.hash = "";
  vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => setTimeout(() => cb(performance.now()), 5) as unknown as number);
  vi.stubGlobal("cancelAnimationFrame", (h: number) => clearTimeout(h));
  vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
    const path = url.replace("http://localhost:8000", "");
    const method = init?.method ?? "GET";
    calls.push({ url: path, method, body: init?.body ? JSON.parse(init.body as string) : null });
    if (path === "/observers") return json(PRESETS);
    if (path === "/profiles" && method === "GET") return json(profiles);
    if (path === "/profiles" && method === "POST") return json({ id: "abc123", progress: {} }, 201);
    if (path === "/profiles/abc123/next") return json({ done: true, progress: { n: 12, max_trials: 50, min_trials: 12, width: 0.3, target_width: 0.4, done: true } });
    if (path === "/profiles/abc123/finish") { profiles.push(profileRow("abc123")); return json({ summary: SUMMARY }); }
    if (path === "/correct") return json({});
    return json({}, 404);
  }));
});
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

const correctIds = () => calls.filter((c) => c.url === "/correct").map((c) => c.body.observer_id);

async function toSim() {
  await screen.findByText("Choose a scene to explore");
  for (const t of ["Cockpit", "ICU", "Construction site", "Rail / airport"]) {
    const card = screen.getByText(t).closest("div")!;
    expect(card.querySelector("button")!.disabled).toBe(true);
    expect(card.textContent).toContain("Coming soon");
  }
  const traffic = screen.getByText("Traffic intersection").closest("div")!;
  expect(traffic.querySelector("button")!.disabled).toBe(false);
  fireEvent.click(screen.getByText("Toy scene").closest("div")!.querySelector("button")!);
  await waitFor(() => expect(location.hash).toBe("#/sim/toy"));
}

describe("app flow", () => {
  it("welcome -> preset -> scenes -> sim", async () => {
    render(<Root />);
    await screen.findByText(/not a diagnosis/);
    expect(location.hash).toBe("#/welcome");
    fireEvent.click(screen.getByText("Continue"));
    fireEvent.click(await screen.findByText(/preset t0/));
    await toSim();
    await waitFor(() => expect(correctIds().length).toBeGreaterThan(0));
    expect(new Set(correctIds())).toEqual(new Set(["t0"]));
    expect(localStorage.getItem(OBS_KEY)).toBe("t0");
    await screen.findByLabelText("scene");
  });

  it("calibration finish stores profile id and sim uses profile:<id>", async () => {
    render(<Root />);
    await screen.findByText(/not a diagnosis/);
    fireEvent.click(screen.getByText("Continue"));
    fireEvent.click(await screen.findByText("Run calibration"));
    expect(location.hash).toBe("#/calibrate");
    fireEvent.click(await screen.findByText("Start"));
    fireEvent.click(await screen.findByText(/Use this profile/));
    await toSim();
    expect(localStorage.getItem(OBS_KEY)).toBe("profile:abc123");
    await waitFor(() => expect(correctIds().length).toBeGreaterThan(0));
    expect(new Set(correctIds())).toEqual(new Set(["profile:abc123"]));
  });

  it("switches observer in the sim without reloading", async () => {
    localStorage.setItem(OBS_KEY, "t0");
    render(<Root />);
    await toSim();
    await waitFor(() => expect(correctIds()).toContain("t0"));
    fireEvent.change(await screen.findByLabelText("observer"), { target: { value: "h0" } });
    await waitFor(() => expect(correctIds()).toContain("h0"));
    expect(localStorage.getItem(OBS_KEY)).toBe("h0");
  });

  it("dashboard sim has a normal/dim/glare lighting selector", async () => {
    localStorage.setItem(OBS_KEY, "t0");
    render(<Root />);
    await toSim();
    expect(screen.queryByLabelText("lighting")).toBeNull();
    fireEvent.change(await screen.findByLabelText("scene"), { target: { value: "dashboard" } });
    const dial = await screen.findByRole("radiogroup", { name: "lighting" });
    const radios = [...dial.querySelectorAll('[role="radio"]')] as HTMLElement[];
    expect(radios.map((r) => r.textContent)).toEqual(["normal", "dim", "glare"]);
    expect(screen.queryByRole("radiogroup", { name: "environment" })).toBeNull();
    fireEvent.click(radios[2]);
    expect(radios[2].getAttribute("aria-checked")).toBe("true");
    expect(dashboardScene.ENV).toBe("glare");
    await waitFor(() => expect(calls.some((c) => c.url === "/correct" &&
      c.body.objects.some((o: any) => o.id === "master_warning"))).toBe(true));
    dashboardScene.setEnv("normal");
  });

  it("saved profile that still exists skips to scenes", async () => {
    profiles.push(profileRow("abc123"));
    localStorage.setItem(OBS_KEY, "profile:abc123");
    render(<Root />);
    await screen.findByText("Choose a scene to explore");
    expect(location.hash).toBe("#/scenes");
  });

  it("saved observer that no longer exists goes to welcome", async () => {
    localStorage.setItem(OBS_KEY, "profile:gone");
    render(<Root />);
    await screen.findByText(/not a diagnosis/);
    expect(location.hash).toBe("#/welcome");
    expect(localStorage.getItem(OBS_KEY)).toBeNull();
  });
});
