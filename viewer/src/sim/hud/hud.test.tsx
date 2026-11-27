// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { DEFAULT_FLAGS } from "../../overlays";
import { Switch } from "../../ui/Switch";
import { DEFAULT_VIEW } from "../view/layout";
import { Diagnostics } from "./Diagnostics";
import { Drawer } from "./Drawer";
import { CUES, Help } from "./Help";
import { Inspector } from "./Inspector";
import { OverlayDrawer } from "./OverlayDrawer";
import { ModeBar, ModeOptions, PaneLabels, WipeDivider } from "./ViewControls";

afterEach(cleanup);

describe("controls", () => {
  it("switch exposes state to assistive tech and toggles", () => {
    const on = vi.fn();
    render(<Switch label="dust" checked={false} onChange={on} />);
    const s = screen.getByRole("switch", { name: "dust" });
    expect(s.getAttribute("aria-checked")).toBe("false");
    fireEvent.click(s);
    expect(on).toHaveBeenCalledWith(true);
  });
  it("overlay drawer is collapsed by default and shows N only with the graph on", () => {
    const open = vi.fn(), change = vi.fn();
    const props = { onOpen: open, onChange: change, topN: 8, onTopN: () => {}, flags: DEFAULT_FLAGS };
    const { rerender } = render(<OverlayDrawer open={false} {...props} />);
    expect(screen.queryByRole("switch")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Overlays" }));
    expect(open).toHaveBeenCalledWith(true);
    rerender(<OverlayDrawer open {...props} />);
    expect(screen.getAllByRole("switch").map((s) => s.getAttribute("aria-checked"))).toEqual(["false", "false", "false", "false"]);
    expect(screen.queryByLabelText("graph pairs")).toBeNull();
    rerender(<OverlayDrawer open {...props} flags={{ ...DEFAULT_FLAGS, graph: true }} />);
    expect(screen.getByLabelText("graph pairs")).toBeTruthy();
  });
  it("mode pill and pane labels", () => {
    const on = vi.fn();
    render(<><ModeBar mode="split" onMode={on} /><PaneLabels view={DEFAULT_VIEW} /></>);
    fireEvent.click(screen.getByRole("radio", { name: "Wipe" }));
    expect(on).toHaveBeenCalledWith("wipe");
    expect(screen.getByText("B · How you see it")).toBeTruthy();
  });
  it("wipe sides never match: picking the other side's pane swaps them", () => {
    const on = vi.fn();
    render(<ModeOptions view={{ ...DEFAULT_VIEW, mode: "wipe" }} onChange={on} onFeel={() => {}} feeling={false} />);
    fireEvent.change(screen.getByLabelText("left side"), { target: { value: "C" } });
    expect(on).toHaveBeenCalledWith({ left: "C", right: "B" });
  });
  it("divider moves with the arrow keys and clamps", () => {
    const on = vi.fn();
    render(<div><WipeDivider value={0.99} onChange={on} /></div>);
    fireEvent.keyDown(screen.getByRole("slider"), { key: "ArrowRight" });
    expect(on).toHaveBeenCalledWith(1);
  });
});

describe("panels", () => {
  it("closed drawers render no content", () => {
    render(<Drawer id="x" title="Diagnostics" open={false} onClose={() => {}}><p>inside</p></Drawer>);
    expect(screen.queryByText("inside")).toBeNull();
  });
  it("help lists all four cue types and renders nothing when closed", () => {
    const close = vi.fn();
    const { rerender, container } = render(<Help open={false} onClose={close} />);
    expect(container.textContent).toBe("");
    rerender(<Help open onClose={close} />);
    expect(CUES.map((c) => c[0])).toEqual(["outline", "hatch", "icon", "label"]);
    expect(container.querySelectorAll(".sv-cues svg")).toHaveLength(4);
    fireEvent.click(screen.getByText("Close"));
    expect(close).toHaveBeenCalled();
  });
  it("inspector shows swatches, partners and dE00, and selects partners", () => {
    const select = vi.fn();
    render(<Inspector selected="car" ids={["bush", "car"]} cvd={null} onSelect={select} data={{
      id: "car", cls: "vehicle", state: "moving", litBefore: [1, 0, 0], litAfter: [0, 0, 1], tier: "allocated", cues: ["outline"],
      partners: [{ id: "bush", before: 2, after: 3, feasible: false }], worst: { before: 2, after: 3 },
    }} />);
    expect(screen.getByRole("img", { name: "before: rgb(255, 0, 0)" })).toBeTruthy();
    expect(screen.getByRole("img", { name: "after: rgb(0, 0, 255)" })).toBeTruthy();
    expect(screen.getByText("2.0 → 3.0")).toBeTruthy();
    expect(screen.getByText("✗ still below tau")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "bush" }));
    expect(select).toHaveBeenCalledWith("bush");
  });
  it("diagnostics render tier counts and latency bars, and admit missing data", () => {
    const props = { ov: null, stats: null, status: "idle", observerId: "t0", posterior: null, posteriorError: "" };
    const { rerender, container } = render(<Diagnostics {...props} solve={null} />);
    expect(screen.getByText("No tier counts reported yet.")).toBeTruthy();
    expect(screen.getByText("No observer data returned for this observer.")).toBeTruthy();
    rerender(<Diagnostics {...props} solve={{ tier: "allocated", tierCounts: { exact: 3 }, latency: { solving: 10, verify: 5 } }} />);
    expect([...container.querySelectorAll(".sv-bar-track > span")].map((s) => (s as HTMLElement).style.width))
      .toEqual(["100%", "100%", "50%"]);
    expect(container.textContent).not.toMatch(/placeholder/i);
  });
});
