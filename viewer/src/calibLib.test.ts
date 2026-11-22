import { describe, expect, it } from "vitest";
import { cellText, certainty, cssColor, linearToSrgb, mapGrid, patchPx, sliderLabel, type MapCell } from "./calibLib";

describe("calibration helpers", () => {
  it("encodes linear RGB to sRGB (known points, clamped)", () => {
    expect(linearToSrgb(0)).toBe(0);
    expect(linearToSrgb(1)).toBeCloseTo(1, 6);
    expect(linearToSrgb(0.2140411)).toBeCloseTo(0.5, 3);
    expect(linearToSrgb(-1)).toBe(0);
    expect(cssColor([1, 0, 0])).toBe("rgb(255, 0, 0)");
    expect(cssColor([0.2140411, 2, -1])).toMatch(/^rgb\(12[78], 255, 0\)$/);
  });
  it("sizes patches and labels slider", () => {
    expect(patchPx(2)).toBe(80);
    expect(sliderLabel(0)).toBe("identical");
    expect(sliderLabel(1)).toBe("very different");
  });
  it("builds the map grid and text", () => {
    const cells: MapCell[] = ["protan", "lightness"].flatMap((axis) =>
      [0.4, 0.08].map((Y) => ({ axis, Y, delta50: axis === "protan" ? null : 0.1, lo: 0.05, hi: null, resolvable: 1 })));
    const g = mapGrid(cells);
    expect(g.levels).toEqual([0.08, 0.4]);
    expect(g.rows.map((r) => r.axis)).toEqual(["protan", "lightness"]);
    expect(g.rows[1].cells[0].Y).toBe(0.08);
    expect(cellText(cells[0])).toBe("not resolvable");
    expect(cellText(cells[2])).toBe("10% (5–∞)");
  });
  it("maps width to wording", () => {
    expect([certainty(0.2), certainty(0.3), certainty(0.9)]).toEqual(["high", "moderate", "low"]);
  });
});
