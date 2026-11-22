import { describe, expect, it } from "vitest";
import fs from "node:fs";
import path from "node:path";
import { buildCvdUniforms, machadoMatrix, simulateCpu, type CvdKind } from "./cvd";

const fx = JSON.parse(fs.readFileSync(path.resolve(__dirname, "../../tests/fixtures/cvd_expected.json"), "utf8")) as {
  colors: number[][]; cases: { model: string; kind: string; severity: number; out: number[][] }[];
};

describe("CVD uniform builder vs core/perception", () => {
  it("matches core simulate() for every model/kind/severity", () => {
    expect(fx.cases.length).toBeGreaterThan(20);
    for (const c of fx.cases) {
      const u = buildCvdUniforms(c);
      fx.colors.forEach((rgb, i) => {
        const got = simulateCpu(u, rgb);
        got.forEach((v, k) => expect(Math.abs(v - c.out[i][k]), `${c.model}/${c.kind}/${c.severity}`).toBeLessThan(1e-5));
      });
    }
  });
  it("uses Brettel path for brettel observers (incl. tritan) and the matrix path otherwise", () => {
    expect(buildCvdUniforms({ model: "brettel", kind: "tritan", severity: 0.3 }).mode).toBe(1);
    expect(buildCvdUniforms({ model: "machado", kind: "deutan", severity: 0.6 }).mode).toBe(0);
  });
  it("interpolates Machado between published 0.1 steps and rejects bad severity", () => {
    const a = machadoMatrix("protan" as CvdKind, 0.5), b = machadoMatrix("protan", 0.6), m = machadoMatrix("protan", 0.55);
    m.forEach((v, k) => expect(v).toBeCloseTo((a[k] + b[k]) / 2, 9));
    expect(() => machadoMatrix("protan", 1.2)).toThrow();
    expect(() => buildCvdUniforms({ model: "x", kind: "protan", severity: 1 })).toThrow();
  });
  it("severity 0 is identity", () => {
    const u = buildCvdUniforms({ model: "machado", kind: "tritan", severity: 0 });
    expect(simulateCpu(u, [0.3, 0.6, 0.1]).map((v) => +v.toFixed(6))).toEqual([0.3, 0.6, 0.1]);
  });
});
