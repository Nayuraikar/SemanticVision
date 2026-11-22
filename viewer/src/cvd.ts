// CVD uniform builder (mirrors core/perception/cvd.py on linear sRGB). Constants come from cvd.gen.ts.
import { CVD } from "./cvd.gen";

export type CvdModel = "machado" | "vienot" | "brettel";
export type CvdKind = "protan" | "deutan" | "tritan";
export const KIND_INDEX: Record<CvdKind, number> = { protan: 0, deutan: 1, tritan: 2 };

export type Mat9 = number[]; // row-major 3x3
export interface CvdUniforms {
  mode: 0 | 1;          // 0 = matrix path (Machado / Vienot), 1 = Brettel two-half-plane path
  kind: number;         // 0 protan, 1 deutan, 2 tritan
  severity: number;     // observer severity
  blend: number;        // effective mix(rgb, full, blend); Machado matrices already carry severity (blend = 1)
  M: Mat9;
  p1: Mat9; p2: Mat9; nsep: number[]; sign1: number;
  rgb2lms: Mat9; lms2rgb: Mat9;
}
export interface ObserverLike { model: string; kind: string; severity: number }

const flat = (m: readonly (readonly number[])[]) => m.flat();
const I9: Mat9 = [1, 0, 0, 0, 1, 0, 0, 0, 1];
const KIND_KEY: Record<CvdKind, "Protanomaly" | "Deuteranomaly" | "Tritanomaly"> = {
  protan: "Protanomaly", deutan: "Deuteranomaly", tritan: "Tritanomaly",
};
const MACH_KEY = { protan: "protan", deutan: "deutan", tritan: "tritan" } as const;

export function machadoMatrix(kind: CvdKind, severity: number): Mat9 {
  if (!(severity >= 0 && severity <= 1)) throw new Error("severity must be in [0,1]");
  const tab = CVD.machado[MACH_KEY[kind]] as unknown as number[][][];
  const x = severity * 10;
  const i = Math.min(Math.floor(x + 1e-12), 9);
  const f = x - i;
  const a = flat(tab[i]), b = flat(tab[i + 1]);
  return a.map((v, k) => (1 - f) * v + f * b[k]);
}

export function buildCvdUniforms(o: ObserverLike): CvdUniforms {
  const kind = o.kind as CvdKind;
  if (!(kind in KIND_INDEX)) throw new Error(`bad kind ${o.kind}`);
  const br = CVD.brettel[kind];
  const base = {
    kind: KIND_INDEX[kind], severity: o.severity, M: I9,
    p1: flat(br.p1), p2: flat(br.p2), nsep: [...br.nsep], sign1: br.sign1,
    rgb2lms: flat(CVD.rgb2lms), lms2rgb: flat(CVD.lms2rgb),
  };
  if (o.model === "machado") return { ...base, mode: 0, blend: 1, M: machadoMatrix(kind, o.severity) };
  if (o.model === "vienot") return { ...base, mode: 0, blend: o.severity, M: flat(CVD.vienot[kind]) };
  if (o.model === "brettel") return { ...base, mode: 1, blend: o.severity };
  throw new Error(`bad model ${o.model}`);
}

export const IDENTITY_CVD: CvdUniforms = { ...buildCvdUniforms({ model: "machado", kind: "protan", severity: 0 }) };

const mv = (m: Mat9, v: number[]) => [0, 1, 2].map((r) => m[3 * r] * v[0] + m[3 * r + 1] * v[1] + m[3 * r + 2] * v[2]);

/** CPU reference of the GLSL in shaders.ts (same maths), linear RGB in/out, clipped to [0,1]. */
export function simulateCpu(u: CvdUniforms, rgb: number[]): number[] {
  let full: number[];
  if (u.mode === 0) full = mv(u.M, rgb);
  else {
    const lms = mv(u.rgb2lms, rgb);
    const side = Math.sign(lms[0] * u.nsep[0] + lms[1] * u.nsep[1] + lms[2] * u.nsep[2]) === u.sign1;
    full = mv(u.lms2rgb, mv(side ? u.p1 : u.p2, lms));
  }
  // core clips the full-dichromat result before blending
  return rgb.map((v, k) => (1 - u.blend) * v + u.blend * Math.min(1, Math.max(0, full[k])));
}
