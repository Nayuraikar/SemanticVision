// View renderer: one scene state, A original | B CVD(original) | C CVD(corrected), laid out as tiles.
// Per frame: original colours -> RT_orig (A,B), overrides -> RT_corr (C), composite each tile via the CVD shader.
import * as THREE from "three";
import { svMeshes, originalColors, type Overrides, type SVMesh } from "../../adapters/threejs";
import { FRAG, VERT } from "./shaders";
import { Cinematic } from "./sim/cinematic";
import { buildCvdUniforms, IDENTITY_CVD, type CvdUniforms, type ObserverLike } from "./cvd";
import { isVisible, type Rect, type Tile } from "./sim/view/layout";

type Mode = "orig" | "corr";

/** Display-only extras: neither reaches the scene render targets. */
export interface PanelFx { overlay: THREE.Scene | null; cinematic: boolean }

export class ThreePanels {
  readonly renderer: THREE.WebGLRenderer;
  private rtOrig: THREE.WebGLRenderTarget;
  private rtCorr: THREE.WebGLRenderTarget;
  private quad: THREE.Mesh;
  private mat: THREE.ShaderMaterial;
  private qScene = new THREE.Scene();
  private qCam = new THREE.Camera();
  private pal = new Map<string, [number, number, number]>();
  private cvd: CvdUniforms = IDENTITY_CVD;
  private post: Cinematic | null = null;
  private w = 1;
  private h = 1;
  private ratio = 1;

  constructor(canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: false });
    this.renderer.autoClear = true;
    this.renderer.info.autoReset = false; // several passes per frame: report whole-frame counts
    const mk = () => new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, depthBuffer: true, samples: 2 });
    this.rtOrig = mk(); this.rtCorr = mk();
    for (const rt of [this.rtOrig, this.rtCorr]) rt.texture.colorSpace = THREE.LinearSRGBColorSpace;
    this.mat = new THREE.ShaderMaterial({
      vertexShader: VERT, fragmentShader: FRAG, depthTest: false, depthWrite: false,
      uniforms: {
        uTex: { value: null }, uMode: { value: 0 }, uKind: { value: 0 }, uSeverity: { value: 0 }, uBlend: { value: 0 },
        uM: { value: new THREE.Matrix3() }, uP1: { value: new THREE.Matrix3() }, uP2: { value: new THREE.Matrix3() },
        uRGB2LMS: { value: new THREE.Matrix3() }, uLMS2RGB: { value: new THREE.Matrix3() },
        uNsep: { value: new THREE.Vector3() }, uSign1: { value: 1 }, uSim: { value: 0 },
      },
    });
    this.quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.mat);
    this.quad.frustumCulled = false;
    this.qScene.add(this.quad);
    this.setObserver(null);
  }

  /** Canvas size in CSS px; the caller passes a pixel ratio already capped for the WebGL budget. */
  resize(w: number, h: number, ratio: number) {
    this.w = w; this.h = h; this.ratio = ratio;
    this.renderer.setPixelRatio(ratio);
    this.renderer.setSize(w, h, false);
    this.post?.dispose(); this.post = null; // rebuilt at the new size on the next cinematic frame
  }

  setObserver(o: ObserverLike | null) {
    this.cvd = o ? buildCvdUniforms(o) : IDENTITY_CVD;
    const u = this.mat.uniforms, c = this.cvd;
    u.uMode.value = c.mode; u.uKind.value = c.kind; u.uSeverity.value = c.severity; u.uBlend.value = c.blend;
    (u.uM.value as THREE.Matrix3).set(...(c.M as Nine));
    (u.uP1.value as THREE.Matrix3).set(...(c.p1 as Nine));
    (u.uP2.value as THREE.Matrix3).set(...(c.p2 as Nine));
    (u.uRGB2LMS.value as THREE.Matrix3).set(...(c.rgb2lms as Nine));
    (u.uLMS2RGB.value as THREE.Matrix3).set(...(c.lms2rgb as Nine));
    (u.uNsep.value as THREE.Vector3).fromArray(c.nsep);
    u.uSign1.value = c.sign1;
  }

  /** New overrides from the server: build cue children once (applyOverrides), then only toggle per frame. */
  setOverrides(scene: THREE.Scene, ov: Overrides, apply: (s: THREE.Object3D, o: Overrides) => void) {
    apply(scene, ov);
    this.pal = new Map((ov.palette_overrides ?? []).map((p) => [p.id, p.albedo as [number, number, number]]));
    this.show(scene, "orig");
  }

  /** Cheap colour/cue-visibility switch between authored and corrected state. */
  show(scene: THREE.Object3D, mode: Mode) {
    for (const m of svMeshes(scene) as SVMesh[]) {
      const orig = originalColors(m);
      const a = mode === "corr" ? this.pal.get(m.userData.sv.id as string) ?? orig.albedo : orig.albedo;
      m.material.color.setRGB(a[0], a[1], a[2]);
      m.material.emissiveIntensity = 1;
      m.material.emissive.setRGB(orig.emissive[0], orig.emissive[1], orig.emissive[2]);
      for (const c of m.children) if (c.userData.svCue) c.visible = mode === "corr";
    }
  }

  /** `targetW`: width the shared scene image is rendered at (CSS px), the settled tile width. */
  render(scene: THREE.Scene, camera: THREE.PerspectiveCamera, tiles: Tile[], targetW: number, fx?: PanelFx) {
    const r = this.renderer;
    r.info.reset();
    const prevTM = r.toneMapping;
    r.toneMapping = THREE.NoToneMapping;
    const rw = Math.max(1, Math.round(targetW * this.ratio)), rh = Math.max(1, Math.round(this.h * this.ratio));
    for (const rt of [this.rtOrig, this.rtCorr]) if (rt.width !== rw || rt.height !== rh) rt.setSize(rw, rh);
    const shown = tiles.filter(isVisible);
    this.show(scene, "orig");
    if (shown.some((t) => t.pane !== "C")) { r.setRenderTarget(this.rtOrig); r.clear(); r.render(scene, camera); }
    if (shown.some((t) => t.pane === "C")) {
      this.show(scene, "corr");
      r.setRenderTarget(this.rtCorr); r.clear(); r.render(scene, camera);
      this.show(scene, "orig");
    }
    if (fx?.cinematic) this.post ??= new Cinematic(r, this.w, this.h, this.ratio);
    else if (this.post) { this.post.dispose(); this.post = null; }
    const out = this.post?.input ?? null;
    // viewport units: CSS px on the canvas (three applies the ratio), device px on a render target
    const k = out ? this.ratio : 1;
    const place = (view: Rect, clip: Rect) => {
      const v = [view.x * k, (this.h - view.y - view.h) * k, view.w * k, view.h * k] as const;
      const c = [clip.x * k, (this.h - clip.y - clip.h) * k, clip.w * k, clip.h * k] as const;
      if (out) { out.viewport.set(...v); out.scissor.set(...c); out.scissorTest = true; r.setRenderTarget(out); }
      else { r.setViewport(...v); r.setScissor(...c); }
    };
    if (out) { out.scissorTest = false; }
    r.setRenderTarget(out);
    r.setScissorTest(false);
    r.clear();
    if (!out) r.setScissorTest(true);
    r.autoClear = false;
    for (const t of shown) {
      place(t.view, t.clip);
      this.mat.uniforms.uTex.value = (t.pane === "C" ? this.rtCorr : this.rtOrig).texture;
      this.mat.uniforms.uSim.value = t.sim;
      r.render(this.qScene, this.qCam);
      if (fx?.overlay) r.render(fx.overlay, camera);
    }
    r.autoClear = true;
    r.setScissorTest(false);
    if (this.post) {
      this.post.setTiles(shown.length > 1 && shown[0].view.w < this.w * 0.5 ? 3 : 1);
      r.setRenderTarget(null);
      r.setViewport(0, 0, this.w, this.h);
      this.post.render();
    }
    r.toneMapping = prevTM;
  }

  dispose() {
    this.post?.dispose(); this.post = null;
    this.rtOrig.dispose(); this.rtCorr.dispose(); this.mat.dispose(); this.quad.geometry.dispose(); this.renderer.dispose();
  }
}

type Nine = [number, number, number, number, number, number, number, number, number];
