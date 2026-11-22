// Three-panel renderer: one scene state, A original | B CVD(original) | C CVD(corrected).
// Per frame: render original colours -> RT_orig (A,B), apply overrides -> RT_corr (C), composite via CVD shader.
import * as THREE from "three";
import { svMeshes, originalColors, type Overrides, type SVMesh } from "../../adapters/threejs";
import { FRAG, VERT } from "./shaders";
import { buildCvdUniforms, IDENTITY_CVD, type CvdUniforms, type ObserverLike } from "./cvd";

type Mode = "orig" | "corr";

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

  constructor(canvas: HTMLCanvasElement, readonly pw: number, readonly ph: number) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: false });
    this.renderer.setPixelRatio(1);
    this.renderer.setSize(pw * 3, ph, false);
    this.renderer.autoClear = true;
    this.renderer.info.autoReset = false; // 4 passes/frame: report whole-frame counts
    const mk = () => new THREE.WebGLRenderTarget(pw, ph, { type: THREE.HalfFloatType, depthBuffer: true });
    this.rtOrig = mk(); this.rtCorr = mk();
    for (const rt of [this.rtOrig, this.rtCorr]) rt.texture.colorSpace = THREE.LinearSRGBColorSpace;
    this.mat = new THREE.ShaderMaterial({
      vertexShader: VERT, fragmentShader: FRAG, depthTest: false, depthWrite: false,
      uniforms: {
        uTex: { value: null }, uMode: { value: 0 }, uKind: { value: 0 }, uSeverity: { value: 0 }, uBlend: { value: 0 },
        uM: { value: new THREE.Matrix3() }, uP1: { value: new THREE.Matrix3() }, uP2: { value: new THREE.Matrix3() },
        uRGB2LMS: { value: new THREE.Matrix3() }, uLMS2RGB: { value: new THREE.Matrix3() },
        uNsep: { value: new THREE.Vector3() }, uSign1: { value: 1 }, uSim: { value: true },
      },
    });
    this.quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.mat);
    this.quad.frustumCulled = false;
    this.qScene.add(this.quad);
    this.setObserver(null);
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

  render(scene: THREE.Scene, camera: THREE.PerspectiveCamera) {
    const r = this.renderer;
    r.info.reset();
    const prevTM = r.toneMapping;
    r.toneMapping = THREE.NoToneMapping;
    this.show(scene, "orig");
    r.setRenderTarget(this.rtOrig); r.clear(); r.render(scene, camera);
    this.show(scene, "corr");
    r.setRenderTarget(this.rtCorr); r.clear(); r.render(scene, camera);
    this.show(scene, "orig");
    r.setRenderTarget(null);
    r.setScissorTest(true);
    const panels: [THREE.WebGLRenderTarget, boolean][] = [[this.rtOrig, false], [this.rtOrig, true], [this.rtCorr, true]];
    panels.forEach(([rt, sim], i) => {
      r.setViewport(i * this.pw, 0, this.pw, this.ph);
      r.setScissor(i * this.pw, 0, this.pw, this.ph);
      this.mat.uniforms.uTex.value = rt.texture;
      this.mat.uniforms.uSim.value = sim;
      r.render(this.qScene, this.qCam);
    });
    r.setScissorTest(false);
    r.toneMapping = prevTM;
  }

  dispose() {
    this.rtOrig.dispose(); this.rtCorr.dispose(); this.mat.dispose(); this.quad.geometry.dispose(); this.renderer.dispose();
  }
}

type Nine = [number, number, number, number, number, number, number, number, number];
