// Bloom + vignette + warm grade over the finished (sRGB) three-panel composite. Built only while enabled.
import * as THREE from "three";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { TexturePass } from "three/examples/jsm/postprocessing/TexturePass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { ShaderPass } from "three/examples/jsm/postprocessing/ShaderPass.js";

const GRADE = {
  uniforms: { tDiffuse: { value: null as THREE.Texture | null }, uTiles: { value: 3 } },
  vertexShader: /* glsl */ `
varying vec2 vUv;
void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: /* glsl */ `
uniform sampler2D tDiffuse;
uniform float uTiles;
varying vec2 vUv;
void main() {
  vec3 c = texture2D(tDiffuse, vUv).rgb * vec3(1.06, 1.0, 0.9);
  vec2 p = vec2(fract(vUv.x * uTiles), vUv.y) - 0.5;   // one vignette per tile
  c *= 1.0 - 0.7 * smoothstep(0.15, 0.55, dot(p, p));
  gl_FragColor = vec4(c, 1.0);
}`,
};

export class Cinematic {
  readonly input: THREE.WebGLRenderTarget;
  private composer: EffectComposer;
  private passes: { dispose(): void }[];

  private grade: ShaderPass;

  /** w, h in CSS px; `ratio` is the renderer's pixel ratio. */
  constructor(renderer: THREE.WebGLRenderer, w: number, h: number, ratio = 1) {
    this.input = new THREE.WebGLRenderTarget(Math.round(w * ratio), Math.round(h * ratio), { depthBuffer: false });
    this.composer = new EffectComposer(renderer);
    const source = new TexturePass(this.input.texture);
    const bloom = new UnrealBloomPass(new THREE.Vector2(w, h), 0.55, 0.4, 0.78);
    this.grade = new ShaderPass(GRADE);
    this.passes = [source, bloom, this.grade];
    for (const p of [source, bloom, this.grade]) this.composer.addPass(p);
  }

  setTiles(n: number) { this.grade.uniforms.uTiles.value = n; }

  render() { this.composer.render(); }

  dispose() {
    for (const p of this.passes) p.dispose();
    this.composer.dispose();
    this.input.dispose();
  }
}
