// Bloom + vignette + warm grade over the finished (sRGB) three-panel composite. Built only while enabled.
import * as THREE from "three";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { TexturePass } from "three/examples/jsm/postprocessing/TexturePass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { ShaderPass } from "three/examples/jsm/postprocessing/ShaderPass.js";

const GRADE = {
  uniforms: { tDiffuse: { value: null as THREE.Texture | null } },
  vertexShader: /* glsl */ `
varying vec2 vUv;
void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: /* glsl */ `
uniform sampler2D tDiffuse;
varying vec2 vUv;
void main() {
  vec3 c = texture2D(tDiffuse, vUv).rgb * vec3(1.06, 1.0, 0.9);
  vec2 p = vec2(fract(vUv.x * 3.0), vUv.y) - 0.5;   // one vignette per panel
  c *= 1.0 - 0.7 * smoothstep(0.15, 0.55, dot(p, p));
  gl_FragColor = vec4(c, 1.0);
}`,
};

export class Cinematic {
  readonly input: THREE.WebGLRenderTarget;
  private composer: EffectComposer;
  private passes: { dispose(): void }[];

  constructor(renderer: THREE.WebGLRenderer, w: number, h: number) {
    this.input = new THREE.WebGLRenderTarget(w, h, { depthBuffer: false });
    this.composer = new EffectComposer(renderer);
    const source = new TexturePass(this.input.texture);
    const bloom = new UnrealBloomPass(new THREE.Vector2(w, h), 0.55, 0.4, 0.78);
    const grade = new ShaderPass(GRADE);
    this.passes = [source, bloom, grade];
    for (const p of [source, bloom, grade]) this.composer.addPass(p);
  }

  render() { this.composer.render(); }

  dispose() {
    for (const p of this.passes) p.dispose();
    this.composer.dispose();
    this.input.dispose();
  }
}
