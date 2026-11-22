// Fullscreen CVD pass, linear RGB in -> sRGB out. Matrix path (Machado/Vienot) or Brettel two-half-plane path.
export const VERT = /* glsl */ `
varying vec2 vUv;
void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;

export const FRAG = /* glsl */ `
precision highp float;
uniform sampler2D uTex;
uniform int uMode;        // 0 matrix, 1 brettel
uniform int uKind;        // 0 protan, 1 deutan, 2 tritan (selects the missing cone; used by the Brettel planes)
uniform float uSeverity;  // observer severity (informational; uBlend is the effective mix)
uniform float uBlend;
uniform mat3 uM, uP1, uP2, uRGB2LMS, uLMS2RGB;
uniform vec3 uNsep;
uniform float uSign1;
uniform bool uSim;        // false = passthrough (panel A)
varying vec2 vUv;

vec3 linearToSrgb(vec3 c) {
  c = clamp(c, 0.0, 1.0);
  return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
}
vec3 simulate(vec3 rgb) {
  vec3 full;
  if (uMode == 0) full = uM * rgb;
  else {
    vec3 lms = uRGB2LMS * rgb;
    bool side = sign(dot(lms, uNsep)) == uSign1;
    full = uLMS2RGB * ((side ? uP1 : uP2) * lms);
  }
  return mix(rgb, clamp(full, 0.0, 1.0), uBlend);
}
void main() {
  vec3 c = texture2D(uTex, vUv).rgb;
  if (uSim) c = simulate(c);
  gl_FragColor = vec4(linearToSrgb(c), 1.0);
}`;
