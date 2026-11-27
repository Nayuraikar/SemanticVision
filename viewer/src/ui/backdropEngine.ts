// Shell background: one renderer showing a decorative, warm-lit traffic intersection (own TrafficSim instance,
// never exported to the core). Also renders static scene thumbnails and the camera fly-in. Cosmetic only.
import * as THREE from "three";
import { TrafficSim } from "../../../adapters/threejs/trafficSim";
import { initPhysics, physicsReady, DT } from "../../../adapters/threejs/physics";
import { STREET_LAMPS } from "../../../adapters/threejs/environments";
import { SCENES, getScene } from "../../../adapters/threejs/scenes/registry";
import * as traffic from "../../../adapters/threejs/scenes/traffic";
import { cappedPixelRatio, easeInOutCubic } from "./budget";

const FLY_MS = 900;
const ORBIT_SPEED = 0.035; // rad/s
const LOOP_S = 100;
const PARTICLES = 320;

function glowTexture(): THREE.Texture {
  const cv = document.createElement("canvas");
  cv.width = cv.height = 64;
  const g = cv.getContext("2d")!;
  const gr = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  gr.addColorStop(0, "rgba(255,255,255,1)"); gr.addColorStop(0.35, "rgba(255,255,255,.35)"); gr.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = gr; g.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(cv);
}

function disposeScene(scene: THREE.Scene) {
  scene.traverse((o) => {
    const m = o as THREE.Mesh;
    m.geometry?.dispose?.();
    for (const mat of ([] as THREE.Material[]).concat(m.material ?? [])) {
      (mat as THREE.MeshStandardMaterial).map?.dispose();
      mat.dispose();
    }
  });
}

interface Flight {
  scene: THREE.Scene; camera: THREE.PerspectiveCamera;
  from: THREE.PerspectiveCamera; to: THREE.PerspectiveCamera; start: number; done: () => void;
}

export class BackdropEngine {
  private renderer: THREE.WebGLRenderer;
  private sim: TrafficSim;
  private camera = new THREE.PerspectiveCamera(46, 16 / 9, 0.1, 500);
  private dust: THREE.Points;
  private ring: THREE.Mesh;
  private sun: THREE.DirectionalLight;
  private raf = 0;
  private last = 0;
  private angle = 0.7;
  private clock = 0;
  private active = true;
  private still = false;
  private perf = false;
  private disposed = false;
  private flight: Flight | null = null;
  /** non-home scene left on screen after a flight */
  private away: { scene: THREE.Scene; camera: THREE.PerspectiveCamera } | null = null;

  constructor(private canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "low-power" });
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.sim = new TrafficSim({ seed: 3, scenario: "normal", env: "day", aspect: 16 / 9 });
    const S = this.sim.scene;
    // golden-hour relight (replaces the environment lights on this private instance only)
    for (const o of [...S.children]) if (o.userData.svEnvLight) S.remove(o);
    S.traverse((o) => { if (o.userData.sv?.id === "sky") o.visible = false; });
    S.background = new THREE.Color().setRGB(0.42, 0.24, 0.2);
    S.fog = new THREE.FogExp2(0, 0.008);
    S.fog.color.setRGB(0.42, 0.24, 0.2);
    S.add(new THREE.HemisphereLight(0x8a8fd0, 0x4a2c1c, 1.1));
    this.sun = new THREE.DirectionalLight(0xffa24d, 3.6);
    this.sun.position.set(42, 17, 22);
    this.sun.shadow.mapSize.set(1024, 1024);
    const sc = this.sun.shadow.camera;
    sc.left = sc.bottom = -45; sc.right = sc.top = 45; sc.far = 140;
    S.add(this.sun, this.sun.target);

    const tex = glowTexture();
    for (const p of STREET_LAMPS) {
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, color: 0xffb561, blending: THREE.AdditiveBlending, depthWrite: false }));
      sprite.position.set(...p); sprite.scale.setScalar(5);
      const light = new THREE.PointLight(0xffb561, 60, 0, 2);
      light.position.set(...p);
      const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.1, p[1], 8), new THREE.MeshStandardMaterial({ color: 0x1c1a20 }));
      pole.position.set(p[0], p[1] / 2, p[2]);
      S.add(sprite, light, pole);
    }
    const pos = new Float32Array(PARTICLES * 3);
    for (let i = 0; i < PARTICLES; i++) pos.set([(Math.random() - 0.5) * 60, Math.random() * 16, (Math.random() - 0.5) * 60], i * 3);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    this.dust = new THREE.Points(geo, new THREE.PointsMaterial({
      map: tex, color: 0xffd9a0, size: 0.28, transparent: true, opacity: 0.75, blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    this.ring = new THREE.Mesh(new THREE.RingGeometry(2.4, 2.75, 48),
      new THREE.MeshBasicMaterial({ color: 0xffc274, transparent: true, opacity: 0.5, depthWrite: false }));
    this.ring.rotation.x = -Math.PI / 2; this.ring.position.y = 0.06;
    S.add(this.dust, this.ring);

    this.resize();
    void initPhysics().then(() => {
      if (this.disposed) return;
      this.restart();
      this.draw();
    });
    this.schedule();
  }

  private restart() {
    this.sim.reset();
    for (let i = 0; i < Math.round(8 / DT); i++) this.sim.tick(); // start with traffic already on the road
    this.sim.scene.userData.svEvents = [];
  }

  private get animating() { return !this.disposed && (!!this.flight || (this.active && !this.still)); }

  private schedule() {
    if (!this.raf && this.animating) this.raf = requestAnimationFrame(this.loop);
  }

  private loop = (ms: number) => {
    this.raf = 0;
    if (this.disposed) return;
    const dt = this.last ? Math.min((ms - this.last) / 1000, 0.1) : 0;
    if (this.perf && !this.flight && this.last && dt < 1 / 31) { this.schedule(); return; } // ~30 fps cap
    this.last = ms;
    if (this.flight) this.fly(ms);
    else {
      this.clock += dt;
      this.angle += dt * ORBIT_SPEED;
      if (physicsReady() && !this.away) {
        this.sim.advance(dt);
        this.sim.scene.userData.svEvents = [];
        if (this.sim.time > LOOP_S) this.restart();
      }
      const p = this.dust.geometry.getAttribute("position") as THREE.BufferAttribute;
      for (let i = 0; i < p.count; i++) {
        const y = p.getY(i) + dt * (0.15 + (i % 7) * 0.03);
        p.setY(i, y > 16 ? 0 : y);
        p.setX(i, p.getX(i) + Math.sin(this.clock * 0.4 + i) * dt * 0.2);
      }
      p.needsUpdate = true;
      this.ring.scale.setScalar(1 + 0.08 * Math.sin(this.clock * 2));
      this.draw();
    }
    this.schedule();
  };

  private orbit() {
    this.camera.position.set(Math.sin(this.angle) * 30, 12, Math.cos(this.angle) * 30);
    this.camera.lookAt(0, 2, 0);
  }

  private tone(home: boolean) {
    this.renderer.toneMapping = home ? THREE.ACESFilmicToneMapping : THREE.NoToneMapping;
    this.renderer.shadowMap.enabled = home && !this.perf;
  }

  draw() {
    if (this.disposed) return;
    if (this.away) { this.tone(false); this.renderer.render(this.away.scene, this.away.camera); return; }
    this.orbit();
    this.tone(true);
    this.sun.castShadow = !this.perf;
    this.sim.scene.traverse((o) => { if ((o as THREE.Mesh).isMesh) { o.castShadow = o.receiveShadow = !this.perf; } });
    this.renderer.render(this.sim.scene, this.camera);
  }

  resize() {
    const w = this.canvas.clientWidth || window.innerWidth, h = this.canvas.clientHeight || window.innerHeight;
    this.renderer.setPixelRatio(cappedPixelRatio(window.devicePixelRatio, this.perf));
    this.renderer.setSize(w, h, false);
    for (const c of [this.camera, this.away?.camera]) if (c) { c.aspect = w / h; c.updateProjectionMatrix(); }
    this.draw();
  }

  configure(o: { active: boolean; still: boolean; performance: boolean }) {
    const perfChanged = o.performance !== this.perf;
    this.active = o.active; this.still = o.still; this.perf = o.performance;
    this.dust.visible = !o.performance && !o.still;
    if (perfChanged) this.resize();
    if (!this.animating) {
      cancelAnimationFrame(this.raf); this.raf = 0; this.last = 0;
      if (this.active) this.draw(); // reduced motion: one static frame (when paused, the held frame is kept)
    }
    this.schedule();
  }

  private swapFade() {
    this.canvas.classList.remove("sv-swap");
    void this.canvas.offsetWidth;
    this.canvas.classList.add("sv-swap");
  }

  /** Return to the orbiting intersection. */
  home() {
    this.flight?.done();
    this.flight = null;
    if (!this.away) return;
    disposeScene(this.away.scene);
    this.away = null;
    this.swapFade();
    this.draw();
  }

  /** Eased camera flight into the scene's authored viewpoint; resolves on arrival. */
  flyTo(sceneId: string): Promise<void> {
    const def = getScene(sceneId);
    if (!def || this.disposed) return Promise.resolve();
    this.home();
    const aspect = this.camera.aspect;
    let scene: THREE.Scene, to: THREE.PerspectiveCamera, from: THREE.PerspectiveCamera;
    if (sceneId === traffic.id) {
      this.orbit();
      scene = this.sim.scene;
      to = this.sim.camera.clone();
      from = this.camera.clone();
    } else {
      const built = def.build(aspect);
      scene = built.scene;
      to = built.camera.clone();
      from = to.clone();
      const dir = to.getWorldDirection(new THREE.Vector3());
      const target = to.position.clone().addScaledVector(dir, to.position.length());
      from.position.sub(target).applyAxisAngle(new THREE.Vector3(0, 1, 0), 0.6).multiplyScalar(2.2).add(target);
      from.lookAt(target);
      this.away = { scene, camera: from.clone() };
      this.swapFade();
    }
    to.aspect = from.aspect = aspect;
    return new Promise((done) => {
      this.flight = { scene, camera: from.clone(), from, to, start: 0, done };
      this.schedule();
    });
  }

  private fly(ms: number) {
    const f = this.flight!;
    f.start ||= ms;
    const x = Math.min(1, (ms - f.start) / FLY_MS), e = easeInOutCubic(x);
    f.camera.position.lerpVectors(f.from.position, f.to.position, e);
    f.camera.quaternion.slerpQuaternions(f.from.quaternion, f.to.quaternion, e);
    f.camera.fov = f.from.fov + (f.to.fov - f.from.fov) * e;
    f.camera.aspect = f.to.aspect;
    f.camera.updateProjectionMatrix();
    this.tone(f.scene === this.sim.scene);
    this.renderer.render(f.scene, f.camera);
    if (x < 1) return;
    this.flight = null;
    // hold the arrival frame: it stays on the canvas while the background is paused behind the simulation
    if (this.away) this.away.camera = f.camera;
    this.active = false;
    f.done();
  }

  /** One static frame per scene, rendered with the shared renderer and restored within the same task. */
  async thumbnails(): Promise<Record<string, string>> {
    await initPhysics();
    const out: Record<string, string> = {};
    if (this.disposed || this.flight) return out;
    const ratio = this.renderer.getPixelRatio();
    const size = this.renderer.getSize(new THREE.Vector2());
    this.renderer.setPixelRatio(1);
    this.renderer.setSize(384, 216, false);
    for (const def of SCENES) {
      const own = def.id !== traffic.id;
      const built = own ? def.build(16 / 9) : { scene: this.sim.scene, camera: this.sim.camera.clone() };
      built.camera.aspect = 16 / 9;
      built.camera.updateProjectionMatrix();
      this.tone(!own);
      this.renderer.render(built.scene, built.camera);
      out[def.id] = this.canvas.toDataURL("image/jpeg", 0.85);
      if (own) disposeScene(built.scene);
    }
    this.renderer.setPixelRatio(ratio);
    this.renderer.setSize(size.x, size.y, false);
    this.draw();
    return out;
  }

  dispose() {
    this.disposed = true;
    cancelAnimationFrame(this.raf);
    this.flight?.done();
    if (this.away) disposeScene(this.away.scene);
    disposeScene(this.sim.scene);
    this.sim.physics?.free();
    this.sim.physics = null;
    this.renderer.dispose();
  }
}
