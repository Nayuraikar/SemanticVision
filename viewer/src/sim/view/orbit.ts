// Damped orbit / zoom / fly-to applied ON TOP of the sim's authored camera: at rest the framing is unchanged.
import * as THREE from "three";

export const PITCH_MIN = -0.45, PITCH_MAX = 0.85, ZOOM_MIN = 0.2, ZOOM_MAX = 2.4, FOCUS_ZOOM = 0.45;
const DAMPING = 8;
const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));

export const damp = (x: number, goal: number, dt: number, lambda = DAMPING) => goal + (x - goal) * Math.exp(-lambda * dt);

export interface OrbitPose { yaw: number; pitch: number; zoom: number; focus: number }

export class OrbitRig {
  readonly goal: OrbitPose = { yaw: 0, pitch: 0, zoom: 1, focus: 0 };
  readonly cur: OrbitPose = { yaw: 0, pitch: 0, zoom: 1, focus: 0 };
  private point = new THREE.Vector3();
  private pointGoal = new THREE.Vector3();

  rotate(dYaw: number, dPitch: number) {
    this.goal.yaw += dYaw;
    this.goal.pitch = clamp(this.goal.pitch + dPitch, PITCH_MIN, PITCH_MAX);
  }
  zoomBy(factor: number) { this.goal.zoom = clamp(this.goal.zoom * factor, ZOOM_MIN, ZOOM_MAX); }
  focusOn(p: THREE.Vector3) {
    if (this.cur.focus < 0.01) this.point.copy(p);
    this.pointGoal.copy(p);
    this.goal.focus = 1;
    this.goal.zoom = Math.min(this.goal.zoom, FOCUS_ZOOM);
  }
  reset() { Object.assign(this.goal, { yaw: 0, pitch: 0, zoom: 1, focus: 0 }); }

  /** dt in seconds; snap = reduced motion */
  update(dt: number, snap = false) {
    for (const k of ["yaw", "pitch", "zoom", "focus"] as const) {
      this.cur[k] = snap ? this.goal[k] : damp(this.cur[k], this.goal[k], dt);
    }
    if (snap) this.point.copy(this.pointGoal);
    else this.point.lerp(this.pointGoal, 1 - Math.exp(-DAMPING * dt));
  }

  /** Writes position/orientation of `out` from the authored camera plus the current orbit. */
  apply(base: THREE.Camera, out: THREE.Camera) {
    const { yaw, pitch, zoom, focus } = this.cur;
    const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(base.quaternion);
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(base.quaternion);
    // pivot: the point on the view axis closest to the scene origin
    const dist = Math.max(1, -base.position.dot(fwd));
    const pivot = base.position.clone().addScaledVector(fwd, dist);
    const offset = base.position.clone().sub(pivot);
    const q = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), yaw)
      .multiply(new THREE.Quaternion().setFromAxisAngle(right, -pitch));
    offset.applyQuaternion(q).multiplyScalar(zoom);
    pivot.lerp(this.point, focus);
    out.position.copy(pivot).add(offset);
    out.quaternion.copy(q).multiply(base.quaternion);
    out.updateMatrixWorld(true);
  }
}
