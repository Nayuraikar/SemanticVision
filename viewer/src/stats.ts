// CPU/GPU stats: fps, frame ms, JS heap, renderer.info, optional GPU timer query (EXT_disjoint_timer_query_webgl2).
import type * as THREE from "three";

export interface StatsSnapshot {
  fps: number; frameMs: number; heapMB: number | null; calls: number; triangles: number;
  geometries: number; textures: number; gpuMs: number | null;
}

export class Stats {
  private last = 0; private ema = 16;
  private ext: any = null; private gl: WebGL2RenderingContext | null = null;
  private pending: WebGLQuery | null = null; private gpuMs: number | null = null;
  private open = false;

  constructor(private renderer: THREE.WebGLRenderer) {
    const gl = renderer.getContext() as WebGL2RenderingContext;
    if (gl && "getExtension" in gl) {
      this.gl = gl;
      this.ext = gl.getExtension("EXT_disjoint_timer_query_webgl2");
    }
  }
  /** Call around the frame's GPU work. */
  begin() {
    if (!this.ext || !this.gl || this.pending || this.open) return;
    const q = this.gl.createQuery();
    if (!q) return;
    this.gl.beginQuery(this.ext.TIME_ELAPSED_EXT, q);
    this.pending = q; this.open = true;
  }
  end(now: number) {
    if (this.ext && this.gl && this.open) { this.gl.endQuery(this.ext.TIME_ELAPSED_EXT); this.open = false; }
    else if (this.ext && this.gl && this.pending) {
      const q = this.pending;
      const ready = this.gl.getQueryParameter(q, this.gl.QUERY_RESULT_AVAILABLE);
      const disjoint = this.gl.getParameter(this.ext.GPU_DISJOINT_EXT);
      if (ready) {
        if (!disjoint) this.gpuMs = (this.gl.getQueryParameter(q, this.gl.QUERY_RESULT) as number) / 1e6;
        this.gl.deleteQuery(q); this.pending = null;
      }
    }
    if (this.last) this.ema += 0.1 * (now - this.last - this.ema);
    this.last = now;
  }
  snapshot(): StatsSnapshot {
    const info = this.renderer.info;
    const mem = (performance as any).memory?.usedJSHeapSize as number | undefined;
    return {
      fps: 1000 / this.ema, frameMs: this.ema, heapMB: mem ? mem / 1048576 : null, calls: info.render.calls,
      triangles: info.render.triangles, geometries: info.memory.geometries, textures: info.memory.textures, gpuMs: this.gpuMs,
    };
  }
}
