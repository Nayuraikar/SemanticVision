import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import {
  applyOverrides, buildToyScene, exportSnapshot, postCorrect, postGT, renderGroundTruth, stepToyScene,
  type Overrides,
} from "../../adapters/threejs";

const API = "http://localhost:8000";
const W = 640, H = 360;

export function App() {
  const ref = useRef<HTMLCanvasElement>(null);
  const capture = useRef<() => Promise<void>>(async () => {});
  const [status, setStatus] = useState("idle");
  const [ov, setOv] = useState<Overrides | null>(null);

  useEffect(() => {
    const renderer = new THREE.WebGLRenderer({ canvas: ref.current!, antialias: true });
    renderer.setSize(W, H, false);
    const { scene, camera } = buildToyScene(W / H);
    const prev = new Map<string, [number, number]>();
    let frame = 0, prevT: number | undefined, raf = 0, busy = false;
    const loop = (ms: number) => {
      const t = ms / 1000;
      stepToyScene(scene, t);
      renderer.render(scene, camera);
      if (frame % 15 === 0 && !busy) {
        busy = true;
        const snap = exportSnapshot(scene, camera, { frame, t, observerId: "viewer", budgetMs: 16, prev, prevT });
        postCorrect(API, snap)
          .then((o) => { applyOverrides(scene, o); setOv(o); setStatus("ok"); })
          .catch((e) => setStatus(String(e)))
          .finally(() => { busy = false; });
      }
      prevT = t;
      frame++;
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    capture.current = async () => {
      const gt = renderGroundTruth(renderer, scene, camera, W, H);
      await postGT(API, frame, gt);
    };
    return () => { cancelAnimationFrame(raf); renderer.dispose(); };
  }, []);

  return (
    <div style={{ padding: 12 }}>
      <canvas ref={ref} width={W} height={H} style={{ width: W, height: H }} />
      <div>
        <button onClick={() => capture.current().then(() => setStatus("gt saved")).catch((e) => setStatus(String(e)))}>
          capture GT
        </button>{" "}
        {status} | overrides: {ov?.palette_overrides?.length ?? 0} | tier: {ov?.diagnostics?.solver_tier ?? "-"}
      </div>
    </div>
  );
}
