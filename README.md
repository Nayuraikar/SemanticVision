# SemanticVision

Observer-conditioned, lighting-aware colour correction for dynamic 3D scenes.
Accessibility research tool, not a diagnostic tool.

## Setup

    git clone https://github.com/Nayuraikar/SemanticVision.git
    cd SemanticVision
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    cd viewer && npm install && cd ..

## Run the app (two terminals, from the repo root)

Terminal 1:

    source .venv/bin/activate
    uvicorn core.server:app --port 8000

Terminal 2:

    cd viewer && npm run dev

Open the URL Vite prints (http://localhost:5173).
Flow: Welcome -> calibrate or pick a preset observer -> pick a scene -> three-panel view (A original, B simulated CVD, C corrected).

## Tests

    source .venv/bin/activate && pytest -q
    cd viewer && npm test && npm run build

## Benchmarks (results go to bench/results/)

    python -m bench run --config solver_compare --seed 0
    python -m bench run --config calibration_recovery --seed 0
    python -m bench.ablation
    python -m bench.metric_report

Configs live in bench/configs/.

## Where things are

- docs/SPEC.md: project spec
- docs/PROGRESS.md: what is done, APIs, known gaps (read this first)
- CLAUDE.md: rules for Claude Code (never runs git, terse, tests required)
- core/ (perception, lighting, solver, server), calibration/, adapters/, viewer/, bench/, tests/

## Status

Done: protocol, perception, synthetic generator and replay, Three.js adapter and viewer, solver, lighting ablation, calibration, app flow.
Next: Phase 6 robust correction, 7 temporal stability, 8 allocation, 9 learned solver, 10 C++ kernels, 11 physics, scenes and self-labelling, 12 evaluation.
