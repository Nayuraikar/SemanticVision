# SemanticVision: Observer-Conditioned, Lighting-Aware Colour Correction for Any Dynamic 3D Environment

*Revised project specification: architecture, contributions, build plan, evaluation*

## 1. One-paragraph summary

SemanticVision is an engine-agnostic system that makes colour-coded information in live 3D scenes distinguishable for people with Colour Vision Deficiency (CVD). It consumes a per-frame **scene snapshot** (objects, states, colours, lighting, events) from any simulation, predicts which object pairs a specific observer will confuse under the current lighting and adaptation state, and returns **material-colour overrides** and **non-colour cues** that fix the confusion with minimal deviation from the original look. Correction is learned (one observer-conditioned network approximates an exact solver), robust to calibration uncertainty (optimised over a Bayesian posterior of the observer), compute-budgeted (MCKP allocation across objects), and temporally stable (event-driven updates with a perceptual slew limit). Generality is an architectural property: the core never imports a rendering engine, and is proven on two structurally different environments plus a pixel-only black-box mode.

This is a simulation and accessibility research platform. It is not a clinical diagnostic tool, and the cockpit and medical scenarios are not operational systems.

## 2. Problem and gap

Existing CVD correction applies one fixed transform to a whole image. That fails in four ways:

1. **It ignores semantics.** A decorative wall and a traffic signal get the same treatment.
2. **It ignores lighting and adaptation.** Two signals separable at noon can collapse at night, in fog, or under tinted light.
3. **It assumes the observer model is right.** Profiles come from a coarse category (protan, deutan, tritan) and a severity slider. When the profile is wrong, the correction can be useless or harmful, and nobody reports how often.
4. **It does not run live.** Dynamic scenes need per-object, per-frame decisions under a compute budget, without flicker.

SemanticVision targets all four with a single pipeline and measures each claim against held-out observers and held-out scenes.

## 3. Contributions

Each contribution has one falsifiable claim and one experiment.

| ID | Contribution | Claim | Primary experiment |
| --- | --- | --- | --- |
| K1 | Observer-conditioned amortised solver | A single network, conditioned on an observer embedding, approximates the exact solver on unseen observers and scenes with small optimality gap and 100x or more lower latency | Gap, latency, constraint violations versus exact solver |
| K2 | Posterior-robust correction | Optimising worst-case or CVaR ΔE00 over a Bayesian posterior of the observer beats point-estimate correction when the profile is uncertain or wrong | Correction success under profile error, with and without robustness |
| K3 | Lighting- and adaptation-aware confusion prediction | Modelling lit colour (SH-L2), emissive sources, and chromatic adaptation predicts confusion better than albedo-only prediction, especially for emissive small-field objects | Ablation: albedo-only, SH, SH plus CAT16, rendered ground truth |
| K4 | Budgeted allocation with temporal stability | MCKP allocation with learned benefit estimates beats uniform, round-robin and greedy at equal compute, while event-driven updates plus slew limiting cut flicker | Quality at fixed budget, flicker metric, frame-time percentiles |
| K5 | Engine-agnostic contract with white-box and black-box modes | The same unchanged core corrects two structurally different environments, and the white-box vs black-box gap is measurable | Held-out-scene transfer, perception-gap measurement |

A parallel-performance study (Section 9) is a separate deliverable for the PAGP course, built on the same hot kernel.

## 4. System architecture

### 4.1 Design principle

The core is a pure function of a snapshot and a budget. It has zero imports from Three.js, Unity, or any engine. Everything engine-specific lives in a thin adapter.

```
[ Simulation / Engine ]
   |  Adapter (export snapshot, apply overrides)
   v
[ Snapshot protocol: versioned JSON or protobuf ]
   v
[ SemanticVision Core ]
   DETECT -> DECIDE -> CORRECT -> ALLOCATE -> STABILISE
   v
[ Overrides: palette + cues + diagnostics ]
   |  Adapter applies to materials / overlays
   v
[ Rendered scene, 3-panel viewer ]
```

### 4.2 Snapshot contract (input)

Per frame:

- `objects[]`: id, class, state, albedo\_rgb, emissive\_rgb, projected screen area, bbox, depth, velocity, task\_tags, optional lit\_color
- `lighting`: SH-L2 coefficients, or ambient plus dominant light, plus white point and exposure
- `events[]`: state\_change, collision, enter\_view, exit\_view, lighting\_change
- `observer_id` and `budget_ms`
- `schema_version`

### 4.3 Override contract (output)

- `palette_overrides[]`: id, albedo, emissive
- `cues[]`: id, type (outline, hatch, icon, label), parameters
- `confusion_pairs[]`: ids, ΔE00 before and after, feasibility flag
- `diagnostics`: per-stage latency, solver tier used, fallback flags

### 4.4 Operating modes

- **White-box:** the engine exports ground-truth classes, states, colours and lighting. Highest fidelity.
- **Black-box:** only pixels are available. Open-vocabulary detection plus SAM masks recover objects; a VLM assigns task relevance to unseen classes; colours are estimated from masks. Lower fidelity, clearly labelled as degraded mode.
- **Replay:** recorded snapshot streams run with no engine. This is the basis for every benchmark.

**Self-labelling mode (primary):** a perception model reads the rendered frame and emits the same snapshot the engine would export. Engine ground truth is used only to train and score it, never at runtime. See Section 5.11.

### 4.5 Adapters

1. Three.js adapter (primary).
2. Synthetic generator adapter (headless): N random objects, palettes, lighting, motion, events. This is the benchmark source and the generality test bed.
3. Unity or other engine adapter: optional, later. It proves the claim but is not required for v1.

## 5. Core components

### 5.1 Perception model

Pipeline per object colour:

`lit colour → chromatic adaptation (CAT16) → linear RGB → LMS → CVD simulation → CIELAB or CAM16-UCS → ΔE`

- **CVD models:** Machado (severity-parameterised), Viénot (dichromat), Brettel (dichromat with two half-planes). All three are implemented and cross-validated against published matrices and the `colour-science` library before anything else is built on them.
- **Pair distance:** D(i,j) = ΔE00(CVD(Ci), CVD(Cj)). A pair is *confusable* when D(i,j) falls below an observer- and context-dependent threshold τ(i,j).
- **Threshold model:** τ is not a constant. It is a calibrated function of projected size, background luminance, and the observer's lightness and chroma sensitivity.
- **Emissive handling:** CIEDE2000 was fit to reflective surface colours. Traffic lights and annunciators are small, emissive fields. The system evaluates CAM16-UCS alongside CIEDE2000 on emissive cases and reports where they disagree.

### 5.2 Light-transport-aware lit colour (K3)

1. Read material albedo and emissive terms.
2. Evaluate SH-L2 irradiance for the object's normal and position.
3. Classify lit, penumbra and shadow conditions.
4. Add emissive contribution separately (emissive sources do not scale with irradiance).
5. Apply chromatic adaptation to the scene white point before CVD simulation.

Ablation ladder: albedo-only → SH-L2 → SH-L2 plus shadow state → SH-L2 plus CAT16 → rendered ground truth. The distance between rungs is the K3 result.

### 5.3 Observer model and calibration

**Parameterisation (minimal, identifiable):**

- CVD type (continuous mixture over protan, deutan, tritan axes) and severity s in \[0,1\]
- Threshold scale k
- Lightness sensitivity and chroma sensitivity
- Optional background-dependence term

**Calibration:**

- Interactive trials: same/different, and slider-based discriminability.
- **Bayesian active learning:** maintain a posterior p(θ | data) over observer parameters. Choose each next stimulus by maximum expected information gain (QUEST+ style, or a Gaussian-process model of the discrimination surface).
- Stop when posterior width on the parameters that matter for correction falls below a target.
- Output: a posterior (samples or a parametric fit), not a point estimate.

**Validation:** simulated observers with known parameters. The test is parameter recovery versus trial count. Without human data, claims stop there. The calibration tool is for accessibility research, not diagnosis.

### 5.4 Confusion prediction and semantic scene graph

- Nodes: objects with class, state, importance, lit colour, screen area.
- Edges: pairs with ΔE00, threshold, adjacency, occlusion, and relation labels (signal controls intersection, vehicle approaches pedestrian).
- Pair weighting uses adjacency, projected size, and task relevance, so far-apart decorative pairs do not consume budget.
- The graph is built from the snapshot in white-box mode, or from detections in black-box mode.

### 5.5 Exact correction solver (K1 reference, K2)

**Objective for one object set and one observer θ:**

maximise the minimum (or a soft-minimum) over critical pairs of D\_θ(i,j), subject to:

- deviation from original: ΔE00(Ci', Ci) ≤ δ\_i
- display gamut and luminance bounds
- memory-colour protection (sky, skin, grass, brand colours stay inside a tight region)
- semantic consistency (same class and state share a palette entry)
- temporal stability via the slew limit

**Algorithms:**

- Baselines: no correction, global daltonisation, local daltonisation.
- Primary exact solver: Lab-space coordinate descent with projection onto constraints, from coarse-grid initialisation.
- Offline optimality reference: differential evolution.
- Everything else (PSO, SA, NSGA-II/III, Bayesian optimisation) is deferred. NSGA-II is the only optional addition, for a Pareto plot of separation vs naturalness.

**Posterior-robust variant (K2):** replace D\_θ with a risk measure over sampled observers θ₁…θ\_M \~ posterior:

maximise CVaR\_α of { D\_θm(i,j) } over m

Compare against the point-estimate solver (posterior mean) at several levels of profile error.

**Verification loop:** after correction, re-evaluate in screen space. Any pair still below threshold is retried within budget, then escalated to non-colour cues.

### 5.5.1 Non-colour fallback

When a pair cannot be separated within constraints, the system emits cues: outline style, hatching pattern, shape marker, label or icon. Cue assignment is a small constraint-satisfaction step so adjacent confusable objects get *different* cues. Infeasible-pair count is a reported metric.

### 5.6 Observer-conditioned learned solver (K1)

**Task:** given an observer embedding z(θ), object colours, lighting summary, and importance weights, predict corrected palette in one forward pass.

**Architecture:**

- Set encoder over objects (DeepSets or small transformer) so object count is variable.
- Observer conditioning by FiLM layers or a small hypernetwork driven by z(θ).
- Output: per-object ΔLab residual, clipped by a differentiable projection to the constraint set.
- Start with MLP or residual MLP. A GNN is a justified upgrade only if it beats the set encoder on held-out scenes.

**Training:**

- Data: exact-solver outputs across sampled observers, severities, lightings, and palettes.
- Differentiable observer: RGB → LMS → CVD → Lab, plus differentiable CIEDE2000.
- Loss: perceptual separation loss through the observer, naturalness loss, temporal stability loss, constraint-violation penalty, and a supervised term against exact-solver output.
- Splits: held-out observers (different model family, severity range, added threshold noise) and held-out scenes.

**Deployment:** export to ONNX, run via ONNX Runtime in Python or in the C++ core. Verified by screen-space check, with fallback to the exact solver. The **fallback rate** is a reported metric.

**Reported:** optimality gap, latency, model size, memory, constraint violations, generalisation gap, fallback rate.

### 5.7 Importance estimation

1. **Baseline:** rule-based score from class, state, state-change rate, motion, collision proximity, screen area, distance, task tags.
2. **Learned model (conditional):** only kept if it beats rules on held-out scenes.
3. **Label source must not be circular.** Labels from a task simulator that shares the system's own confusion model make accuracy self-graded. Use an independent computational observer instead: a signal-detection or visual-search model (d′ as a function of size and eccentricity, with a reaction-time penalty), with parameters taken from published psychophysics.

### 5.8 Budgeted allocation (K4)

Per object, tiers:

| Tier | Action | Cost | Benefit |
| --- | --- | --- | --- |
| 0 | Skip | \~0 | 0 |
| 1 | Reuse cached correction | tiny | cached quality, decays with scene change |
| 2 | Learned solver | low | high |
| 3 | Coarse exact | medium | higher |
| 4 | Fine exact | high | highest |

This is a Multiple-Choice Knapsack Problem: choose exactly one tier per object to maximise total expected perceptual benefit under budget B ms.

**Solvers compared:** greedy by benefit density, DP, convex-hull pruning with Lagrangian relaxation, and branch-and-bound as the offline optimum. Benchmarked on synthetic workloads from 50 to 5,000 objects.

**Learned benefit:** a contextual bandit estimates expected benefit per (object features, tier) online, replacing hand-written estimates. Compared against static MCKP.

**Prediction:** a Kalman or constant-velocity predictor marks objects about to enter view so their correction is precomputed.

### 5.9 Temporal stability (K4)

- **Event-driven recomputation** on: object enters view, collision, state change, lighting change, importance jump, new confusing pair, profile change.
- **Hysteresis** on confusion decisions (separate enter and exit thresholds).
- **Perceptual slew limiter:** ΔE00(p(t), p(t−1)) ≤ r\_max · Δt.
- **Double-buffered palette:** the renderer always reads the last validated palette while workers compute the next.
- **Flicker metric:** sum over frames of ΔE00 between consecutive displayed colours. Ablation: none → EMA → hysteresis → slew → event-driven, each measured alone and combined.

### 5.10 Black-box perception (K5)

- Open-vocabulary detector (OWL-ViT or Grounding DINO) plus SAM for masks, replacing a fixed-class YOLO.
- A VLM or text-conditioned scorer assigns task relevance to unseen classes given the task description.
- Colour and lighting estimated from masks and image statistics.
- **Reported:** white-box vs black-box gap in correction quality and latency. This gap is itself a result.

A lightweight YOLO model stays as a fast baseline for known classes.

### 5.11 Self-labelling perception front end (primary mode)

The model analyses the rendered scene and produces the labels itself, so the engine exports nothing at runtime. This makes "the AI labels everything" literally true while the rest of the pipeline stays unchanged, and it gives a measured labeller-accuracy result.

- **Input:** the rendered frame (plus depth if the engine provides it), the observer ID and the task text.
- **Model:** an instance segmenter (RF-DETR-Seg or a SAM-class model), an open-vocabulary classifier (OWL-ViT, Grounding DINO or a CLIP head) for unseen classes, a state classifier on object crops (for example signal state), a colour and lighting estimator, and a relevance head conditioned on the task text.
- **Output:** the standard snapshot: objects with stable IDs, classes, states, colours, masks, screen area, depth, importance, and events derived from tracking (enter view, state change). Stable IDs come from a tracker such as ByteTrack or a Kalman-filter tracker.
- **Ground truth is for training and scoring only.** The engine renders ground-truth ID, class, state and colour buffers for free, so no manual annotation is needed. They are never an input at runtime.
- **Training data:** generated from the simulations and the synthetic generator with randomised lighting, weather, palettes and layouts. Evaluation uses held-out scenes and held-out environments.
- **Latency:** the labeller runs at a lower rate than rendering. Masks are propagated between updates by tracking or optical flow, and the correction reads the cached palette per track ID. A distilled small model replaces the large one where speed requires it.
- **Confidence handling:** low-confidence objects are not recoloured. They receive non-colour cues or the last validated palette, and the screen-space verification stage still checks the result.

**Applying corrections without engine labels.** The labeller sees lit pixels, not materials. Material-space correction (Section 5.5) therefore needs a mapping from each detected object to an engine material. Two options:

1. **Mapped mode:** the adapter looks up each mask in the engine's ID buffer to find the material. The ID buffer is used only for this lookup, not for labels. Material-space quality is preserved.
2. **Pixel-only mode:** the correction is applied per mask in screen space. This works on any application, but loses lighting-aware material correction and is reported as the degraded mode.

**Metrics:**

- Mask mIoU and class accuracy, including unseen classes.
- State accuracy.
- ID consistency over time (ID switches, IDF1).
- Colour estimation error in ΔE00.
- Relevance ranking agreement with the independent task-observer importance (Section 5.7), not with labels produced by the same model.
- Latency and labeller update rate.
- **Downstream gap:** minimum ΔE00 of critical pairs and correction success rate with self-labelled input versus white-box input.
- **Error-injection study:** inject controlled labeller errors (wrong class, missed object, wrong state) into white-box input to show how correction quality degrades, which separates labeller error from solver error.

**Front ends are swappable.** White-box export, the self-labelling model and a fixed-class YOLO baseline all emit the same snapshot. The white-box versus self-labelling gap is the K5 perception-gap result. In the build plan this replaces the black-box mode of Phase 11 as the primary front end, and its training data can be generated as soon as the Three.js adapter exists in Phase 2.

## 6. Dynamic simulation layer

- Three.js rendering with Rapier3D physics.
- Physics events (collision, trigger entry) become snapshot events. A collision is a semantic event: it changes object states, activates hazard indicators, and changes importance.
- Deterministic seeded scenarios with fixed timestep so any run is exactly reproducible.
- Record and replay of snapshot streams to disk.

### Environments

| Priority | Environment | Why |
| --- | --- | --- |
| v1 | Traffic intersection (day, night, fog, rain) | Reflective plus emissive objects, moving, collision events |
| v1 | Structurally different second environment, e.g. instrument panel or monitor dashboard with many emissive indicators | Proves generality; held-out scene for transfer |
| Future | Cockpit, ICU, construction, rail and airport | Described as extensions, built only if time remains |

The second environment is **not tuned against**. It is the held-out test of the engine-agnostic claim.

## 7. Three-panel viewer

All panels render the same live simulation state:

- **A:** original scene
- **B:** simulated CVD perception of the original
- **C:** corrected scene passed through the same simulation

Overlays: confusion graph, importance heatmap, bounding boxes, per-object ΔE00 before and after, solver tier used, per-stage latency, CPU/GPU stats, observer posterior summary.

## 8. Technology stack

| Layer | Choice |
| --- | --- |
| Frontend | React, TypeScript, Three.js, GLSL, dashboards |
| Physics | Rapier3D |
| Backend and AI | Python, FastAPI, PyTorch, NumPy, SciPy, OpenCV, ONNX Runtime |
| Perception refs | `colour-science` for validation |
| Open-vocab | OWL-ViT or Grounding DINO, SAM |
| Native core | C++20, OpenMP, SIMD (NEON on M1), one GPU path |
| Protocol | Versioned JSON schema or protobuf |
| Storage | SQLite (profiles, experiment metadata), CSV and NumPy (benchmarks), `.pt` and ONNX (models) |
| Dev machine | MacBook Air M1; Kaggle for training |

Expensive work runs asynchronously so it never blocks the render loop.

## 9. Parallel performance study (PAGP)

**Hot kernel:** pairwise ΔE00 over objects × candidate palettes × observers.

**Implementations:**

1. Sequential C++ reference
2. SIMD (NEON)
3. OpenMP
4. GPU (pick one: Metal compute or WebGPU)
5. Hybrid scheduler choosing CPU or GPU by batch size

**Metrics:**

- Speedup = T\_seq / T\_par
- Parallel efficiency = Speedup / workers
- GPU vs CPU **crossover point** by workload size
- CPU/GPU utilisation, memory, synchronisation overhead
- Frame time P50, P95, P99
- Quality at equal compute budget

Every parallel version must reproduce the sequential output within numerical tolerance. Speedup without correctness checks is not reported.

Metal compute is a separate engineering effort from WebGL shaders. If the app is browser-based, WebGPU compute is cheaper. CUDA is out of scope.

## 10. Five-stage runtime pipeline

1. **DETECT:** read snapshot (white-box) or run perception (black-box); update tracks, states, events.
2. **DECIDE:** compute lit colours, confusion graph, importance, task relevance; rank objects and pairs.
3. **CORRECT:** run the tier assigned to each object (learned, coarse or fine exact), with verification and robustness over the observer posterior.
4. **ALLOCATE:** MCKP picks tiers under budget, using learned benefit and predicted visibility. (Runs before CORRECT each frame; shown fifth for data-flow clarity in the original document.)
5. **STABILISE AND FALLBACK:** hysteresis, slew limit, double buffering, non-colour cues; emit overrides.

## 11. Evaluation methodology

### 11.1 Rules that make results credible

- **Never score under the observer you optimised for.** Held-out observers differ in model family, severity, and added threshold noise.
- **Held-out scenes** are not used for tuning any threshold, weight, or hyperparameter.
- **Independent task observer** for importance and task-performance results (Section 5.7).
- All runs are seeded and replayable. Report mean and confidence intervals over seeds, with paired significance tests.
- Compare against **published baselines**: Machado and Viénot-based daltonisation, and at least one recolouring method from the literature.
- Every number comes from a script, not from a manual run.

### 11.2 Experiments by contribution

| Contribution | Experiment | Key metrics |
| --- | --- | --- |
| K1 | Learned vs exact solver on held-out observers and scenes | Optimality gap, latency, size, violations, fallback rate |
| K2 | Point-estimate vs posterior-robust correction under profile error | Min ΔE00 of critical pairs, success rate, worst-case degradation |
| K3 | Lit-colour ablation ladder; CIEDE2000 vs CAM16-UCS on emissive objects | Confusion prediction accuracy vs rendered ground truth |
| K4 | Allocation at fixed budget; temporal-stability ablation | Quality per ms, flicker, P95/P99 frame time |
| K5 | Run unchanged core on held-out environment and in black-box mode | Transfer gap, white-box vs black-box gap |
| Calibration | Simulated-observer parameter recovery | Posterior error vs trials |
| Parallel | Five implementations across workload sizes | Speedup, efficiency, crossover |
| Self-labelling front end | Self-labelled vs white-box snapshots on held-out scenes; error-injection study | mIoU, class and state accuracy, ID consistency, colour error, downstream correction gap |

### 11.3 Core metrics

Minimum and mean ΔE00 across critical pairs; correction success rate; naturalness deviation; infeasible-pair count; solver time; optimality gap; detection precision and recall (black-box); tracking consistency; flicker; collision-event response latency; CPU and GPU utilisation; memory; frame-time percentiles; speedup and efficiency; scaling with object count.

### 11.4 Task-performance results

Reported from a computational observer (d′-style) in simulated tasks (identify a signal, find a warning indicator). Labelled as simulation results only.

### 11.5 Optional human pilot

A small consented pilot with participants who have CVD (even n≈5) upgrades the paper from simulation-only to initial human evidence. Requires approval through your institution's ethics process before any data collection.

## 12. Build plan

| Phase | Deliverable | Exit criterion |
| --- | --- | --- |
| 0 | Scope freeze, headline claim, snapshot schema | One-page spec signed off |
| 1 | Python reference perception core: CVD models, CAT16, Lab, CIEDE2000, confusion graph | Matches `colour-science` and published matrices; unit tests pass |
| 2 | Contract, Three.js adapter, synthetic generator, record/replay, benchmark harness CLI | Same core runs on toy scene and generated scenes unchanged |
| 3 | Solver (K1 reference): baselines, coordinate descent, DE reference, constraints, three-panel viewer | Beats global daltonisation on min ΔE00 at bounded deviation |
| 4 | Lighting-aware prediction (K3) with ablation ladder | Ablation table complete |
| 5 | Calibration: observer parameterisation, Bayesian active learning, simulated-observer recovery | Known parameters recovered within tolerance |
| 6 | Posterior-robust correction (K2) | Robust beats point-estimate under profile error |
| 7 | Temporal stability and non-colour fallback (K4a) | Flicker ablation complete |
| 8 | Importance baseline, MCKP solvers, bandit benefit (K4b) | Quality per ms beats greedy and uniform at equal budget |
| 9 | Learned solver (K1): data generation, training, ONNX, verification and fallback | Gap and latency table on held-out observers and scenes |
| 10 | C++ kernel and parallel study (can start after Phase 1) | Correct outputs, speedup table, crossover found |
| 11 | Physics, traffic environment, second environment, black-box mode (K5) | Held-out environment run with no core changes |
| 12 | Full evaluation, figures, write-up | All experiments scripted and reproducible |

**Parallel tracks:** Phase 10 is independent of Phases 4 to 9 once the Phase 1 kernel exists, so schedule it by the PAGP deadline. Phase 11 environment work can run alongside Phases 6 to 9.

**If time runs short**, keep Phases 1, 2, 3, 7, 10 for a working generic engine with a measured speedup. Phases 4, 6, 8, 9 add the strongest research claims. Phases 5 and 11 are the first to trim.

## 13. Repository layout

```
semanticvision/
  core/           # pure Python reference: perception, solver, allocator, stabiliser
  native/         # C++20 kernels, OpenMP, SIMD, GPU path
  protocol/       # schema, versioning, validators
  adapters/
    threejs/
    synthetic/
    unity/        # optional
  learning/       # datasets, training, observer-conditioned solver, ONNX export
  calibration/    # trials, active learning, posterior
  blackbox/       # open-vocab detection, SAM, relevance scoring
  viewer/         # React three-panel app
  bench/          # configs, runners, replay logs, plots
  tests/
  docs/
```

## 14. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Scope too large | Frozen contribution set; deferred list kept explicit |
| Learned solver does not generalise to unseen observers | Verification and exact fallback; report the gap honestly; widen training distribution |
| Self-graded or circular evaluation | Held-out observers and scenes; independent task observer |
| Perceptual metric invalid for emissive small-field objects | Compare CIEDE2000 with CAM16-UCS; report disagreement |
| GPU slower than CPU on small batches | Treat as a finding; report crossover |
| Metal engineering time | Choose one GPU path early; OpenMP and SIMD are the safe fallback |
| Black-box mode weak | Present as degraded-mode demo with measured gap, not a full feature |
| Calibration unvalidated on humans | State simulation-only limits; optional ethics-approved pilot |

## 15. Scope and limitations

- Experimental accessibility research and simulation platform. Not clinical, not for operational aviation, medical, or safety decisions.
- Initial evaluation uses computational observers. No claim of clinical benefit without studies involving people with CVD.
- Display characteristics, transparency, complex reflections and highly dynamic illumination limit lighting prediction. These are documented and tested, not hidden.
- Calibration estimates accessibility parameters. It does not diagnose.

## 16. Expected outcomes

1. An engine-agnostic correction core with a versioned snapshot protocol and replay support.
2. A Bayesian calibration module producing observer posteriors.
3. A lighting- and adaptation-aware confusion predictor with a measured ablation.
4. An exact solver, a posterior-robust solver, and an observer-conditioned learned solver with a measured optimality gap.
5. A budgeted allocator and a temporal-stability layer with measured flicker reduction.
6. A white-box and black-box perception path, proven on two environments.
7. A parallel-performance study across sequential, SIMD, OpenMP, GPU and hybrid.
8. A reproducible benchmark harness and a three-panel viewer.

## 17. How to describe the novelty

The novelty is the combination: an observer-conditioned amortised solver, correction that is robust to calibration uncertainty, lighting- and adaptation-aware confusion prediction, budgeted allocation with temporal stability, and an engine-agnostic contract that works with or without privileged scene access. Each piece has its own experiment, and every comparison is against held-out observers and scenes, so the results cannot be dismissed as self-graded.
