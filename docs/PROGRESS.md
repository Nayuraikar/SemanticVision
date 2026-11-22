# Progress

## Phase: protocol (SPEC 4.1-4.5)
- Done: protocol/{models,builder,io,schema}.py, protocol/schemas/*.v1.0.schema.json, docs/SCOPE.md, tests/test_protocol.py (14 pass).
- API: Snapshot, Overrides, SceneObject, SHLighting|AmbientDominantLighting, Event, PaletteOverride, Cue, ConfusionPair, Diagnostics; SCHEMA_VERSION="1.0" (major-version check).
- API: SnapshotBuilder(observer_id,budget_ms).{sh2,ambient_dominant,add_object,add_event,build}; write_json/read_json(path,model); write_ndjson(items,path)/read_ndjson(path,model); export_schemas(dir).
- Wire: JSON uses key "class" (alias of SceneObject.cls); dump with by_alias=True. RGB linear; albedo in [0,1], emissive HDR >=0.
- Gaps: no protobuf; no per-class state enums; lit_color unvalidated vs lighting.

## Phase: perception (SPEC 5.1)
- Done: core/perception/{color,cvd,distance,threshold,graph}.py + _machado_table.py; tests/test_perception.py cross-validated vs colour-science, Machado/Vienot published matrices, Sharma dE00 pairs.
- API: srgb_to_linear/linear_to_srgb, linear_to_lms (HPE), cat16_matrix(xyz_src,xyz_dst,D)/cat16_adapt/cat16_degree, xyz_to_lab, ciede2000, xyz_to_cam16ucs(xyz,xyz_w,L_A,Y_b,surround)/cam16ucs_distance; cvd.machado(rgb,kind,sev)/vienot(rgb,kind)/brettel(rgb,kind)/simulate(rgb,model,kind,sev), kind in protan|deutan|tritan, all on linear sRGB.
- API: distance.perceived(rgb,model,kind,severity,illum_xyz,L_A)/pair_distance/pair_matrix(space de2000|cam16); threshold.tau(size_deg,bg_lum,s_L,s_C,k,params); graph.build_graph(colors,ids,size_deg,area,importance,bg_lum,adjacency|bbox,far_weight,...)->ConfusionGraph.
- Gaps: tau params uncalibrated defaults; Vienot tritan = published coeffs (approx); colour-science Machado interpolation is buggy between table points (ours is true linear); Brettel/Vienot severity<1 = RGB blend; per-node s_L/s_C reduced by min.

## Phase: synthetic + bench (SPEC 4.4, 4.5, 6, 11.1)
- Done: adapters/synthetic/generator.py, core/observers.py, bench/{registry,replay,stats,runner,__main__}.py, bench/experiments/observer_gap.py, bench/configs/observer_gap.json, tests/test_synthetic_bench.py (all pass).
- API: SynthConfig(n_objects,n_frames,dt,palette,lighting,motion,...).from_dict; generate(cfg,seed)->Iterator[Snapshot]; sample_observers(n,seed,split)/train_observers/heldout_observers -> Observer(id,split,model,kind,severity,k,s_L,s_C).sim_kwargs()/key(); train=Machado sev[0.5,1], heldout=Brettel/Vienot(no tritan) sev[0.1,0.5)+lognormal noise on k,s_L,s_C.
- API: @experiment(name) fn(params,seed)->{arm:{metric:float}}, auto-discovered from bench/experiments/*.py; config JSON {experiment,params,n_seeds,baseline}; `python -m bench run --config X --seed S [--n-seeds N --out D]`, `python -m bench list`; writes <name>_{raw,summary,paired}.csv to bench/results; bench.replay.record/replay/record_synthetic (NDJSON).
- Gaps: JSON configs only (no pyyaml); no scene-class/environment presets (traffic/dashboard are palettes only); threshold noise is carried on Observer (k,s_L,s_C) but not yet consumed by graph builder; no published-baseline comparisons yet.

## Phase: threejs adapter + server (SPEC 4.5, 6, 8)
- Done: viewer/ (Vite+React+TS), adapters/threejs/{snapshot,overrides,gt,toyScene,index}.ts + protocol.gen.ts (generated: `python -m protocol.ts`, sync-tested), core/server.py, protocol/ts.py, tests/fixtures/toy_scene.ndjson, tests/test_server.py, adapters/threejs/adapter.test.ts (vitest; UPDATE_FIXTURE=1 re-records).
- API: exportSnapshot(scene,camera,{frame,t,observerId,budgetMs,prev,prevT}); applyOverrides(scene,ov) (idempotent, restores); renderGroundTruth(renderer,scene,camera,w,h)/postGT/encodeGT; mesh.userData.sv={id,cls,state,tags}; bbox normalised viewport top-left.
- API: core.server.create_app(gt_dir)/app; POST /correct Snapshot->Overrides (identity stub), POST /gt GTFrame -> bench/logs/gt/gt_<frame>.npz+.json (instance/class/state/color RGBA8, row0=top, index 0=bg), GET /health; CORS localhost:5173.
- Gaps: GT render + cues (hatch/icon/label) untested without WebGL; lit_color not exported; events not emitted by adapter; no Rapier physics yet; viewer is single-panel (no A/B/C); run: uvicorn core.server:app --port 8000.

## Phase: solver (SPEC 5.5, 5.5.1)
- Done: core/solver/{problem,baselines,solver,cues,metrics}.py, bench/experiments/solver_compare.py, bench/configs/solver_compare.json, bench/pareto.py (own NSGA-II), tests/test_solver.py (9 pass). Bench (20 seeds, held-out scoring): cd vs global_dalt min dE00 +4.5 (p<1e-3), success +0.68, max deviation 8 vs 22 (delta=15).
- API: Scene.from_snapshot(snap,fov,adj_margin,...)/Scene.build; Scene.graph(obs,colors); SolverConfig(delta,delta_mem,L_range,cap,beta,lam,budget_ms,retries,n_cues); correct(scene,train_obs,cfg,seed)->Solution(colors,palette_rgb,infeasible,cues,cue_unresolved,attempts,time_ms,nat); de_reference(scene,obs,cfg,seed,maxiter,popsize); coordinate_descent(Problem); screen_check(scene,rgb,obs); assign_cues(pairs,vocab)->(cues,clashes).
- API: baselines none/global_daltonisation/local_daltonisation(colors,kind,scene=)/kuhn2008(colors,kind) on lit linear RGB; naturalness(orig,new,protected)->mean/max dE00, mem_drift; pair_scores(colors,obs,edges,tau). `python -m bench.pareto`.
- Gaps: palette entry = class+state+colour cluster (tol 8 dE00), same-group pairs excluded from critical set; lit colour = albedo (no lighting/CAT in solve); Kuhn is a stress-minimisation variant (not original mass-spring); local dalt is heuristic; solver sees train observers of the held-out kind (kind assumed diagnosed); temporal slew + CVaR (K2) not implemented; cues are not yet wired into server /correct.
