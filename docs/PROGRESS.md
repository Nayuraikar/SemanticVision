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
