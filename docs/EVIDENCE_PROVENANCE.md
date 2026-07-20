# Evidence Provenance and Interpretation Boundary

This repository separates executable workflow code from the experimental records available in the current public package.

## Available Records

- Training, rendering, RGB-D observation construction, registration, state writing, simulation synchronization, and asset-conversion source code.
- The aggregate values used by the manuscript's supplementary Fig. B.1 in `figure8_table_derived_diagnostic.csv`.
- Qualitative homepage media showing real/simulation scene variants and object-level composition.

## Records Not Present in the Public Package

- The itemized scene, object/region, and view manifest behind manuscript Tables 2 and 3.
- Per-frame pose outputs, failure traces, and method-specific run records behind Table 4.
- Raw per-frame object-load and runtime logs behind Tables 5 and 6.
- Repeated-run distributions, numerical ablation runs, and task-level reliability trials.

The missing records are not reconstructed from generated, interpolated, or model-estimated values. Aggregate manuscript entries are interpreted only within their stated evaluation scope.

## State-Acceptance Rule

The released online path uses FPFH + RANSAC initialization and GICP refinement. The active script uses `out_nb=20`, `out_std=1.5`, `src_pre_voxel=0.007`, and `tgt_pre_voxel=0.003`. It writes a state only when the returned transform is a finite, invertible `4 x 4` matrix and GICP fitness is at least `0.05`. RGB-D capture runs at 10 fps, cached object-cloud updates are capped at 5 Hz, and registration is triggered on demand. Independent pose-jump gating, SE(3) temporal smoothing, and automatic continuous retriggering are not implemented in the released execution path.

## Fig. B.1

The CSV contains 12 table-derived rows: eight PSNR entries copied from manuscript Table 5 and four cumulative stage means derived from the stage entries in Table 6. It is a plotting record for a supplementary diagnostic, not a raw per-object, per-frame benchmark.
