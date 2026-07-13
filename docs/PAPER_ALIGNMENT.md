# Paper Alignment Notes

This repository follows the manuscript wording of:

`DGSRSim: Object-Level Decoupled 3D Gaussian Assets for Robot Simulation and Online State Synchronization`

Project homepage: https://rtdgs.github.io/DGSRSim/

## Core Claims Reflected in the Repository

- Object-level reconstruction: `GaussianModel/` partitions one shared Gaussian scene model into independently addressable object subsets and a retained background subset.
- Online state estimation: `FastSAMRealtime/` turns RGB-D observations into object-level point clouds, applies filtering, and aligns them with asset-derived reference geometry.
- Simulation synchronization: `Simulation/` imports converted assets and writes estimated object states into the corresponding simulation objects.
- Scene mutability: the simulation scene can be composed from a background asset, movable object assets, and robot assets; object assets can be added, removed, replaced, or rearranged at the asset level.

## Evidence Scope

The manuscript reports results for the offline evaluation set and the current online test sequences under the reported implementation configuration. The repository documentation uses the same boundary:

- PSNR, SSIM, and LPIPS describe rendering or visual-consistency quality.
- Pose errors describe registration/state-estimation accuracy under the evaluated sequences.
- The reported stage means sum to approximately 73 ms per processed update. Their 13.7 Hz reciprocal is arithmetic rather than observed throughput; the released path captures at 10 fps, updates the cached object cloud at up to 5 Hz, and triggers registration on demand.
- Fig. 8 in the manuscript is a supplementary load diagnostic, not a complete per-object, per-frame runtime scaling law.
- The minimal pick demonstration is an interface-chain record, not a task-level reliability metric.
- Tables 2-4, 6, and 7 are aggregate manuscript entries; the public repository does not currently contain the itemized offline manifest or per-frame records required for independent recomputation.

The active `rt_seg_strict_align_cut_object_pcd5.py` configuration uses `out_nb=20`, `out_std=1.5`, `src_pre_voxel=0.007`, and `tgt_pre_voxel=0.003`. The released state-acceptance rule requires a finite, invertible `4 x 4` transform and GICP fitness of at least `0.05`. The observed point cloud is in Kinect CameraSpace. The checked-in operational calibration defines the metric scene frame to coincide with CameraSpace and can be overridden by a measured installation-specific transform. The state writer records `Q_normalized = c_source + s * (Q_raw - c_target)` and composes `A_scene_from_asset_raw = T_scene_from_camera @ inv(T_normalized_target_from_camera) @ A_normalized_target_from_asset_raw`. This scale-preserving similarity is the primary simulation state; the rigid `T_tgt_to_scene.npy` output is retained for compatibility. Independent pose-jump gating, SE(3) temporal smoothing, and automatic continuous retriggering are not part of the released execution path or the reported evaluation.

Robot task reliability, physical contact stability, collision penetration, grasp success rate, and long-horizon closed-loop manipulation require independent task-level protocols and are outside the current evaluation scope.

## Repository-to-Paper Mapping

| Paper component | Repository location | Role |
| --- | --- | --- |
| Offline object/background asset construction | `GaussianModel/` | Object Gaussian assets, background field, rendering, and editing workflow |
| Online RGB-D object observation | `FastSAMRealtime/` | RGB-D alignment, segmentation, object point-cloud cropping, filtering |
| Coarse-to-fine state estimation | `FastSAMRealtime/` | Point-cloud registration and state-estimation scripts |
| Asset conversion | `third_party/3dgrut_conversion/` | Gaussian PLY to mesh/USD/USDZ/collision conversion |
| Simulation synchronization | `Simulation/` | Scale-preserving JSON state consumption, raw-asset transform writing, scene composition |
| Camera-to-scene calibration | `FastSAMRealtime/configs/calibration/` | Operational frame definition and installation-specific override point |
| Asset scale provenance | `FastSAMRealtime/configs/assets/` | Target PLY and converted USDZ identity, bounds, and hashes |
| External large files | `docs/LARGE_FILES.md` | Placement paths for weights, raw data, point clouds, and simulation assets |
| Evidence provenance | `docs/EVIDENCE_PROVENANCE.md` | Available records, missing run-level records, and interpretation boundary |
| Fig. 8 source values | `docs/figure8_table_derived_diagnostic.csv` | Direct copy of the aggregate entries plotted from manuscript Tables 6 and 7 |

## Terminology Used Here

The documentation uses "online processing capability" and "online interaction potential" for the reported runtime behavior. It avoids unconditional robot-control, task-reliability, or physical-interaction-success claims because those require separate task-level experiments.
