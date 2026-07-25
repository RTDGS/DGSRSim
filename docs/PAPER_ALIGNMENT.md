# Paper Alignment Notes

This repository follows the manuscript wording of:

`DGSRSim: An Object-Level Decoupled 3D Gaussian Scene Representation for Robot Simulation and Online State Synchronization`

Project homepage: https://rtdgs.github.io/DGSRSim/

## Core Claims Reflected in the Repository

- Object-level reconstruction: `GaussianModel/` partitions one shared Gaussian scene model into independently addressable object subsets and a retained background subset.
- Online state estimation: `FastSAMRealtime/` turns RGB-D observations into object-level point clouds, applies filtering, and aligns them with asset-derived reference geometry.
- Simulation synchronization: `Simulation/` imports converted assets and writes estimated object states into the corresponding simulation objects.
- Scene mutability: the simulation scene can be composed from a background asset, movable object assets, and robot assets. The runtime watches the binding file for add, remove, and replacement changes, while state-bundle tombstones can deactivate configured objects.

## Evidence Scope

The manuscript reports results for the offline evaluation set and the current online test sequences under the reported implementation configuration. The repository documentation uses the same boundary:

- PSNR, SSIM, and LPIPS describe rendering or visual-consistency quality.
- Pose errors describe registration/state-estimation accuracy under the evaluated sequences.
- The reported stage means sum to approximately 73 ms per processed update. Their 13.7 Hz reciprocal is arithmetic rather than observed throughput; the released path captures at 10 fps, updates the cached object cloud at up to 5 Hz, and triggers registration on demand.
- Fig. B.1 in the manuscript is a supplementary load diagnostic, not a complete per-object, per-frame runtime scaling law.
- The minimal pick demonstration is an interface-chain record, not a task-level reliability metric.

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
| Camera-to-scene configuration | `FastSAMRealtime/configs/calibration/` | Operational frame definition and installation-specific override point |
| Asset scale provenance | `FastSAMRealtime/configs/assets/` | Target PLY and converted USDZ identity, bounds, and hashes |
| Fig. B.1 source values | `docs/figure8_table_derived_diagnostic.csv` | Direct copy of the aggregate entries plotted from manuscript Tables 5 and 6 |

## Implementation and Reproduction Mapping

The manuscript appendix retains scientific parameters and evaluation protocols. Repository-specific entry points and file conventions are documented here instead.

### Offline asset construction

- `GaussianModel/train.py` optimizes one Gaussian scene model and its instance classifier. The Gaussian Grouping classifier head accepts at most 256 labels, including background label 0. The released configuration sets `num_classes` to 256 and uses only the background and active-object label identifiers as targets for each scene. The released main entry uses 10,000 iterations even though the inherited optimization-parameter default is 30,000. Numbered training snapshots are retained only for source provenance and terminate with a pointer to this entry point.
- `GaussianModel/script/train.sh` records the scene source, image scale, output directory, and training configuration used by the command-line workflow.
- `GaussianModel/render.py` loads the selected checkpoint and classifier state, then writes reconstructions, references, object-identifier maps, and feature visualizations for the configured split.
- `third_party/3dgrut_conversion/crop_gaussian_ply.py` and the adjacent conversion scripts provide Gaussian filtering, mesh conversion, USD/USDZ conversion, and collision-proxy preparation.
- DEVA is an optional producer of cross-view masks. The canonical `GaussianModel/train.py` path consumes existing masks and does not import or execute DEVA.

### Online RGB-D observation and registration

- `FastSAMRealtime/rt_seg_strict_align_cut_object_pcd5.py` is the evaluated RGB-D entry point. It acquires Kinect color/depth frames, selects an instance, constructs the CameraSpace cloud, and triggers registration.
- `FastSAMRealtime/utils/kinect_strict_align.py` performs calibrated depth-to-camera and camera-to-color mapping.
- `FastSAMRealtime/utils/registration_async.py` applies filtering, FPFH + RANSAC initialization, GICP refinement, and the fitness gate.
- The evaluated settings are `out_nb=20`, `out_std=1.5`, `src_pre_voxel=0.007`, `tgt_pre_voxel=0.003`, RANSAC sample size 4, at most 100,000 RANSAC iterations, at most 80 GICP iterations, and minimum accepted GICP fitness 0.05.
- `FastSAMRealtime/configs/calibration/` stores `T_scene_from_camera`; installation-specific deployments replace the checked-in identity-frame calibration with a measured rigid transform.

### State writing and simulation synchronization

- `FastSAMRealtime/utils/pose_utils.py` composes the accepted registration, configured camera-to-scene transform, and raw-asset normalization into `A_scene_from_asset_raw`.
- Each observation worker identifies its tracked asset with `DGSRSIM_OBJECT_ID`, writes a compatibility single-object packet, and atomically updates the corresponding entry in the versioned `object_states.json` bundle. Every entry records the similarity transform, scale and centers, asset and calibration hashes, registration diagnostics, timestamp, and validity status. The rigid `.npy` transform remains a compatibility output.
- `Simulation/source/leisaac/leisaac/utils/pose_sync_pipeline.py` consumes active bundle entries, skips unchanged transforms, and resolves object identifiers through `Simulation/configs/object_bindings.example.json`. Inactive tombstones are forwarded to the runtime lifecycle manager. The same binding file is watched for enabled-object additions, removals, and asset-record replacements.
- Independent pose-jump gating, SE(3) smoothing, and automatic continuous retriggering are disabled in the evaluated execution path.

### Metrics and supplementary diagnosis

- `GaussianModel/metrics1.py` implements object-mask, object-complement background, and full-image PSNR, SSIM, and spatial LPIPS aggregation on paired render/reference images. `GaussianModel/metrics.py` is a compatibility wrapper around the same implementation.
- `docs/figure8_table_derived_diagnostic.csv` contains the four object-count PSNR entries from manuscript Table 5 and the four stage means from manuscript Table 6. The plotted cumulative times in Fig. B.1 are 12, 48, 55, and 73 ms.

## Terminology Used Here

The documentation uses "online processing capability" and "online interaction potential" for the reported runtime behavior. It avoids unconditional robot-control, task-reliability, or physical-interaction-success claims because those require separate task-level experiments.
