# DGSRSim

This repository is the public implementation companion to the paper "DGSRSim: An Object-Level Decoupled 3D Gaussian Scene Representation for Robot Simulation and Online State Synchronization". It exposes the paper-aligned algorithms, interfaces, configuration schemas, and selected qualitative media. Model weights, raw captures, trained assets, complete evaluation records, and simulation assets are not part of the public repository during manuscript review.

This release copy was reorganized from the original `code/` directory. The original source directory was not modified. `AGENTS.md` is an internal collaboration instruction file and is intentionally not included in this GitHub repository.

Project page: https://rtdgs.github.io/DGSRSim/

## Paper-Aligned Scope

DGSRSim provides an implementation path for object-level reconstruction, online object-state estimation, and simulation-side state synchronization. The repository is organized around two visible ideas:

- Real/virtual decoupled reconstruction: one shared Gaussian scene model is partitioned into independently addressable object subsets and a retained background subset in the same reconstruction frame.
- Real-to-simulation synchronization: online RGB-D observations produce object-level point clouds, the recovered object state is written to the corresponding simulation asset, and the virtual scene can update only the objects whose states change.

The simulation scene is asset-composable. Locally available object assets can be added, removed, replaced, or rearranged by updating the binding configuration while the simulator is running. State records can also activate or deactivate configured objects. Robot assets such as a manipulator can be introduced into the same scene. These capabilities support object-level state tracking and interaction studies, while the repository does not claim robot task reliability, grasp success rate, contact stability, or long-horizon closed-loop control performance from the current evidence alone.

For the exact paper/repository correspondence, see `docs/PAPER_ALIGNMENT.md`.

## Repository Structure

```text
GaussianModel/                  Main code for decoupled 3D Gaussian Splatting model training, rendering, and object editing
FastSAMRealtime/                Kinect/RGB-D acquisition, FastSAM segmentation, object point-cloud cropping, and online registration
third_party/3dgrut_conversion/  Conversion scripts from Gaussian PLY to mesh.ply, USD/USDZ, and collision assets
Simulation/                     LeIsaac/IsaacLab simulation import, scene loading, and object-state synchronization code
requirements/                   Module-specific dependency references
docs/                           Release organization notes and external large-file list
third_party_licenses/           License texts for retained third-party source code
```

## Modules

`GaussianModel/` is the core module of this project. It trains one shared Gaussian scene model with instance supervision and a Gaussian Grouping classifier head. The classifier supports 2 to 256 output labels, including background label 0; the released training configuration uses 256 outputs, while each scene assigns targets only to its active object labels and the background. Object and background assets are obtained by partitioning the shared Gaussian space according to learned instance ownership. The module supports mask preparation, training, partitioned rendering, asset extraction, and object editing. `docs/CANONICAL_ENTRYPOINTS.md` identifies the paper-aligned entry points; numbered historical variants are retained only for provenance and are not canonical evaluation commands.

`FastSAMRealtime/` handles online observation-side processing, including Kinect image acquisition, strict RGB-D alignment, FastSAM segmentation, object point-cloud cropping, filtering, and point-cloud registration. This module supports the online object-state estimation pipeline discussed in the paper. The reported stage means sum to approximately 73 ms per processed update. Their arithmetic reciprocal is about 13.7 Hz, but it is not measured loop throughput: the released script captures at 10 fps, updates the cached object cloud at up to 5 Hz, and triggers the selected-object registration on demand. The repository therefore describes online processing capability or online interaction potential rather than unconditional robot-control performance.

The released registration path applies FPFH + RANSAC initialization followed by GICP refinement. The active `rt_seg_strict_align_cut_object_pcd5.py` configuration uses `out_nb=20`, `out_std=1.5`, `src_pre_voxel=0.007`, `tgt_pre_voxel=0.003`, and a GICP fitness threshold of `0.05`. A state is written only when the returned transform is a finite, invertible `4 x 4` matrix and the fitness threshold is met; the released path does not implement an additional pose-jump gate, SE(3) temporal smoother, or automatic continuous retriggering.

The observed Kinect point cloud is expressed in CameraSpace. The evaluated configuration defines the metric simulation scene frame to coincide with CameraSpace, so the checked-in operational calibration `FastSAMRealtime/configs/calibration/kinect_camera_space_to_scene.json` is the identity transform by frame definition. This file is loaded by default and records its provenance explicitly; `DGSRSIM_T_SCENE_FROM_CAMERA` can override it with a measured `4 x 4` rigid calibration for another setup.

AABB normalization is carried through the full state chain. Registration constructs `Q_normalized = c_source + s * (Q_raw - c_target)`, records the corresponding similarity matrix, and uses the normalized target in both RANSAC and GICP. Each producer writes a legacy single-object `object_state.json` packet and atomically updates its entry in `object_states.json`. Both packets include the rigid normalized-target pose, `scale_raw_to_normalized`, normalization centers, and the complete `A_scene_from_asset_raw` similarity transform. The simulation-side `ObjectStateFileStream` consumes all active entries and applies the same per-object scale to each raw converted asset; the legacy `T_tgt_to_scene.npy` file remains only as a rigid-pose fallback.

`third_party/3dgrut_conversion/` keeps only the conversion-chain code used by this project. It converts Gaussian PLY outputs from `GaussianModel` into `mesh.ply` and then into simulation-side USD/USDZ or collision assets. This conversion code is adapted from the export and asset-conversion workflow of the 3DGRUT / 3D Gaussian Ray Tracing project. The original 3DGRUT README, full project, and unrelated files are not included. Citation details are provided in `THIRD_PARTY_NOTICES.md`.

`Simulation/` is reorganized from LeIsaac/IsaacLab simulation-side code. It imports converted assets and synchronizes object states estimated online into the simulation environment. The active `teleop_se3_agent.py` path reads the scale-preserving multi-object bundle, resolves every object identifier through `Simulation/configs/object_bindings.example.json`, optionally spawns all enabled runtime assets, and writes each accepted state independently. Converted objects remain in their raw shared-frame coordinates, and scene placement removes the background display scale before composing each object world transform. Large USD, USDZ, PLY, and ZIP assets are not stored in GitHub; after downloading them, place them back under the corresponding paths in `Simulation/assets/`.

## Evidence Boundary

The paper evaluates reconstruction quality, pose estimation, and visual synchronization under unified protocols. The current wording used by the paper and this repository is:

- Tables for reconstruction cover the offline evaluation set used in the manuscript.
- Online state estimation and synchronization are reported for the current online test sequences and implementation configuration.
- PSNR, SSIM, and LPIPS measure rendering or visual-consistency quality; they are not physical interaction metrics.
- The minimal pick demonstration is an interface-chain record, not a robot task-reliability experiment.
- Fig. B.1 in the manuscript is treated as a supplementary load diagnostic, not as a complete per-object, per-frame runtime scaling law.

The table-derived Fig. B.1 values are recorded in `docs/figure8_table_derived_diagnostic.csv`.

## Large Files and Data

This GitHub repository contains only source code, configuration files, license files, and documentation. The following files are excluded from GitHub:

- SAM, DEVA, GroundingDINO, FastSAM, LaMA, and related model weights;
- GaussianModel training outputs, checkpoints, point clouds, and compressed datasets;
- PLY point-cloud outputs generated by the FastSAM online pipeline;
- LeIsaac/IsaacLab scenes, robots, USD/USDZ, FBX, GLB, and other simulation assets;
- large videos, high-resolution image dumps, and internal QA artifacts. Small qualitative homepage GIFs are retained under `static/results/` to show real-to-simulation synchronization and composable scene variants.

See `docs/LARGE_FILES.md` for the non-public large-file inventory and the local placement conventions. During manuscript review, this public repository does not promise or expose a complete asset, checkpoint, raw-data, or evaluation archive. Authorized supplementary or reviewer material is handled separately from the public project page.

## Installation

Use separate environments for different modules when possible. The Gaussian training environment, Kinect online-processing environment, and Isaac Sim/IsaacLab simulation environment have different dependency constraints.

```bash
# GaussianModel reference environment
conda env create -f requirements/pytorch24_reference.yml
conda run -n dgsrsim-pytorch24-reference pip install -r requirements/gaussian_model.txt
conda run -n dgsrsim-pytorch24-reference pip install -e GaussianModel/submodules/diff-gaussian-rasterization
conda run -n dgsrsim-pytorch24-reference pip install -e GaussianModel/submodules/simple-knn

# FastSAMRealtime
pip install -r requirements/fastsam_realtime.txt

# Conversion and simulation utilities
pip install -r requirements/conversion_simulation.txt
```

The paper records PyTorch 2.4 and Kinect SDK 2.0. `requirements/paper_environment.md` separates those recorded facts from the current reference environment. Exact historical CUDA, driver, and auxiliary-package patch identifiers were not retained and are not reconstructed. The module requirement files therefore describe direct dependencies rather than claiming a byte-identical historical lock. Isaac Sim and IsaacLab are installed separately; the retained LeIsaac metadata targets Isaac Sim 5.1.0 and IsaacLab 2.3.0.

## Typical Workflow

1. Prepare captured data and cross-view instance masks, then run conversion and training. `train.py` consumes masks from `data/<scene_name>/object_mask`; it does not import DEVA.

```bash
cd GaussianModel
python convert.py -s data/<scene_name>
bash script/train.sh <scene_name> <image_scale>
```

DEVA is an optional mask-preparation helper rather than a dependency of the training objective. When masks are not supplied by the dataset or another annotation tool, the following helper can populate `object_mask` before training:

```bash
bash script/prepare_pseudo_label0.sh <scene_name> <image_scale>
```

2. Use `GaussianModel` to output object-level and background/object-decoupled Gaussian PLY assets in a shared world frame.

3. Use `FastSAMRealtime/` for strict Kinect RGB-D alignment, segmentation, object point-cloud cropping, registration-quality filtering, and coarse-to-fine registration. The default calibration can be overridden for another installation:

```bash
set DGSRSIM_T_SCENE_FROM_CAMERA=path\to\measured_camera_to_scene.json
python FastSAMRealtime\rt_seg_strict_align_cut_object_pcd5.py
```

4. Use `third_party/3dgrut_conversion/` to convert Gaussian PLY files into simulation-side mesh or USD/USDZ assets. Example:

```bash
python third_party/3dgrut_conversion/gaussian_ply_to_3dgrut_mesh_ply.py \
  --input path/to/point_cloud.ply \
  --output path/to/mesh.ply
```

To use the original 3DGRUT export path `python -m threedgrut.export.scripts.ply_to_usd`, prepare a complete 3DGRUT runtime environment separately.

5. Place locally authorized simulation assets under `Simulation/assets/`, declare each object ID, prim path, and optional spawn record in `Simulation/configs/object_bindings.example.json`, then run `Simulation/scripts/environments/teleoperation/teleop_se3_agent.py`. It reads `FastSAMRealtime/rt_ply_out/object_states.json` by default and uses `T_tgt_to_scene.npy` only as a legacy fallback. The binding file is watched at runtime: enabling or adding a complete spawn record adds an object, disabling or deleting a record removes it, and changing its asset record replaces it. `FastSAMRealtime/object_state_control.py` activates or deactivates retained state records. Only changed transforms are written to USD.

6. Evaluate rendered appearance with an explicit region. Binary masks may use any nonzero value; `--mask_label` selects one grayscale instance identifier from a label map. Object and background commands use the same masks, with the latter evaluating their complement:

```bash
python metrics1.py -r <renders> -g <references> --mask_dir <masks> --region object --output_json object_metrics.json
python metrics1.py -r <renders> -g <references> --mask_dir <masks> --region background --output_json background_metrics.json
```

PSNR is normalized by selected pixels, SSIM is averaged over the selected SSIM-map locations, and spatial LPIPS is aggregated only over the selected region. `metrics.py` forwards to the same implementation for compatibility.

## Third-Party Code

This repository contains reorganized third-party research code and selected local adaptations. The original project README files are not kept as top-level documentation in this release copy. The root `LICENSE` applies only to DGSRSim-specific material that carries no different notice. Third-party citations, licenses, exceptions, and source notes are consolidated in `NOTICE.md`, `THIRD_PARTY_NOTICES.md`, and `third_party_licenses/`.

`GaussianModel/submodules/diff-gaussian-rasterization/` retains a non-commercial research license. The root AGPL-3.0 license does not override that restriction or any other third-party term.
