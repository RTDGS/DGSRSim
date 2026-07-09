# DGSRSim

This repository is the release copy of the code for the paper "DGSRSim: Object-Level Decoupled 3D Gaussian Assets for Robot Simulation and Online State Synchronization". It keeps the source files required to reproduce the main workflow and removes nested Git metadata, historical third-party README files, model weights, point clouds, training outputs, simulation assets, and internal collaboration files.

This release copy was reorganized from the original `code/` directory. The original source directory was not modified. `AGENTS.md` is an internal collaboration instruction file and is intentionally not included in this GitHub repository.

Project page: https://rtdgs.github.io/DGSRSim/

## Paper-Aligned Scope

DGSRSim provides an implementation path for object-level reconstruction, online object-state estimation, and simulation-side state synchronization. The repository is organized around two visible ideas:

- Real/virtual decoupled reconstruction: real RGB-D and multi-view observations are converted into independently managed object Gaussian assets and a separate background field in a shared world frame.
- Real-to-simulation synchronization: online RGB-D observations produce object-level point clouds, the recovered object state is written to the corresponding simulation asset, and the virtual scene can update only the objects whose states change.

The simulation scene is asset-composable. Object assets can be added, removed, replaced, or rearranged, and robot assets such as a manipulator can be introduced into the same scene. These capabilities support object-level state tracking and interaction studies, while the repository does not claim robot task reliability, grasp success rate, contact stability, or long-horizon closed-loop control performance from the current evidence alone.

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

`GaussianModel/` is the core module of this project. It generates object-level decoupled Gaussian Splatting scene representations from captured data and supports background/object separation, pseudo-label preparation, training, rendering, and object editing. `GaussianModel/Usage` records the original step-by-step workflow used after dataset capture to generate independent Gaussian models.

`FastSAMRealtime/` handles online observation-side processing, including Kinect image acquisition, strict RGB-D alignment, FastSAM segmentation, object point-cloud cropping, filtering, and point-cloud registration. This module supports the online object-state estimation pipeline discussed in the paper. A stage-wise runtime of approximately 73 ms/frame corresponds to about 13-14 FPS under the reported implementation configuration; this repository therefore describes the system as having online processing capability or online interaction potential, rather than making an unconditional robot-control claim.

`third_party/3dgrut_conversion/` keeps only the conversion-chain code used by this project. It converts Gaussian PLY outputs from `GaussianModel` into `mesh.ply` and then into simulation-side USD/USDZ or collision assets. This conversion code is adapted from the export and asset-conversion workflow of the 3DGRUT / 3D Gaussian Ray Tracing project. The original 3DGRUT README, full project, and unrelated files are not included. Citation details are provided in `THIRD_PARTY_NOTICES.md`.

`Simulation/` is reorganized from LeIsaac/IsaacLab simulation-side code. It imports the converted simulation assets and synchronizes object poses or states estimated online into the simulation environment. Large USD, USDZ, PLY, and ZIP assets are not stored in GitHub; after downloading them, place them back under the corresponding paths in `Simulation/assets/`.

## Evidence Boundary

The paper evaluates reconstruction quality, pose estimation, and visual synchronization under unified protocols. The current wording used by the paper and this repository is:

- Tables for reconstruction cover the offline evaluation set used in the manuscript.
- Online state estimation and synchronization are reported for the current online test sequences and implementation configuration.
- PSNR, SSIM, and LPIPS measure rendering or visual-consistency quality; they are not physical interaction metrics.
- The minimal pick demonstration is an interface-chain record, not a robot task-reliability experiment.
- Fig. 8 in the manuscript is treated as a supplementary load diagnostic, not as a complete per-object, per-frame runtime scaling law.

## Large Files and Data

This GitHub repository contains only source code, configuration files, license files, and documentation. The following files are excluded from GitHub:

- SAM, DEVA, GroundingDINO, FastSAM, LaMA, and related model weights;
- GaussianModel training outputs, checkpoints, point clouds, and compressed datasets;
- PLY point-cloud outputs generated by the FastSAM online pipeline;
- LeIsaac/IsaacLab scenes, robots, USD/USDZ, FBX, GLB, and other simulation assets;
- large videos, high-resolution image dumps, and internal QA artifacts. Small qualitative homepage GIFs are retained under `static/results/` to show real-to-simulation synchronization and composable scene variants.

See `docs/LARGE_FILES.md` for the external large-file list and the expected placement paths after download. Large assets, raw sensor captures, checkpoints, and third-party weights are provided through the supplementary/reviewer package or source-specific acquisition paths when licenses and storage constraints allow.

## Installation

Use separate environments for different modules when possible. The Gaussian training environment, Kinect online-processing environment, and Isaac Sim/IsaacLab simulation environment have different dependency constraints.

```bash
# GaussianModel
pip install -r requirements/gaussian_model.txt
pip install -e GaussianModel/submodules/diff-gaussian-rasterization
pip install -e GaussianModel/submodules/simple-knn

# FastSAMRealtime
pip install -r requirements/fastsam_realtime.txt

# Conversion and simulation utilities
pip install -r requirements/conversion_simulation.txt
```

CUDA, PyTorch, Isaac Sim/IsaacLab, Kinect SDK, and GPU driver versions should be matched to the local machine. Isaac Sim and IsaacLab are not installed through the requirements files above.

## Typical Workflow

1. Prepare captured data, then follow the order recorded in `GaussianModel/Usage` for data conversion, pseudo-label preparation, and training.

```bash
cd GaussianModel
python convert.py -s data/<scene_name>
bash script/prepare_pseudo_label0.sh <scene_name> <gpu_id>
bash script/train.sh <scene_name> <gpu_id>
```

2. Use `GaussianModel` to output object-level and background/object-decoupled Gaussian PLY assets in a shared world frame.

3. Use `FastSAMRealtime/` for strict Kinect RGB-D alignment, segmentation, object point-cloud cropping, registration-quality filtering, and coarse-to-fine registration.

4. Use `third_party/3dgrut_conversion/` to convert Gaussian PLY files into simulation-side mesh or USD/USDZ assets. Example:

```bash
python third_party/3dgrut_conversion/gaussian_ply_to_3dgrut_mesh_ply.py \
  --input path/to/point_cloud.ply \
  --output path/to/mesh.ply
```

To use the original 3DGRUT export path `python -m threedgrut.export.scripts.ply_to_usd`, prepare a complete 3DGRUT runtime environment separately.

5. Place the externally downloaded simulation assets under `Simulation/assets/`, then run the LeIsaac/IsaacLab task scripts for simulation import and object-state synchronization. The scene can be composed from a background asset, movable object assets, and robot assets; object addition, removal, and replacement are handled at the asset level.

## Third-Party Code

This repository contains reorganized third-party research code and selected local adaptations. The original project README files are not kept as top-level documentation in this release copy. Citations, licenses, and source notes are consolidated in `THIRD_PARTY_NOTICES.md` and `third_party_licenses/`.

Important license note: `FastSAMRealtime/` retains FastSAM and related Ultralytics code. The original FastSAM project is licensed under AGPL-3.0. Before final public distribution, confirm that the root project license is compatible with AGPL-3.0 and follow all third-party license requirements.
