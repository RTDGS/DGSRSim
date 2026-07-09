# Paper Alignment Notes

This repository follows the manuscript wording of:

`DGSRSim: Object-Level Decoupled 3D Gaussian Assets for Robot Simulation and Online State Synchronization`

Project homepage: https://rtdgs.github.io/DGSRSim/

## Core Claims Reflected in the Repository

- Object-level reconstruction: `GaussianModel/` builds independently managed object Gaussian assets and a separate background field in a shared world frame.
- Online state estimation: `FastSAMRealtime/` turns RGB-D observations into object-level point clouds, applies filtering, and aligns them with asset-derived reference geometry.
- Simulation synchronization: `Simulation/` imports converted assets and writes estimated object states into the corresponding simulation objects.
- Scene mutability: the simulation scene can be composed from a background asset, movable object assets, and robot assets; object assets can be added, removed, replaced, or rearranged at the asset level.

## Evidence Scope

The manuscript reports results for the offline evaluation set and the current online test sequences under the reported implementation configuration. The repository documentation uses the same boundary:

- PSNR, SSIM, and LPIPS describe rendering or visual-consistency quality.
- Pose errors describe registration/state-estimation accuracy under the evaluated sequences.
- The average stage-wise runtime of approximately 73 ms/frame supports online processing capability or online interaction potential.
- Fig. 8 in the manuscript is a supplementary load diagnostic, not a complete per-object, per-frame runtime scaling law.
- The minimal pick demonstration is an interface-chain record, not a task-level reliability metric.

Robot task reliability, physical contact stability, collision penetration, grasp success rate, and long-horizon closed-loop manipulation require independent task-level protocols and are outside the current evaluation scope.

## Repository-to-Paper Mapping

| Paper component | Repository location | Role |
| --- | --- | --- |
| Offline object/background asset construction | `GaussianModel/` | Object Gaussian assets, background field, rendering, and editing workflow |
| Online RGB-D object observation | `FastSAMRealtime/` | RGB-D alignment, segmentation, object point-cloud cropping, filtering |
| Coarse-to-fine state estimation | `FastSAMRealtime/` | Point-cloud registration and state-estimation scripts |
| Asset conversion | `third_party/3dgrut_conversion/` | Gaussian PLY to mesh/USD/USDZ/collision conversion |
| Simulation synchronization | `Simulation/` | LeIsaac/IsaacLab import, object-state writing, scene composition |
| External large files | `docs/LARGE_FILES.md` | Placement paths for weights, raw data, point clouds, and simulation assets |

## Terminology Used Here

The documentation uses "online processing capability" and "online interaction potential" for the reported runtime behavior. It avoids unconditional robot-control, task-reliability, or physical-interaction-success claims because those require separate task-level experiments.
