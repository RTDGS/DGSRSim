# Canonical Paper-Aligned Entry Points

The repository retains upstream and historical numbered variants for source
provenance. Divergent training, RGB-D, orchestration, and simulation snapshots
exit with a pointer to the current entry point instead of running silently.
Only the following files define the public paper-aligned workflow.

| Stage | Canonical entry point |
| --- | --- |
| Multi-view conversion | `GaussianModel/convert.py` |
| Optional DEVA mask preparation | `GaussianModel/script/prepare_pseudo_label0.sh` |
| Shared-field training | `GaussianModel/train.py` through `GaussianModel/script/train.sh` |
| Rendering and class-conditioned outputs | `GaussianModel/render.py` |
| Region-aware PSNR/SSIM/LPIPS | `GaussianModel/metrics1.py` (`metrics.py` is a wrapper) |
| Gaussian filtering for conversion | `third_party/3dgrut_conversion/crop_gaussian_ply.py` |
| RGB-D observation and registration | `FastSAMRealtime/rt_seg_strict_align_cut_object_pcd5.py` |
| State activation/deactivation | `FastSAMRealtime/object_state_control.py` |
| Runtime simulation | `Simulation/scripts/environments/teleoperation/teleop_se3_agent.py` |

Files with numeric suffixes, experimental multi-GPU variants, and inherited
upstream demos are not used to define the manuscript equations, reported
configuration, or evaluation protocol unless this table names them explicitly.
