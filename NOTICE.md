# DGSRSim License and Distribution Notice

DGSRSim is a mixed-license research release. The root `LICENSE` contains the
GNU Affero General Public License version 3 (AGPL-3.0). DGSRSim-specific source
code and documentation that do not carry another copyright or license notice
are released under AGPL-3.0-only.

The root license does not relicense third-party material. Files derived from or
distributed with third-party projects remain governed by their original terms.
The corresponding notices and license texts are listed in
`THIRD_PARTY_NOTICES.md` and `third_party_licenses/`.

In particular:

- `FastSAMRealtime/fastsam/` and `FastSAMRealtime/ultralytics/` retain the
  FastSAM/Ultralytics AGPL-3.0 terms.
- Gaussian Grouping, 3DGRUT conversion code, LaMA, LeIsaac, Grounded Segment
  Anything, GroundingDINO, Segment Anything, and DEVA retain the terms stated
  in `THIRD_PARTY_NOTICES.md`.
- `GaussianModel/submodules/diff-gaussian-rasterization/` retains the Inria/MPII
  Gaussian-Splatting license. That component is restricted to non-commercial
  research or evaluation unless its licensors grant additional permission.
- Model weights, raw captures, trained Gaussian assets, simulation assets, and
  benchmark datasets are not licensed by the root `LICENSE` unless a separate
  license accompanies the individual file.

No single license grants rights to every file in this repository. Users are
responsible for checking the notice attached to each component and for obtaining
any permissions required by their intended use. This notice describes the
repository layout and is not legal advice.
