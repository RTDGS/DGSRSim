#!/usr/bin/env python3
"""
Controlled manual workflow for hinged 3DGRUT assets.

This script intentionally keeps orchestration out of
run_split_3dgrut_hinge_pipeline.py. It performs the higher-level workflow:

1. filter the original Gaussian PLY,
2. open the manual Open3D hinge/cut picker,
3. create split preview PLYs,
4. call run_split_3dgrut_hinge_pipeline.py on the filtered PLY and picked JSON.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], env: dict[str, str]) -> None:
    print("\n[run]", " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter, manually pick a hinge split, then run the 3DGRUT hinged USDZ pipeline.")
    parser.add_argument("input_ply")
    parser.add_argument("--work-dir", default=None, help="Defaults to input_ply parent/manual_hinge_workflow.")
    parser.add_argument("--skip-pick", action="store_true", help="Reuse existing --filtered-ply and --split-json, then only run the 3DGRUT pipeline.")
    parser.add_argument("--filtered-ply", default=None)
    parser.add_argument("--split-json", default=None)

    parser.add_argument("--filter-mode", choices=["cluster-distance", "bbox"], default="cluster-distance")
    parser.add_argument("--filter-min-opacity", type=float, default=0.005)
    parser.add_argument("--filter-max-points", type=int, default=0)
    parser.add_argument("--crop-min", default=None, help="Optional x,y,z lower crop bound for filtering.")
    parser.add_argument("--crop-max", default=None, help="Optional x,y,z upper crop bound for filtering.")
    parser.add_argument("--cluster-opacity", type=float, default=0.7)
    parser.add_argument("--cluster-eps", type=float, default=0.5)
    parser.add_argument("--cluster-min-points", type=int, default=10)
    parser.add_argument("--cluster-include-radius", type=float, default=0.12)

    parser.add_argument("--picker-min-opacity", type=float, default=0.05)
    parser.add_argument("--picker-max-points", type=int, default=300000)
    parser.add_argument("--overlap-width", type=float, default=0.002)
    parser.add_argument("--no-preview", action="store_true")

    parser.add_argument("--lower-limit-deg", type=float, default=0.0)
    parser.add_argument("--upper-limit-deg", type=float, default=90.0)
    parser.add_argument("--fixed-body", choices=["A", "B", "none"], default="A")
    parser.add_argument("--simplify-target-faces", type=int, default=25000)
    parser.add_argument("--mesh-min-opacity", type=float, default=0.16)
    parser.add_argument("--set-invisible", action="store_true")

    parser.add_argument("--cc", default="/usr/bin/gcc-12")
    parser.add_argument("--cxx", default="/usr/bin/g++-12")
    parser.add_argument("--cuda-host-cxx", default="/usr/bin/g++-12")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    input_ply = Path(args.input_ply)
    if not input_ply.is_file():
        raise FileNotFoundError(input_ply)

    work_dir = Path(args.work_dir) if args.work_dir else input_ply.parent / "manual_hinge_workflow"
    work_dir.mkdir(parents=True, exist_ok=True)

    filtered_ply = Path(args.filtered_ply) if args.filtered_ply else work_dir / f"{input_ply.stem}_filtered.ply"
    split_json = Path(args.split_json) if args.split_json else work_dir / "picked_hinge_split.json"
    pipeline_dir = work_dir / "hinge_split"

    env = os.environ.copy()
    env["CC"] = args.cc
    env["CXX"] = args.cxx
    env["CUDAHOSTCXX"] = args.cuda_host_cxx

    if not args.skip_pick:
        pick_cmd = [
            sys.executable,
            str(script_dir / "pick_hinge_split_open3d.py"),
            str(input_ply),
            "--output-dir",
            str(work_dir),
            "--output-json",
            str(split_json),
            "--filter-first",
            "--filtered-ply",
            str(filtered_ply),
            "--filter-mode",
            args.filter_mode,
            "--filter-min-opacity",
            str(args.filter_min_opacity),
            "--filter-max-points",
            str(args.filter_max_points),
            "--cluster-opacity",
            str(args.cluster_opacity),
            "--cluster-eps",
            str(args.cluster_eps),
            "--cluster-min-points",
            str(args.cluster_min_points),
            "--cluster-include-radius",
            str(args.cluster_include_radius),
            "--min-opacity",
            str(args.picker_min_opacity),
            "--max-points",
            str(args.picker_max_points),
            "--overlap-width",
            str(args.overlap_width),
            "--run-split",
        ]
        if args.crop_min:
            pick_cmd.append(f"--crop-min={args.crop_min}")
        if args.crop_max:
            pick_cmd.append(f"--crop-max={args.crop_max}")
        if args.no_preview:
            pick_cmd.append("--no-preview")
        run(pick_cmd, env)
    else:
        if not filtered_ply.is_file():
            raise FileNotFoundError(f"--skip-pick requires existing filtered PLY: {filtered_ply}")
        if not split_json.is_file():
            raise FileNotFoundError(f"--skip-pick requires existing split JSON: {split_json}")

    pipeline_cmd = [
        sys.executable,
        str(script_dir / "run_split_3dgrut_hinge_pipeline.py"),
        str(filtered_ply),
        "--out-dir",
        str(pipeline_dir),
        "--split-json",
        str(split_json),
        "--fixed-body",
        args.fixed_body,
        "--lower-limit-deg",
        str(args.lower_limit_deg),
        "--upper-limit-deg",
        str(args.upper_limit_deg),
        "--simplify-target-faces",
        str(args.simplify_target_faces),
        "--mesh-min-opacity",
        str(args.mesh_min_opacity),
        "--cc",
        args.cc,
        "--cxx",
        args.cxx,
        "--cuda-host-cxx",
        args.cuda_host_cxx,
    ]
    if args.set_invisible:
        pipeline_cmd.append("--set-invisible")
    run(pipeline_cmd, env)

    print("\n== Done ==")
    print(f"[filtered_ply] {filtered_ply.resolve()}")
    print(f"[split_json]   {split_json.resolve()}")
    print(f"[preview_dir]   {(work_dir / 'split_preview').resolve()}")
    print(f"[part_a_usdz]   {(pipeline_dir / 'part_a' / 'part_a_with_collision.usdz').resolve()}")
    print(f"[part_b_usdz]   {(pipeline_dir / 'part_b' / 'part_b_with_collision.usdz').resolve()}")
    print(f"[assembly]      {(pipeline_dir / 'hinged_assembly.usda').resolve()}")


if __name__ == "__main__":
    main()
