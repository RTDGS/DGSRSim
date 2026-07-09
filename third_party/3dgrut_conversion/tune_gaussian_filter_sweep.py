#!/usr/bin/env python3
"""
Create a controlled sequence of filtered Gaussian PLYs.

The goal is to tune filtering from "almost unchanged" to progressively
stronger, with a report showing how many points each stage keeps. Use the
chosen output PLY as the collision/mesh input, while preserving the original
PLY for the visual USDZ when appearance matters.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from ply_to_mesh import read_vertex_ply


DEFAULT_CLUSTER_STAGES = [
    {
        "name": "05_cluster_very_light",
        "min_opacity": 0.001,
        "cluster_opacity": 0.10,
        "cluster_eps": 1.20,
        "cluster_include_radius": 2.00,
    },
    {
        "name": "06_cluster_light",
        "min_opacity": 0.003,
        "cluster_opacity": 0.20,
        "cluster_eps": 1.00,
        "cluster_include_radius": 1.00,
    },
    {
        "name": "07_cluster_medium",
        "min_opacity": 0.005,
        "cluster_opacity": 0.35,
        "cluster_eps": 0.80,
        "cluster_include_radius": 0.50,
    },
    {
        "name": "08_cluster_stronger",
        "min_opacity": 0.010,
        "cluster_opacity": 0.50,
        "cluster_eps": 0.60,
        "cluster_include_radius": 0.25,
    },
]


def count_vertices(path: Path) -> int:
    vertices, _ = read_vertex_ply(path)
    return int(len(vertices))


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    print("\n[run]", " ".join(str(part) for part in cmd), flush=True)
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def fmt_float(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def parse_cluster_stage(text: str) -> dict:
    parts = [part.strip() for part in text.split(":")]
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            "Expected name:min_opacity:cluster_opacity:cluster_eps:cluster_include_radius"
        )
    name, min_opacity, cluster_opacity, cluster_eps, include_radius = parts
    return {
        "name": name,
        "min_opacity": float(min_opacity),
        "cluster_opacity": float(cluster_opacity),
        "cluster_eps": float(cluster_eps),
        "cluster_include_radius": float(include_radius),
    }


def parse_float_csv(text: str) -> list[float]:
    if not text.strip():
        return []
    return [float(part.strip()) for part in text.split(",")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate gradual Gaussian PLY filter variants for tuning.")
    parser.add_argument("input_ply")
    parser.add_argument("--out-dir", default=None, help="Defaults to input parent/filter_sweep.")
    parser.add_argument("--crop-script", default=None, help="Defaults to sibling crop_gaussian_ply.py.")
    parser.add_argument("--python-bin", default=sys.executable)

    parser.add_argument(
        "--opacity-stages",
        default="0,0.001,0.003,0.005",
        help="Comma-separated min-opacity values for opacity-only stages.",
    )
    parser.add_argument(
        "--cluster-stage",
        action="append",
        type=parse_cluster_stage,
        default=None,
        help="Custom cluster stage: name:min_opacity:cluster_opacity:cluster_eps:cluster_include_radius. Can repeat.",
    )
    parser.add_argument("--no-default-cluster-stages", action="store_true")
    parser.add_argument("--max-points", type=int, default=0)
    parser.add_argument("--crop-min", default=None, help="Optional x,y,z lower bound. Use --crop-min=-1,0,0 for negative values.")
    parser.add_argument("--crop-max", default=None, help="Optional x,y,z upper bound. Use --crop-max=1,2,3 for negative values.")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_ply = Path(args.input_ply)
    if not input_ply.is_file():
        raise FileNotFoundError(input_ply)

    script_dir = Path(__file__).resolve().parent
    crop_script = Path(args.crop_script) if args.crop_script else script_dir / "crop_gaussian_ply.py"
    if not crop_script.is_file():
        raise FileNotFoundError(crop_script)

    out_dir = Path(args.out_dir) if args.out_dir else input_ply.parent / "filter_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    original_count = count_vertices(input_ply)
    report: list[dict] = []

    original_out = out_dir / "00_original_unfiltered.ply"
    shutil.copy2(input_ply, original_out)
    report.append(
        {
            "stage": "00_original_unfiltered",
            "mode": "copy",
            "status": "ok",
            "output": str(original_out),
            "points": original_count,
            "kept_ratio": 1.0,
        }
    )

    stages = []
    for i, min_opacity in enumerate(parse_float_csv(args.opacity_stages), start=1):
        stages.append(
            {
                "name": f"{i:02d}_opacity_{fmt_float(min_opacity)}",
                "mode": "opacity-only",
                "min_opacity": float(min_opacity),
            }
        )

    if not args.no_default_cluster_stages:
        stages.extend(DEFAULT_CLUSTER_STAGES)
    if args.cluster_stage:
        stages.extend(args.cluster_stage)

    for stage in stages:
        name = stage["name"]
        output = out_dir / f"{name}.ply"
        if stage.get("mode") == "opacity-only":
            cmd = [
                args.python_bin,
                str(crop_script),
                str(input_ply),
                "--output",
                str(output),
                "--filter-mode",
                "bbox",
                "--no-auto-crop-largest-cluster",
                "--min-opacity",
                str(stage["min_opacity"]),
                "--max-points",
                str(args.max_points),
            ]
        else:
            cmd = [
                args.python_bin,
                str(crop_script),
                str(input_ply),
                "--output",
                str(output),
                "--filter-mode",
                "cluster-distance",
                "--min-opacity",
                str(stage["min_opacity"]),
                "--cluster-opacity",
                str(stage["cluster_opacity"]),
                "--cluster-eps",
                str(stage["cluster_eps"]),
                "--cluster-include-radius",
                str(stage["cluster_include_radius"]),
                "--max-points",
                str(args.max_points),
            ]
        if args.crop_min:
            cmd.append(f"--crop-min={args.crop_min}")
        if args.crop_max:
            cmd.append(f"--crop-max={args.crop_max}")

        result = run(cmd)
        record = {
            "stage": name,
            "mode": stage.get("mode", "cluster-distance"),
            "status": "ok" if result.returncode == 0 else "error",
            "output": str(output),
            "stdout": result.stdout,
            **{key: value for key, value in stage.items() if key != "name"},
        }

        if result.returncode == 0 and output.is_file():
            points = count_vertices(output)
            record["points"] = points
            record["kept_ratio"] = points / max(original_count, 1)
        else:
            record["points"] = None
            record["kept_ratio"] = None
            print(result.stdout)
            if args.stop_on_error:
                raise RuntimeError(f"Stage failed: {name}")

        report.append(record)

    report_path = out_dir / "filter_sweep_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n== Filter Sweep Report ==")
    print(f"{'stage':<28} {'mode':<17} {'points':>12} {'kept':>9} output")
    for item in report:
        points = item.get("points")
        ratio = item.get("kept_ratio")
        points_text = "-" if points is None else str(points)
        ratio_text = "-" if ratio is None else f"{ratio * 100:6.2f}%"
        print(f"{item['stage']:<28} {item['mode']:<17} {points_text:>12} {ratio_text:>9} {item['output']}")
    print(f"\n[report] {report_path.resolve()}")
    print("\nPick the weakest stage that removes the floating junk, then use that PLY as your collision/filter input.")


if __name__ == "__main__":
    main()
