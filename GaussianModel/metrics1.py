"""Evaluate rendered appearance in explicit object, background, or full regions."""

from __future__ import annotations

import json
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import lpips
import numpy as np
import torch
import torchvision.transforms.functional as tf
from PIL import Image
from tqdm import tqdm

from utils.region_metrics import masked_psnr, masked_spatial_mean, masked_ssim


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class EvaluationSample:
    name: str
    render: torch.Tensor
    target: torch.Tensor
    region_mask: torch.Tensor


def _image_files(path: Path) -> list[Path]:
    return sorted(
        candidate
        for candidate in path.iterdir()
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES
    )


def _matching_file(directory: Path, image_path: Path) -> Path:
    exact = directory / image_path.name
    if exact.exists():
        return exact
    matches = [candidate for candidate in _image_files(directory) if candidate.stem == image_path.stem]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one file for stem {image_path.stem!r} in {directory}, found {len(matches)}"
        )
    return matches[0]


def _load_rgb(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        return tf.to_tensor(image.convert("RGB")).unsqueeze(0).to(device)


def _load_region_mask(
    mask_path: Path,
    image_shape: tuple[int, int],
    region: str,
    mask_label: int | None,
    device: torch.device,
) -> torch.Tensor:
    with Image.open(mask_path) as image:
        values = np.asarray(image.convert("L"), dtype=np.uint8)
    if values.shape != image_shape:
        raise ValueError(
            f"mask {mask_path} has shape {values.shape}; expected {image_shape}. "
            "Evaluation masks are not resized implicitly."
        )
    selected = values == mask_label if mask_label is not None else values != 0
    if region == "background":
        selected = ~selected
    mask = torch.from_numpy(selected.copy()).unsqueeze(0).unsqueeze(0).to(device)
    if not bool(mask.any()):
        raise ValueError(f"selected {region} region is empty in {mask_path}")
    return mask


def load_samples(
    renders_dir: str | Path,
    gt_dir: str | Path,
    mask_dir: str | Path | None,
    region: str,
    mask_label: int | None,
    device: torch.device,
) -> list[EvaluationSample]:
    renders_path = Path(renders_dir)
    targets_path = Path(gt_dir)
    masks_path = Path(mask_dir) if mask_dir else None
    if region != "full" and masks_path is None:
        raise ValueError(f"--mask_dir is required for region={region!r}")

    samples: list[EvaluationSample] = []
    for render_path in _image_files(renders_path):
        target_path = _matching_file(targets_path, render_path)
        render = _load_rgb(render_path, device)
        target = _load_rgb(target_path, device)
        if render.shape != target.shape:
            raise ValueError(
                f"render and target differ for {render_path.name}: {render.shape} vs {target.shape}"
            )
        if region == "full":
            mask = torch.ones(
                (1, 1, render.shape[-2], render.shape[-1]),
                dtype=torch.bool,
                device=device,
            )
        else:
            mask_path = _matching_file(masks_path, render_path)
            mask = _load_region_mask(
                mask_path,
                (render.shape[-2], render.shape[-1]),
                region,
                mask_label,
                device,
            )
        samples.append(EvaluationSample(render_path.name, render, target, mask))
    return samples


def compute_metrics(
    samples: Iterable[EvaluationSample],
    lpips_fn: torch.nn.Module,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    per_view: dict[str, dict[str, float]] = {}
    for sample in tqdm(list(samples), desc="Metric evaluation progress"):
        with torch.no_grad():
            psnr_value = masked_psnr(sample.render, sample.target, sample.region_mask)[0]
            ssim_value = masked_ssim(sample.render, sample.target, sample.region_mask)[0]
            lpips_map = lpips_fn(sample.render * 2.0 - 1.0, sample.target * 2.0 - 1.0)
            lpips_value = masked_spatial_mean(lpips_map, sample.region_mask)[0]
        per_view[sample.name] = {
            "PSNR": float(psnr_value.item()),
            "SSIM": float(ssim_value.item()),
            "LPIPS": float(lpips_value.item()),
        }
    if not per_view:
        raise ValueError("no aligned render/reference image pairs were found")
    summary = {
        metric: float(np.mean([values[metric] for values in per_view.values()]))
        for metric in ("PSNR", "SSIM", "LPIPS")
    }
    return summary, per_view


def _evaluate_directory(
    renders_dir: Path,
    gt_dir: Path,
    mask_dir: Path | None,
    args: Namespace,
    lpips_fn: torch.nn.Module,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    samples = load_samples(
        renders_dir,
        gt_dir,
        mask_dir,
        args.region,
        args.mask_label,
        args.device,
    )
    return compute_metrics(samples, lpips_fn)


def evaluate(args: Namespace) -> None:
    args.device = torch.device(args.device)
    if args.device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; pass --device cpu for CPU evaluation")
    lpips_fn = lpips.LPIPS(net=args.lpips_net, spatial=True).to(args.device).eval()

    if args.renders_dir and args.gt_dir:
        summary, per_view = _evaluate_directory(
            Path(args.renders_dir),
            Path(args.gt_dir),
            Path(args.mask_dir) if args.mask_dir else None,
            args,
            lpips_fn,
        )
        payload = {
            "region": args.region,
            "mask_label": args.mask_label,
            "summary": summary,
            "per_view": per_view,
        }
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return

    if not args.model_paths:
        raise ValueError("provide --model_paths or both --renders_dir and --gt_dir")

    all_results = {}
    for scene_value in args.model_paths:
        scene_dir = Path(scene_value)
        scene_results = {}
        for method_dir in sorted((scene_dir / "test").iterdir()):
            if not method_dir.is_dir():
                continue
            mask_dir = method_dir / args.mask_subdir if args.region != "full" else None
            summary, per_view = _evaluate_directory(
                method_dir / "renders",
                method_dir / "gt",
                mask_dir,
                args,
                lpips_fn,
            )
            scene_results[method_dir.name] = {"summary": summary, "per_view": per_view}
        all_results[str(scene_dir)] = scene_results
        output = scene_dir / f"results_{args.region}.json"
        output.write_text(json.dumps(scene_results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(all_results, indent=2))


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="DGSRSim region-aware appearance evaluation")
    parser.add_argument("--model_paths", "-m", nargs="+", type=str)
    parser.add_argument("--renders_dir", "-r", type=str)
    parser.add_argument("--gt_dir", "-g", type=str)
    parser.add_argument("--mask_dir", type=str)
    parser.add_argument("--mask_subdir", default="masks")
    parser.add_argument("--region", choices=("object", "background", "full"), default="full")
    parser.add_argument(
        "--mask_label",
        type=int,
        help="Optional exact grayscale instance label. Without it, all nonzero pixels form the object region.",
    )
    parser.add_argument("--lpips_net", choices=("alex", "vgg", "squeeze"), default="vgg")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", type=str)
    return parser


def main() -> None:
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
