"""Region-aware image metrics used by the paper-aligned evaluator."""

from __future__ import annotations

from math import exp

import torch
import torch.nn.functional as F


def _validate_images(image: torch.Tensor, target: torch.Tensor) -> None:
    if image.shape != target.shape:
        raise ValueError(
            f"image and target must have the same shape, got {image.shape} and {target.shape}"
        )
    if image.ndim != 4:
        raise ValueError(f"images must be [B, C, H, W], got {image.shape}")


def normalize_region_mask(mask: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    """Return a boolean [B, 1, H, W] mask aligned with ``image``."""
    if mask.ndim == 2:
        mask = mask[None, None]
    elif mask.ndim == 3:
        mask = mask[:, None] if mask.shape[0] == image.shape[0] else mask[None]
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"mask must resolve to [B, 1, H, W], got {mask.shape}")
    if mask.shape[0] == 1 and image.shape[0] > 1:
        mask = mask.expand(image.shape[0], -1, -1, -1)
    if mask.shape[0] != image.shape[0] or mask.shape[-2:] != image.shape[-2:]:
        raise ValueError(f"mask shape {mask.shape} is not aligned with image shape {image.shape}")
    mask = mask.to(device=image.device, dtype=torch.bool)
    if not torch.all(mask.flatten(1).any(dim=1)):
        raise ValueError("each evaluated image must contain at least one selected pixel")
    return mask


def masked_psnr(
    image: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute PSNR from the channel-wise MSE inside the selected region."""
    _validate_images(image, target)
    region = normalize_region_mask(mask, image).to(image.dtype)
    weighted_error = (image - target).square() * region
    denominator = region.sum(dim=(1, 2, 3)) * image.shape[1]
    mse = weighted_error.sum(dim=(1, 2, 3)) / denominator.clamp_min(1.0)
    return 20.0 * torch.log10(1.0 / torch.sqrt(mse))


def _gaussian_window(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    values = torch.tensor(
        [exp(-((x - window_size // 2) ** 2) / (2.0 * sigma**2)) for x in range(window_size)],
        device=device,
        dtype=dtype,
    )
    values /= values.sum()
    return values[:, None] @ values[None, :]


def masked_ssim(
    image: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    window_size: int = 11,
) -> torch.Tensor:
    """Average the SSIM map only over pixels selected by ``mask``."""
    _validate_images(image, target)
    region = normalize_region_mask(mask, image)
    channels = image.shape[1]
    window = _gaussian_window(window_size, 1.5, image.device, image.dtype)
    window = window[None, None].expand(channels, 1, -1, -1).contiguous()
    padding = window_size // 2

    mu_image = F.conv2d(image, window, padding=padding, groups=channels)
    mu_target = F.conv2d(target, window, padding=padding, groups=channels)
    mu_image_sq = mu_image.square()
    mu_target_sq = mu_target.square()
    mu_cross = mu_image * mu_target
    sigma_image = (
        F.conv2d(image.square(), window, padding=padding, groups=channels) - mu_image_sq
    )
    sigma_target = (
        F.conv2d(target.square(), window, padding=padding, groups=channels) - mu_target_sq
    )
    sigma_cross = (
        F.conv2d(image * target, window, padding=padding, groups=channels) - mu_cross
    )

    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = ((2.0 * mu_cross + c1) * (2.0 * sigma_cross + c2)) / (
        (mu_image_sq + mu_target_sq + c1)
        * (sigma_image + sigma_target + c2)
    )
    ssim_map = ssim_map.mean(dim=1, keepdim=True)
    weights = region.to(ssim_map.dtype)
    return (ssim_map * weights).sum(dim=(1, 2, 3)) / weights.sum(
        dim=(1, 2, 3)
    ).clamp_min(1.0)


def masked_spatial_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Aggregate a spatial metric map, such as LPIPS, over a selected region."""
    if values.ndim != 4:
        raise ValueError(f"spatial metric values must be [B, C, H, W], got {values.shape}")
    reference = torch.empty(
        values.shape[0],
        1,
        mask.shape[-2],
        mask.shape[-1],
        device=values.device,
        dtype=values.dtype,
    )
    region = normalize_region_mask(mask, reference).to(values.dtype)
    region = F.interpolate(region, size=values.shape[-2:], mode="nearest")
    if values.shape[1] != 1:
        region = region.expand(-1, values.shape[1], -1, -1)
    return (values * region).sum(dim=(1, 2, 3)) / region.sum(
        dim=(1, 2, 3)
    ).clamp_min(1.0)
