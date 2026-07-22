from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


GAUSSIAN_ROOT = Path(__file__).resolve().parents[1]
if str(GAUSSIAN_ROOT) not in sys.path:
    sys.path.insert(0, str(GAUSSIAN_ROOT))

from utils.region_metrics import masked_psnr, masked_spatial_mean, masked_ssim


class RegionMetricTests(unittest.TestCase):
    def test_object_psnr_ignores_background_error(self):
        target = torch.zeros(1, 3, 16, 16)
        render = target.clone()
        render[:, :, :, 8:] = 1.0
        object_mask = torch.zeros(1, 1, 16, 16, dtype=torch.bool)
        object_mask[:, :, :, :8] = True
        background_mask = ~object_mask

        self.assertTrue(torch.isinf(masked_psnr(render, target, object_mask)).all())
        self.assertAlmostEqual(masked_psnr(render, target, background_mask).item(), 0.0)

    def test_identical_region_has_unit_ssim(self):
        image = torch.rand(1, 3, 16, 16)
        mask = torch.zeros(1, 1, 16, 16, dtype=torch.bool)
        mask[:, :, 4:12, 4:12] = True
        self.assertAlmostEqual(masked_ssim(image, image, mask).item(), 1.0, places=5)

    def test_spatial_metric_uses_selected_pixels_only(self):
        values = torch.tensor([[[[1.0, 3.0], [5.0, 7.0]]]])
        mask = torch.tensor([[[[True, False], [True, False]]]])
        self.assertAlmostEqual(masked_spatial_mean(values, mask).item(), 3.0)


if __name__ == "__main__":
    unittest.main()
