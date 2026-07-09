import os
import json
import torch
import lpips
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser
import torchvision.transforms.functional as tf

# 导入本地工具函数
from utils.loss_utils import ssim
from utils.image_utils import psnr


def readImages(renders_dir, gt_dir):
    """
    从指定目录读取渲染图和真值图，并确保它们一一对应。
    """
    renders = []
    gts = []
    image_names = []

    renders_path = Path(renders_dir)
    gt_path = Path(gt_dir)

    # 获取并排序文件名，确保对齐
    fnames = sorted([f for f in os.listdir(renders_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    for fname in fnames:
        render_path = renders_path / fname
        target_path = gt_path / fname

        if not target_path.exists():
            print(f"Warning: Ground truth not found for {fname}, skipping.")
            continue

        render = Image.open(render_path)
        gt = Image.open(target_path)

        # 转换为 Tensor [1, 3, H, W] 并移至 GPU
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)

    return renders, gts, image_names


def compute_metrics(renders, gts, image_names, lpips_fn):
    """
    计算核心指标：SSIM, PSNR, LPIPS。
    """
    ssims = []
    psnrs = []
    lpipss = []

    with torch.no_grad():
        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            r = renders[idx]
            g = gts[idx]

            # 1. SSIM & PSNR (通常输入范围 [0, 1])
            ssims.append(ssim(r, g).item())
            psnrs.append(psnr(r, g).item())

            # 2. LPIPS (必须缩放到 [-1, 1])
            lp_val = lpips_fn(r * 2.0 - 1.0, g * 2.0 - 1.0)
            lpipss.append(lp_val.item())

    # 计算均值
    m_ssim = torch.tensor(ssims).mean().item()
    m_psnr = torch.tensor(psnrs).mean().item()
    m_lpips = torch.tensor(lpipss).mean().item()

    return m_ssim, m_psnr, m_lpips, ssims, psnrs, lpipss


def evaluate(args):
    # 实例化 LPIPS 模型 (VGG 网络)
    lpips_fn = lpips.LPIPS(net='vgg').cuda()

    # --- 模式 A: 自定义路径模式 (-r 和 -g) ---
    if args.renders_dir and args.gt_dir:
        print(f"\n[Custom Mode]")
        print(f"Renders Dir: {args.renders_dir}")
        print(f"GT Dir:      {args.gt_dir}")

        renders, gts, image_names = readImages(args.renders_dir, args.gt_dir)
        if not renders:
            print("No images found! Check your paths.")
            return

        m_ssim, m_psnr, m_lpips, _, _, _ = compute_metrics(renders, gts, image_names, lpips_fn)

        print("\nResults:")
        print(f"  SSIM : {m_ssim:>12.7f}")
        print(f"  PSNR : {m_psnr:>12.7f}")
        print(f"  LPIPS: {m_lpips:>12.7f}")
        return

    # --- 模式 B: 原始模型文件夹遍历模式 (-m) ---
    if not args.model_paths:
        print("Error: Please provide either --model_paths or both --renders_dir and --gt_dir")
        return

    full_dict = {}
    for scene_dir in args.model_paths:
        try:
            print(f"\nScene: {scene_dir}")
            full_dict[scene_dir] = {}
            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                method_path = test_dir / method
                if not method_path.is_dir(): continue

                print(f"Method: {method}")
                renders_dir = method_path / "renders"
                gt_dir = method_path / "gt"

                renders, gts, image_names = readImages(renders_dir, gt_dir)
                m_ssim, m_psnr, m_lpips, ssims, psnrs, lpipss = compute_metrics(renders, gts, image_names, lpips_fn)

                print(f"  SSIM : {m_ssim:>12.7f}")
                print(f"  PSNR : {m_psnr:>12.7f}")
                print(f"  LPIPS: {m_lpips:>12.7f}")

                full_dict[scene_dir][method] = {"SSIM": m_ssim, "PSNR": m_psnr, "LPIPS": m_lpips}

            # 保存结果到场景目录
            with open(os.path.join(scene_dir, "results.json"), 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)

        except Exception as e:
            print(f"Unable to compute metrics for {scene_dir}. Error: {e}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Metric Evaluation Script")
    # 模式 B 参数
    parser.add_argument('--model_paths', '-m', nargs="+", type=str, help="List of model output paths")
    # 模式 A 参数
    parser.add_argument('--renders_dir', '-r', type=str, help="Direct path to rendered images folder")
    parser.add_argument('--gt_dir', '-g', type=str, help="Direct path to ground truth images folder")

    args = parser.parse_args()
    evaluate(args)