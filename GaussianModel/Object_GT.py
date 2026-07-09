import os
import cv2
import numpy as np

gt_dir = "/home/ubuntu/Desktop/gaussian-grouping01/data/ball-com/images"
mask_dir = "/home/ubuntu/Desktop/gaussian-grouping01/data/ball-com/inpaint_object_mask_255"
out_dir = "/home/ubuntu/Desktop/gaussian-grouping01/output/ball-com/train/ours_object_removal/iteration_10000/Object_gt"

os.makedirs(out_dir, exist_ok=True)

# 收集并排序所有 GT（保证编号顺序稳定）
gt_names = sorted([
    n for n in os.listdir(gt_dir)
    if n.lower().endswith(".jpg")
])

out_idx = 0  # 输出编号计数器

for gt_name in gt_names:
    stem = os.path.splitext(gt_name)[0]   # 原始前缀名
    mask_name = stem + ".png"

    gt_path = os.path.join(gt_dir, gt_name)
    mask_path = os.path.join(mask_dir, mask_name)

    if not os.path.exists(mask_path):
        print(f"[WARN] mask not found: {mask_name}")
        continue

    gt = cv2.imread(gt_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if gt is None or mask is None:
        print(f"[WARN] failed to read: {stem}")
        continue

    # 尺寸不一致时，最近邻 resize mask
    if mask.shape[:2] != gt.shape[:2]:
        mask = cv2.resize(
            mask,
            (gt.shape[1], gt.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    mask = (mask > 0).astype(np.uint8)

    # 背景置黑
    gt_black_bg = gt * mask[:, :, None]

    # 生成新文件名：00000.png
    out_name = f"{out_idx:05d}.png"
    out_path = os.path.join(out_dir, out_name)

    cv2.imwrite(out_path, gt_black_bg)
    out_idx += 1

print(f"Done. Saved {out_idx} images.")
