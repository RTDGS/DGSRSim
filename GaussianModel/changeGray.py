# -*- coding: utf-8 -*-
import argparse
import os
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser(description="Convert selected grayscale IDs into binary masks.")
parser.add_argument("--input", required=True, help="Input mask directory")
parser.add_argument("--output", required=True, help="Output binary-mask directory")
parser.add_argument("--target-grays", nargs="+", type=int, default=[3], help="Grayscale IDs mapped to 255")
args = parser.parse_args()

input_folder = args.input
output_folder = args.output
target_grays = args.target_grays

# 创建输出目录
os.makedirs(output_folder, exist_ok=True)

count_total = 0
count_converted = 0

for filename in sorted(os.listdir(input_folder)):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
        continue

    img_path = os.path.join(input_folder, filename)

    try:
        # 转为灰度模式
        img = Image.open(img_path).convert("L")
        arr = np.array(img)

        # 判断是否存在目标灰度值（多个）
        mask = np.isin(arr, target_grays)
        changed_pixels = np.sum(mask)

        # 如果没有这些灰度值，跳过保存
        if changed_pixels == 0:
            print(f"跳过：{filename}（未找到灰度值 {target_grays}）")
            continue

        # 生成结果图：目标灰度值 -> 255，其余 -> 0
        result = np.where(mask, 255, 0).astype(np.uint8)

        out_img = Image.fromarray(result)
        out_path = os.path.join(output_folder, filename)
        out_img.save(out_path)

        count_converted += 1
        print(f"✅ 处理完成：{filename}，变白像素数：{changed_pixels}")

    except Exception as e:
        print(f"❌ 处理失败：{filename}，错误：{e}")

    count_total += 1

print(f"\n🎉 全部处理完成！共扫描 {count_total} 张图像，其中成功输出 {count_converted} 张。")
