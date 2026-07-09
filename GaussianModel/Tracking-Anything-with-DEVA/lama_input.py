import os
import glob
import argparse
from pathlib import Path

import cv2
import numpy as np


def imread_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def binarize(img: np.ndarray, thr: int) -> np.ndarray:
    return (img > thr)


def keep_largest_cc(mask_bool: np.ndarray) -> np.ndarray:
    """Keep largest connected component in a binary mask. If empty, return as-is."""
    mask_u8 = mask_bool.astype(np.uint8)
    num, labels = cv2.connectedComponents(mask_u8, connectivity=8)
    if num <= 1:
        return mask_bool
    areas = np.bincount(labels.reshape(-1))[1:]  # drop background
    keep = int(np.argmax(areas) + 1)
    return (labels == keep)


def refine_one(
        deva_path: str,
        obj_path: str,
        out_path: str,
        thr_deva: int,
        thr_obj: int,
        dilate_ks: int,
        keep_lcc: bool,
) -> None:
    deva = imread_gray(deva_path)
    obj = imread_gray(obj_path)

    deva_bin = binarize(deva, thr_deva)
    obj_bin = binarize(obj, thr_obj)

    ks = int(dilate_ks)
    if ks % 2 == 0:
        ks += 1
    kernel = np.ones((ks, ks), np.uint8)
    obj_dil = cv2.dilate(obj_bin.astype(np.uint8), kernel, iterations=1).astype(bool)

    # 核心逻辑修改：这里直接取 DEVA 识别到的区域
    # 如果你也想结合原始物体位置，保持 logical_and；
    # 但根据你的图，你更需要的是将 DEVA 的散点集中化，可以使用 logical_or 或者直接处理 deva_bin
    unseen = np.logical_and(deva_bin, obj_dil)

    if keep_lcc:
        unseen = keep_largest_cc(unseen)

    out = (unseen.astype(np.uint8) * 255)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, out)


def get_sorted_files(folder: str):
    """获取文件夹内所有图片并按名称排序"""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deva_dir", required=True, help="Directory of DEVA masks")
    parser.add_argument("--obj_dir", required=True, help="Directory of object masks")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--thr_deva", type=int, default=128)
    parser.add_argument("--thr_obj", type=int, default=128)
    parser.add_argument("--dilate_ks", type=int, default=41)
    parser.add_argument("--keep_lcc", action="store_true", help="Keep only largest connected component")
    args = parser.parse_args()

    # 修改点：不再使用 build_map，直接获取排序列表
    deva_files = get_sorted_files(args.deva_dir)
    obj_files = get_sorted_files(args.obj_dir)

    if len(deva_files) != len(obj_files):
        print(f"Warning: File count mismatch! DEVA: {len(deva_files)}, OBJ: {len(obj_files)}")
        print("The script will pair them by sequence order.")

    # 取两者中较小的数量进行配对
    num_pairs = min(len(deva_files), len(obj_files))

    if num_pairs == 0:
        raise RuntimeError("No images found in one or both directories.")

    print(f"Processing {num_pairs} pairs based on list order...")

    for i in range(num_pairs):
        deva_path = deva_files[i]
        obj_path = obj_files[i]

        # 重点：这里我们保留 DEVA 的原始文件名输出到结果，不改变它
        out_name = Path(deva_path).name
        out_path = os.path.join(args.out_dir, out_name)

        refine_one(
            deva_path=deva_path,
            obj_path=obj_path,
            out_path=out_path,
            thr_deva=args.thr_deva,
            thr_obj=args.thr_obj,
            dilate_ks=args.dilate_ks,
            keep_lcc=args.keep_lcc,
        )

    print(f"Done. Refined masks saved to: {args.out_dir}")


if __name__ == "__main__":
    main()