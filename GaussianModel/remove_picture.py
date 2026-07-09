import os
import glob


def keep_one_fifth_images(folder):
    # 支持所有常见后缀（大小写）
    exts = [
        "*.jpg", "*.JPG",
        "*.jpeg", "*.JPEG",
        "*.png", "*.PNG",
        "*.bmp", "*.BMP",
        "*.gif", "*.GIF",
        "*.webp", "*.WEBP"
    ]

    # 读取所有图片路径
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder, ext)))

    files.sort()  # 保证顺序一致

    if len(files) == 0:
        print("❌ 文件夹中没有图片，请检查路径/后缀")
        return

    total = len(files)

    # --- 核心修改 ---
    keep_count = max(1, total // 5)  # 至少保留1张
    # ----------------

    print(f"总图片数: {total}")
    print(f"计划保留: {keep_count} 张 (约 1/5)")

    # 均匀采样保留索引
    step = total / keep_count
    keep_indices = {int(i * step) for i in range(keep_count)}

    keep_files = [files[i] for i in keep_indices]
    delete_files = [f for i, f in enumerate(files) if i not in keep_indices]

    print(f"实际保留 {len(keep_files)} 张")
    print(f"实际删除 {len(delete_files)} 张")

    # 删除其余文件
    for f in delete_files:
        try:
            os.remove(f)
        except Exception as e:
            print(f"删除失败：{f}, 错误: {e}")

    print("✅ 保留 1/5 完成！")


if __name__ == "__main__":
    folder_path = r"/home/ubuntu/Desktop/gaussian-grouping01/data/ScanNet0038/input"
    keep_one_fifth_images(folder_path)