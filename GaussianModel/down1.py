from PIL import Image
from pathlib import Path
from typing import Tuple

# ============================================================
# CONFIG (edit here only)

# ============================================================
# data/
# ├── ball-com/
# │   ├── 0001.png          # 原图（保留）
# │   ├── 0002.png
# │   └── input/            # 新建
# │       ├── 0001.png      # resize 后
# │       └── 0002.png

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"

# 要处理的数据集目录名
DATASETS = [
    # "yumi",
    # "hu",
    # "guo",
    # "toumingguogai",
    # "pingdiguo",
    # "gaoyaguo",
    # "gaoyaguogai",
    # "guochan",
    # "beizi",
    # "diancilu",
    # "wan",
"table",
]

# resize 目标尺寸 (width, height)
TARGET_SIZE: Tuple[int, int] = (1500, 1125)
#TARGET_SIZE: Tuple[int, int] = (994, 738)
# 输出子目录名：在 data/<dataset>/ 下创建 input/
OUTPUT_SUBDIR = "input"

# 支持的图片格式
SUPPORTED_FORMATS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif")


# ============================================================
# Core logic
# ============================================================

def resize_image(input_path: Path, output_path: Path, target_size: Tuple[int, int]) -> None:
    try:
        with Image.open(input_path) as img:
            resized = img.resize(target_size, Image.LANCZOS)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            resized.save(output_path)
            print(f"[OK] {input_path.name} -> {output_path}")
    except Exception as e:
        print(f"[ERROR] {input_path}: {e}")


def resize_one_dataset(dataset: str) -> None:
    print("\n" + "-" * 80)
    print(f"[DATASET] {dataset}")

    dataset_dir = DATA_ROOT / dataset
    if not dataset_dir.exists():
        print(f"[SKIP] dataset folder not found: {dataset_dir}")
        return

    output_dir = dataset_dir / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # IMPORTANT:
    # Original images are directly under data/<dataset>/ (NOT under input/)
    # We must ignore any subdirectories (including the newly created input/)
    images = [
        p for p in sorted(dataset_dir.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_FORMATS
    ]

    if not images:
        print(f"[SKIP] no images found directly under: {dataset_dir}")
        return

    print(f"[INFO] Found {len(images)} original images under: {dataset_dir}")
    print(f"[INFO] Resized outputs will be written to: {output_dir}")

    for img_path in images:
        out_path = output_dir / img_path.name
        resize_image(img_path, out_path, TARGET_SIZE)


def main() -> None:
    print("=" * 80)
    print("[START] Resize originals in data/<dataset>/ into data/<dataset>/input/")
    print(f"[TARGET SIZE] {TARGET_SIZE}")
    print(f"[OUTPUT DIR ] {OUTPUT_SUBDIR}")
    print("=" * 80)

    for dataset in DATASETS:
        resize_one_dataset(dataset)

    print("\n[ALL DONE] Resize preprocessing finished.")


if __name__ == "__main__":
    main()