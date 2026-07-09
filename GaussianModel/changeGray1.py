# -*- coding: utf-8 -*-
import sys
from pathlib import Path
from PIL import Image
import numpy as np


def main():
    if len(sys.argv) != 4:
        print(
            "Usage: python changeGray1.py <input_folder> <output_folder> <target_gray>\n"
            "Example:\n"
            "  python changeGray1.py data/combination2/object_mask data/combination2/inpaint_object_mask_255 32"
        )
        return 1

    input_folder = Path(sys.argv[1])
    output_folder = Path(sys.argv[2])
    target_gray = int(sys.argv[3])

    if not input_folder.exists():
        raise FileNotFoundError("Input folder not found: {0}".format(input_folder))
    if target_gray < 0 or target_gray > 255:
        raise ValueError("target_gray must be in [0, 255]")

    output_folder.mkdir(parents=True, exist_ok=True)

    scanned = 0
    exported = 0

    for p in sorted(input_folder.iterdir()):
        if p.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
            continue

        scanned += 1
        try:
            img = Image.open(str(p)).convert("L")
            arr = np.array(img)

            mask = (arr == target_gray)
            changed_pixels = int(mask.sum())

            if changed_pixels == 0:
                print("[SKIP] {0} (gray {1} not found)".format(p.name, target_gray))
                continue

            result = np.where(mask, 255, 0).astype(np.uint8)
            Image.fromarray(result).save(str(output_folder / p.name))

            exported += 1
            print("[OK] {0}, white pixels: {1}".format(p.name, changed_pixels))

        except Exception as e:
            print("[ERROR] {0}: {1}".format(p.name, e))

    print("[DONE] scanned={0}, output={1}, target_gray={2}".format(scanned, exported, target_gray))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())