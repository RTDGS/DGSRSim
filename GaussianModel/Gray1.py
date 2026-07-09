import sys
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt


def pick_gray_id(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError("Failed to read image: {0}".format(image_path))

    picked = {"value": None}

    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray", interpolation="nearest")
    ax.set_title("Left-click to pick a grayscale value. Window will close automatically.")
    ax.axis("on")

    def on_click(event):
        if event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        x = int(round(event.xdata))
        y = int(round(event.ydata))

        if y < 0 or y >= img.shape[0] or x < 0 or x >= img.shape[1]:
            return

        v = int(img[y, x])
        picked["value"] = v
        print("[PICKED] x={0}, y={1}, gray={2}".format(x, y, v))
        plt.close(fig)

    cid = fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()
    fig.canvas.mpl_disconnect(cid)

    if picked["value"] is None:
        raise RuntimeError("No grayscale value picked (window closed without clicking).")

    return picked["value"]


def main():
    """
    Usage:
      python Gray1.py <image_path> <output_json>

    Writes:
      {"target_gray": <int>}
    """
    if len(sys.argv) != 3:
        print("Usage: python Gray1.py <image_path> <output_json>")
        return 1

    image_path = Path(sys.argv[1])
    output_json = Path(sys.argv[2])

    if not image_path.exists():
        raise FileNotFoundError("Image not found: {0}".format(image_path))

    gray = pick_gray_id(image_path)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"target_gray": gray}, indent=4), encoding="utf-8")
    print("[INFO] Saved picked gray value to: {0}".format(output_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())