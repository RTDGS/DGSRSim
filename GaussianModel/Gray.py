import argparse

import cv2
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Inspect a grayscale mask image.")
parser.add_argument("input", help="Path to the grayscale image")
args = parser.parse_args()

img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError(f"图像读取失败: {args.input}")

# 显示图像
plt.imshow(img, cmap='gray')
plt.title("Gray Image")
plt.axis('on')
plt.show()
