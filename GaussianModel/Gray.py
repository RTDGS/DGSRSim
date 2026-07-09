import cv2
import matplotlib.pyplot as plt

# 读取灰度图像7
img = cv2.imread("/home/ubuntu/Desktop/gaussian-grouping01/data/stick/object_mask/DSC_6166.png", cv2.IMREAD_GRAYSCALE)
img = cv2.imread("/home/ubuntu/Desktop/gaussian-grouping01/data/stick/object_mask/DSC_6166.png", cv2.IMREAD_GRAYSCALE)
img = cv2.imread("/home/ubuntu/Desktop/gaussian-grouping01/data/stick/object_mask/DSC_6166.png", cv2.IMREAD_GRAYSCALE)




if img is None:
    print("图像读取失败")
    exit()

# 显示图像
plt.imshow(img, cmap='gray')
plt.title("Gray Image")
plt.axis('on')
plt.show()
