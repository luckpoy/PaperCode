import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取输入图片
img = cv2.imread("/home/liuyingfeng/bo/CCD/my_dataset_trans/OCR.png")

# 将图片转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret , gray = cv2.threshold(gray, 2, 1, 0)

# 计算水平投影直方图
h_projection = np.sum(gray, axis=1)

# 显示水平投影直方图
plt.plot(h_projection)
# plt.title("Horizontal Projection Histogram")
# plt.xlabel("Pixel")
# plt.ylabel("Projection Value")
plt.axis('off')
# plt.show()
plt.savefig("TOCR.png")
