import matplotlib.pyplot as plt
import numpy as np

# def draw_iou(box1, box2):
#     # 计算矩形框的左上角和右下角坐标
#     box1_x1, box1_y1, box1_x2, box1_y2 = box1
#     box2_x1, box2_y1, box2_x2, box2_y2 = box2

#     # 计算矩形框的面积
#     box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
#     box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

#     # 计算矩形框的交集面积
#     inter_x1 = max(box1_x1, box2_x1)
#     inter_y1 = max(box1_y1, box2_y1)
#     inter_x2 = min(box1_x2, box2_x2)
#     inter_y2 = min(box1_y2, box2_y2)
#     inter_w = inter_x2 - inter_x1
#     inter_h = inter_y2 - inter_y1
#     inter_area = inter_w * inter_h
#     iou = inter_area / (box1_area + box2_area - inter_area)

#     # 绘制矩形框
#     fig, ax = plt.subplots()
#     rect1 = plt.Rectangle((box1_x1, box1_y1), box1_x2-box1_x1, box1_y2-box1_y1, fill=False, ec='b', lw=2)
#     rect2 = plt.Rectangle((box2_x1, box2_y1), box2_x2-box2_x1, box2_y2-box2_y1, fill=False, ec='g', lw=2)
#     ax.add_patch(rect1)
#     ax.add_patch(rect2)

#     # 绘制交集矩形框
#     if inter_w > 0 and inter_h > 0:
#         rect_inter = plt.Rectangle((inter_x1, inter_y1), inter_w, inter_h, fill=False, ec='r', lw=2)
#         ax.add_patch(rect_inter)

#     # 设置坐标轴范围
#     plt.xlim([min(box1_x1, box2_x1)-10, max(box1_x2, box2_x2)+10])
#     plt.ylim([min(box1_y1, box2_y1)-10, max(box1_y2, box2_y2)+10])

#     # 显示 IoU 值
#     plt.text(min(box1_x1, box2_x1), min(box1_y1, box2_y1)-5, f'IoU = {iou:.2f}', fontsize=14, color='black')

#     # 显示图形
#     plt.axis('off')
#     plt.savefig("IoU.png")

# # 示例：计算 IoU 值并绘制矩形框
# box1 = [50, 50, 200, 200]
# box2 = [150, 150, 300, 300]
# draw_iou(box1, box2)

import cv2

# 创建一个空白图像
img = np.zeros((500, 500, 3), dtype=np.uint8)
# img = 255

# 画两个矩形框
box1 = (50, 50, 300, 300)
box2 = (200, 200, 450, 450)
cv2.rectangle(img, (box1[0], box1[1]), (box1[2], box1[3]), (255, 0, 0), 2)
cv2.rectangle(img, (box2[0], box2[1]), (box2[2], box2[3]), (0, 0, 255), 2)

# 计算交叉区域
intersection = (max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3]))

# 计算 IoU
area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
intersection_area = max(intersection[2] - intersection[0], 0) * max(intersection[3] - intersection[1], 0)
union_area = area1 + area2 - intersection_area
iou = intersection_area / union_area

# 填充交叉区域
if intersection_area > 0:
    cv2.rectangle(img, (intersection[0], intersection[1]), (intersection[2], intersection[3]), (255, 0, 255), -1)

# 显示 IoU 值
cv2.putText(img, "IoU = {:.2f}".format(iou), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

# 显示图像
cv2.imwrite('IoU.png', img)
# plt.imsave('IoU.png', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
