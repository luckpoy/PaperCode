import os
import cv2
import numpy as np

def get_anno_list(path):
	# path = "/home/bo/桌面/LSVT/train_origin/train_full_images_0/"
	datanames = os.listdir(path)
	list = []
	for i in datanames:
		if os.path.exists(path):                   #判断是否存在文件夹如果不存在则创建为文件夹
			list.append(i)
		else:
			print("haven't pic")
	return list

def decode_anno(path,mode):
	if mode == 'test':
		anno_path = path + 'anno_test/'
	elif mode == 'train':
		anno_path = path + 'anno_train/'
	else:
		print('path is error!')

	# pic_path = path + 'pics/'

	anno_list = get_anno_list(anno_path)
	pic_list = []
	bboxes = []
	for anno_name in anno_list:
		# print(anno_name[0:-3] + str("jpg"))
		pic_list.append(anno_name[0:-3] + str("jpg"))
		bbox = []
		for line in open(anno_path + anno_name): 
			bbox_str = line[:-1].split(' ')[1:]
			# print(bbox_str)
			if len(bbox_str) != 4:
				print("num error!")
				break
			
			bbox_num = []
			for s in bbox_str:
				bbox_num.append(float(s)*512)
			# print(bbox_num)
			x1 = bbox_num[0]-bbox_num[2]/2.0
			y1 = bbox_num[1]-bbox_num[3]/2.0
			x2 = bbox_num[0]+bbox_num[2]/2.0
			y2 = bbox_num[1]-bbox_num[3]/2.0
			x3 = bbox_num[0]+bbox_num[2]/2.0
			y3 = bbox_num[1]+bbox_num[3]/2.0
			x4 = bbox_num[0]-bbox_num[2]/2.0
			y4 = bbox_num[1]+bbox_num[3]/2.0
			bbox_num = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
			bbox.append(bbox_num)
			# print(bbox_num)
		bboxes.append(bbox)
	return pic_list,bboxes
	
def plt_Contours(path, bboxes, pic_list):
	idx = 0
	for img_name in pic_list:
		pic_path = path + 'pics/' + img_name
		img = cv2.imread(pic_path)
			# cv2.imwrite("img.jpg",img)
		for i in bboxes[idx]:
			# print(i)
			cv2.drawContours(img, [np.array(i,dtype=np.int32).reshape((-1, 1, 2))], -1, (0, 255, 0), 1)
		cv2.imwrite("/home/liuyingfeng/bo/dateset/unamed_dataset/counters_pics/" + img_name, img)

		idx = idx + 1
		# if idx == 30:
		# 	break

if __name__ == '__main__':
	path = r'/home/liuyingfeng/bo/dateset/unamed_dataset/'
	pic_list,bboxes = decode_anno(path,'train')
	plt_Contours(path, bboxes, pic_list)