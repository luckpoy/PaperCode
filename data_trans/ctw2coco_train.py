import json
from ctw_convert import CTWConvertor as ctw_train
import numpy as np
import cv2

def polygon_area(points):
    """返回多边形面ji
    """
    area = 0
    q = points[-1]
    for p in points:
        area += p[0] * q[1] - p[1] * q[0]
        q = p
    return abs(area / 2)

# read ctw
CTW_text_convertor = ctw_train(jsonl_path=r'/home/liuyingfeng/bo/CCD/ctw/ctw-annotations/train.jsonl',
                                image_root=r'/home/liuyingfeng/bo/CCD/ctw/images-trainval')
craft_train_list = CTW_text_convertor.convert_to_craft()

im_fn_list = []
target_boxes_list =[]
list_len = len(craft_train_list)

for index in range(list_len):
	try:
		char_box = craft_train_list[index % list_len][1].copy()[0]
		for i in range(1, len(craft_train_list[index % list_len][1])):
			char_box = np.concatenate((char_box, craft_train_list[index % list_len][1].copy()[i]), axis=0)

		# resize bbox
		if not char_box is None:
			char_box[:, :, 0] /= 2
			char_box[:, :, 1] /= 2

		target_boxes_list.append(char_box)
		im_fn_list.append(craft_train_list[index][0])
	except:
		print("a error occur!")

save_img_path = "/home/liuyingfeng/bo/dateset/ctw_mmocr/imgs/training/"
# for im_fn in im_fn_list:
#     im = cv2.imread(im_fn)[:, :, ::-1]
#     im = cv2.resize(im,(1024,1024))
#     im_name = im_fn.split("/")[-1]
#     cv2.imwrite(save_img_path + im_name, im[:, :, ::-1])
# exit()

images = "\"images\": "
categories = "\"categories\": [{\"id\": 1, \"name\": \"text\"}]"
annotations= "\"annotations\": "

image_list = []
anno_list = []
anno_id = 0
for idx, im_fn in enumerate(im_fn_list): 
	im_temp = "{\"file_name\": " + "\"training/" + im_fn.split("/")[-1] + "\", " + \
		"\"height\": 1024, \"width\": 1024, \"segm_file\": \"None\", \"id\": " +str(idx) +"}"
	image_list.append(im_temp)
	# print(im_temp)

	for i,bbox in target_boxes_list[idx]:
		anno_temp = "{\"iscrowd\": 0, \"category_id\": 1, \"bbox\": [" + \
			str(bbox[0][0]) + ", " + str(bbox[0][1]) + ", " + \
			str(bbox[2][0]-bbox[0][0]) + ", " + str(bbox[2][1]-bbox[0][1]) + \
			"], \"area\": 100.0, \"segmentation\": [[" + \
			str(bbox[0][0]) + ", " + str(bbox[0][1]) + ", " + \
			str(bbox[1][0]) + ", " + str(bbox[1][1]) + ", " + \
			str(bbox[2][0]) + ", " + str(bbox[2][1]) + ", " + \
			str(bbox[3][0]) + ", " + str(bbox[3][1]) + \
			"]], \"image_id\": " + str(idx) + ", \"id\": " + str(anno_id) + "}"
		anno_id += 1
		# print(anno_temp)
		anno_list.append(anno_temp)

	print("idx: "+ str(idx))
	# if idx is 2:
	# 	break

j = "{" + images + "[" + ', '.join(image_list) + "], " + \
	categories + ", " + \
	annotations + "[" + ', '.join(anno_list) + "]}" 
# print (j)

json1 = json.loads(j)
with open(r'ctw_coco_train.json', 'w') as f:
	json.dump(json1, f)
	print("New json file is created from data.json file")