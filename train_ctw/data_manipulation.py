from turtle import position
from shapely.geometry import Polygon
import numpy as np
import cv2
import config

# Done so that the edge has a value of ~ 0.4
center = config.window//2
gaussian_heatmap = np.zeros([config.window, config.window], dtype=np.float32)

for i_ in range(config.window):
	for j_ in range(config.window):
		gaussian_heatmap[i_, j_] = 1 / 2 / np.pi / (config.sigma ** 2) * np.exp(
			-1 / 2 * ((i_ - center) ** 2 + (j_ - center) ** 2) / (config.sigma ** 2))

gaussian_heatmap = (gaussian_heatmap / np.max(gaussian_heatmap) * 255).astype(np.uint8)

gaussian_heatmap_aff = np.zeros([config.window, config.window], dtype=np.float32)

for i_ in range(config.window):
	for j_ in range(config.window):
		gaussian_heatmap_aff[i_, j_] = 1 / 2 / np.pi / (config.sigma_aff ** 2) * np.exp(
			-1 / 2 * ((i_ - center) ** 2 + (j_ - center) ** 2) / (config.sigma_aff ** 2))

gaussian_heatmap_aff = (gaussian_heatmap_aff / np.max(gaussian_heatmap_aff) * 255).astype(np.uint8)


def normalize_mean_variance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
	# should be RGB order
	img = in_img.copy().astype(np.float32)

	img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
	img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
	return img


def denormalize_mean_variance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
	# should be RGB order
	img = in_img.copy()
	img *= variance
	img += mean
	img *= 255.0
	img = np.clip(img, 0, 255).astype(np.uint8)
	return img


def four_point_transform(image, pts, size):

	"""
	Using the pts and the image a perspective transform is performed which returns the transformed 2d Gaussian image
	:param image: np.array, dtype=np.uint8, shape = [height, width]
	:param pts: np.array, dtype=np.float32 or np.int32, shape = [4, 2]
	:param size: size of the original image, list [height, width]
	:return:
	"""

	height, width = size

	center_pt = np.mean(pts, axis=0)
	pts = pts - center_pt[None, :]
	pts = pts*center/config.threshold_point
	pts = pts + center_pt[None, :]

	dst = np.array([
		[0, 0],
		[image.shape[1] - 1, 0],
		[image.shape[1] - 1, image.shape[0] - 1],
		[0, image.shape[0] - 1]], dtype="float32")

	warped = cv2.warpPerspective(image, cv2.getPerspectiveTransform(dst, pts), (width, height))

	return warped


def resize(image, character, side=512):

	"""
		Resizing the image while maintaining the aspect ratio and padding with average of the entire image to make the
		reshaped size = (side, side)
		:param image: np.array, dtype=np.uint8, shape=[height, width, 3]
		:param character: np.array, dtype=np.int32 or np.float32, shape = [2, 4, num_characters]
		:param side: new size to be reshaped to
		:return: resized_image, corresponding reshaped character bbox
	"""

	height, width, channel = image.shape
	max_side = max(height, width)
	new_resize = (int(width/max_side*side), int(height/max_side*side))
	image = cv2.resize(image, new_resize)

	character[0, :, :] = character[0, :, :]/width*new_resize[0]
	character[1, :, :] = character[1, :, :]/height*new_resize[1]

	big_image = np.ones([side, side, 3], dtype=np.float32)*np.mean(image)
	big_image[
		(side-image.shape[0])//2: (side-image.shape[0])//2 + image.shape[0],
		(side-image.shape[1])//2: (side-image.shape[1])//2 + image.shape[1]] = image
	big_image = big_image.astype(np.uint8)

	character[0, :, :] += (side-image.shape[1])//2
	character[1, :, :] += (side-image.shape[0])//2

	return big_image, character


def resize_generated(image, character, affinity, side=768):
	"""
		Resizing the image while maintaining the aspect ratio and padding with average of the entire image to make the
		reshaped size = (side, side)
		:param image: np.array, dtype=np.uint8, shape=[height, width, 3]
		:param character: list of np.array, dtype=np.int64, shape = [num_words, num_characters, 4, 1, 2]
		:param affinity: list of np.array, dtype=np.int64, shape = [num_words, num_affinity, 4, 1, 2]
		:param side: new size to be reshaped to
		:return: resized_image, corresponding reshaped character bbox
	"""

	height, width, channel = image.shape
	max_side = max(height, width)
	new_resize = (int(width/max_side*side), int(height/max_side*side))
	image = cv2.resize(image, new_resize)

	for word_no in range(len(character)):

		character[word_no][:, :, :, 0] = character[word_no][:, :, :, 0] / width * new_resize[0]
		character[word_no][:, :, :, 1] = character[word_no][:, :, :, 1] / height * new_resize[1]
		affinity[word_no][:, :, :, 0] = affinity[word_no][:, :, :, 0] / width * new_resize[0]
		affinity[word_no][:, :, :, 1] = affinity[word_no][:, :, :, 1] / height * new_resize[1]

	big_image = np.ones([side, side, 3], dtype=np.float32)*np.mean(image)
	big_image[
		(side-image.shape[0])//2: (side-image.shape[0])//2 + image.shape[0],
		(side-image.shape[1])//2: (side-image.shape[1])//2 + image.shape[1]] = image
	big_image = big_image.astype(np.uint8)

	for word_no in range(len(character)):

		character[word_no][:, :, :, 0] += (side - image.shape[1]) // 2
		character[word_no][:, :, :, 1] += (side - image.shape[0]) // 2
		affinity[word_no][:, :, :, 0] += (side - image.shape[1]) // 2
		affinity[word_no][:, :, :, 1] += (side - image.shape[0]) // 2

	return big_image, character, affinity


def add_character(image, bbox, heatmap=gaussian_heatmap):

	"""
		Add gaussian heatmap for character bbox to the image
		:param image: 2-d array containing character heatmap
		:param bbox: np.array, dtype=np.int32, shape = [4, 2]
		:param heatmap: gaussian heatmap
		:return: image in which the gaussian character bbox has been added
	"""

	# ToDo - Make this function efficient

	if not Polygon(bbox.reshape([4, 2]).astype(np.int32)).is_valid:

		return image

	bbox_top_left = np.min(bbox, axis=0)
	bbox_top_right = np.max(bbox, axis=0)

	transformed = four_point_transform(heatmap, bbox.astype(np.float32), [image.shape[0], image.shape[1]])

	image = np.maximum(image, transformed)

	return image


def add_character_others(image, weight_map, weight_val, bbox, type_='char'):
	"""
		Add gaussian heatmap for character bbox to the image and also generate weighted map for weak-supervision
		:param image: 2-d array containing character heatmap
		:param weight_map: 2-d array containing weight heatmap
		:param weight_val: weight to be given to the current bbox
		:param bbox: np.array, dtype=np.int32, shape = [4, 2]
		:param type_: used to distinguish which gaussian heatmap to use for affinity and characters
		:return:    image in which the gaussian character bbox has been added,
					weight_map in which the weight as per weak-supervision has been calculated
	"""

	# ToDo - Make this function efficient

	if type_ == 'char':
		heatmap = gaussian_heatmap.copy()
	else:
		heatmap = gaussian_heatmap_aff.copy()

	transformed = four_point_transform(
		heatmap, bbox.astype(np.float32), [weight_map.shape[0], weight_map.shape[1]])
	image = np.maximum(image, transformed)
	weight_map = np.maximum(weight_map, np.float32(transformed >= config.THRESHOLD_POSITIVE*255)*weight_val)

	return image, weight_map

def add_affinity(image, bbox_1, bbox_2):

	"""
		Add gaussian heatmap for affinity bbox to the image between bbox_1, bbox_2
		:param image: 2-d array containing affinity heatmap
		:param bbox_1: np.array, dtype=np.int32, shape = [4, 2]
		:param bbox_2: np.array, dtype=np.int32, shape = [4, 2]
		:return: image in which the gaussian affinity bbox has been added
	"""

	if (not Polygon(bbox_1.reshape([4, 2]).astype(np.int32)).is_valid) or (
			not Polygon(bbox_2.reshape([4, 2]).astype(np.int32)).is_valid):
		return image, np.zeros([4, 2])

	center_1, center_2 = np.mean(bbox_1, axis=0), np.mean(bbox_2, axis=0)

	# ToDo - No guarantee that bbox is ordered, hence affinity can be wrong

	# Shifted the affinity so that adjacent affinity do not touch each other

	tl = np.mean([bbox_1[0], bbox_1[1], center_1], axis=0)
	bl = np.mean([bbox_1[2], bbox_1[3], center_1], axis=0)
	tr = np.mean([bbox_2[0], bbox_2[1], center_2], axis=0)
	br = np.mean([bbox_2[2], bbox_2[3], center_2], axis=0)

	affinity = np.array([tl, tr, br, bl])

	return add_character(image, affinity, heatmap=gaussian_heatmap_aff), affinity


def two_char_bbox_to_affinity(bbox_1, bbox_2):

	"""
	Given two character bbox generates the co-ordinates of the affinity bbox between them
	:param bbox_1: type=np.array, dtype=np.int64, shape = [4, 1, 2]
	:param bbox_2: type=np.array, dtype=np.int64, shape = [4, 1, 2]
	:return: affinity bbox, type=np.array, dtype=np.int64, shape = [4, 1, 2]
	"""

	if (not Polygon(bbox_1.reshape([4, 2]).astype(np.int32)).is_valid) or (
			not Polygon(bbox_2.reshape([4, 2]).astype(np.int32)).is_valid):
		return np.zeros([4, 1, 2], dtype=np.int32)

	bbox_1 = bbox_1[:, 0, :].copy()
	bbox_2 = bbox_2[:, 0, :].copy()

	center_1, center_2 = np.mean(bbox_1, axis=0), np.mean(bbox_2, axis=0)

	# Shifted the affinity so that adjacent affinity do not touch each other

	tl = np.mean([bbox_1[0], bbox_1[1], center_1], axis=0)
	bl = np.mean([bbox_1[2], bbox_1[3], center_1], axis=0)
	tr = np.mean([bbox_2[0], bbox_2[1], center_2], axis=0)
	br = np.mean([bbox_2[2], bbox_2[3], center_2], axis=0)

	affinity = np.array([tl, tr, br, bl]).reshape([4, 1, 2])

	return affinity


def add_affinity_others(image, weight, weight_val, bbox_1, bbox_2):

	"""
		Add gaussian heatmap for affinity bbox to the image and also generate weighted map for weak-supervision
		:param image: 2-d array containing affinity heatmap
		:param weight: 2-d array containing weight heatmap
		:param weight_val: weight to be given to the current affinity bbox
		:param bbox_1: np.array, dtype=np.int32, shape = [4, 2]
		:param bbox_2: np.array, dtype=np.int32, shape = [4, 2]
		:return:    image in which the gaussian affinity bbox has been added between bbox_1 and bbox_2,
					weight_map in which the weight as per weak-supervision has been calculated
	"""

	affinity = two_char_bbox_to_affinity(bbox_1, bbox_2)

	return add_character_others(image, weight, weight_val, affinity)

def polygon_area(points):
    """返回多边形面ji
    """
    area = 0
    q = points[-1]
    for p in points:
        area += p[0] * q[1] - p[1] * q[0]
        q = p
    return abs(area / 2)

def generate_target(image_size, character_bbox, weight=None):

	"""

	:param image_size: [3, 768, 768]
	:param character_bbox: np.array, shape = [2, 4, num_characters]
	:param weight: this function is currently only used for synth-text in which we have 100 % confidence so weight = 1
					where the character bbox are present
	:return: if weight is not None then target_character_heatmap otherwise target_character_heatmap,
																			weight for weak-supervision
	"""

	character_bbox = character_bbox.transpose(2, 1, 0)

	channel, height, width = image_size

	target = np.zeros([height, width], dtype=np.uint8)

	for i in range(character_bbox.shape[0]):
		target = add_character(target, character_bbox[i].copy())

	if weight is not None:
		return target/255, np.float32(target >= config.THRESHOLD_POSITIVE*255)
	else:
		return target/255

import matplotlib.pyplot as plt

def getCrossPoint(box):
    LineA=[box[0][0],box[0][1],box[2][0],box[2][1]]
    LineB=[box[1][0],box[1][1],box[3][0],box[3][1]]

    ka = (LineA[3] - LineA[1]) / (LineA[2] - LineA[0])
    kb = (LineB[3] - LineB[1]) / (LineB[2] - LineB[0])

    x = (ka*LineA[0] - LineA[1] - kb*LineB[0] + LineB[1]) / (ka - kb)
    y = (ka*kb*(LineA[0] - LineB[0]) + ka*LineB[1] - kb*LineA[1]) / (ka - kb)
    return (x,y)

def add_enlarge_box(image, bbox, enlarge_rate=0.9, heatmap=gaussian_heatmap, op="add"):
	if not Polygon(bbox.reshape([4, 2]).astype(np.int32)).is_valid:
		return image
	try:
		x,y=getCrossPoint(bbox)
		# https://blog.csdn.net/m0_38007695/article/details/82665861
		for i in range(4):
			bbox[i][0]=enlarge_rate*bbox[i][0] + x*(1-enlarge_rate)
			bbox[i][1]=enlarge_rate*bbox[i][1] + y*(1-enlarge_rate)
	except:
		print("-----------------add_enlarge_box occur error--------------------")
		return image

	transformed = four_point_transform(heatmap, bbox.astype(np.float32), [image.shape[0], image.shape[1]])
	
	if op is "add":
		image = cv2.add(image,transformed)
	else:
		return transformed, bbox

	return image

def generate_cross_affinity_map(image_size, character_bbox):
	'''
	1. 放大字符�?
	2. 生成二维高斯密度�?
	3. 为每个标签和图中对应像素点进行异或运�?
	4. 输出结果作为邻域�?
	'''
	channel, height, width = image_size

	target = np.zeros([height, width], dtype=np.uint8)
	pos_map = np.zeros([height, width], dtype=np.uint8)
	temp = np.zeros([height, width], dtype=np.uint8)


	# cv2.polylines(pos_map, character_bbox.astype(np.int32), 1, 255)

	for i in range(character_bbox.shape[0]):
		target = add_enlarge_box(target, character_bbox[i].copy())

	# cv2.polylines(target, character_bbox.astype(np.int32), 1, 255)
	# plt.imsave("t.png",target)

	for i in range(character_bbox.shape[0]):
		single_target, b= add_enlarge_box(pos_map, character_bbox[i].copy(),op="xor")

		cv2.polylines(pos_map, [b.astype(np.int32)], 1, 255)

		ret, single_target_threshold = cv2.threshold(single_target, 30, 127, 0)
		pos_map = pos_map + single_target_threshold
	
	# plt.imsave("./test.png",pos_map)
	ret,pos_map = cv2.threshold(pos_map, 128, 255, 0)
	
		# pos_map = np.logical_xor(target, )
		# target = target * pos_map
		# target = np.multiply(target,pos_map)
	# cv2.polylines(pos_map, character_bbox.astype(np.int32), 1, 255)
	# plt.imsave("./pos_map.png",pos_map)
	target = np.multiply(target,pos_map/255)
	plt.imsave("./tt.png",target)

	return target/255

def add_affinity_character(image, bbox, heatmap=gaussian_heatmap):
	if not Polygon(bbox.reshape([4, 2]).astype(np.int32)).is_valid:
		return image

	transformed = four_point_transform(heatmap, bbox.astype(np.float32), [image.shape[0], image.shape[1]])

	return transformed

def generate_origin_cross_affinity_map(image_size, character_bbox):
	character_bbox = character_bbox.transpose(2, 1, 0)
	channel, height, width = image_size

	target = np.zeros([height, width], dtype=np.uint8)
	pos_map = np.zeros([height, width], dtype=np.uint8)

	if character_bbox.shape[0] == 0:
		return target/255

	for i in range(character_bbox.shape[0]):
		target = add_character(target, character_bbox[i].copy())

	for i in range(character_bbox.shape[0]):
		single_target = add_affinity_character(pos_map, character_bbox[i].copy())
		ret, single_target_threshold = cv2.threshold(single_target, 40, 127, 0)
		pos_map = pos_map + single_target_threshold
	
	ret,pos_map = cv2.threshold(pos_map, 128, 255, 0)

	target = np.multiply(target,pos_map/255)

	# cv2.polylines(target, character_bbox.astype(np.int32), 1, 255)
	# plt.imsave("./target.png",target)
	# plt.imsave("./pos_map.png",pos_map)
	return target/255

def generate_target_others(image_size, character_bbox, weight, type_='char'):
	"""
		:param image_size: size of the image on which the target needs to be generated
		:param character_bbox: np.array, shape = [word_length, num_characters, 4, 1, 2]
		:param weight: this function is currently only used for icdar2013, so weight is the value of weight
																							for each character bbox
		:param type_: used to differentiate between gaussian heatmap to be used for affinity and characters
		:return: if weight is not None then target_character_heatmap otherwise target_character_heatmap,
																				weight for weak-supervision
		"""

	if len(image_size) == 2:
		height, width = image_size
	else:
		channel, height, width = image_size

	target = np.zeros([height, width], dtype=np.uint8)
	weight_map = np.zeros([height, width], dtype=np.float32)

	for word_no in range(len(character_bbox)):

		for i in range(character_bbox[word_no].shape[0]):

			target, weight_map = add_character_others(
				target, weight_map, weight[word_no], character_bbox[word_no][i].copy()[:, 0, :], type_=type_)

	return target/255, weight_map


def generate_affinity(image_size, character_bbox, text, weight=None):

	"""

	:param image_size: shape = [3, image_height, image_width]
	:param character_bbox: [2, 4, num_characters]
	:param text: [num_words]
	:param weight: This is currently used only for synth-text so specifying weight as not None will generate a heatmap
					having value one where there is affinity
	:return: if weight is not None then target_affinity_heatmap otherwise target_affinity_heatmap,
																				weight for weak-supervision

	"""

	character_bbox = character_bbox.transpose(2, 1, 0)

	if len(image_size) == 2:
		height, width = image_size
	else:
		channel, height, width = image_size

	target = np.zeros([height, width], dtype=np.uint8)

	total_letters = 0

	all_affinity_bbox = []

	for word in text:
		for char_num in range(len(word)-1):
			target, bbox = add_affinity(target, character_bbox[total_letters].copy(), character_bbox[total_letters+1].copy())
			total_letters += 1
			all_affinity_bbox.append(bbox)
		total_letters += 1

	target = target / 255

	if weight is not None:

		return target, np.float32(target >= config.THRESHOLD_POSITIVE)

	else:

		return target, all_affinity_bbox


def generate_affinity_others(image_size, character_bbox, weight):

	"""

	:param image_size: shape = [3, image_height, image_width]
	:param character_bbox: [2, 4, num_characters]
	:param weight: This is currently used only for icdar 2013. it is a list containing weight for each bbox
	:return: target_affinity_heatmap, weight for weak-supervision

	"""

	if len(image_size) == 2:
		height, width = image_size
	else:
		channel, height, width = image_size

	target = np.zeros([height, width], dtype=np.uint8)
	weight_map = np.zeros([height, width], dtype=np.float32)

	for i, word in enumerate(character_bbox):
		for char_num in range(len(word)-1):
			target, weight_map = add_affinity_others(
				target,
				weight_map,
				weight[i],
				word[char_num][:, 0, :].copy(),
				word[char_num+1][:, 0, :].copy())

	return target/255, weight_map
