from torch.utils import data
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import train_ctw.config as config
from .data_manipulation import resize, normalize_mean_variance, polygon_area, generate_target,generate_origin_cross_affinity_map

from .ctw_convert import CTWConvertor as ctw_train
from .ctw_convert_test import CTWConvertor as ctw_test
from .decode_unamedata import decode_anno

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class DataLoaderCTW(data.Dataset):
	def __init__(self, type_):

		self.type_ = type_
		# self.base_path = config.ctw_datasets_path

		if self.type_ == 'ctw':
			print("unsed ctw datasets!")
			CTW_text_convertor = ctw_train(jsonl_path=r'/home/ubuntu/bo/data/ctw/ctw-annotations/train.jsonl',
											image_root=r'/home/ubuntu/bo/data/ctw/images-trainval')
			self.craft_train_list = CTW_text_convertor.convert_to_craft()
			# np.random.shuffle(self.craft_train_list)
		elif self.type_ == 'uname':
			print("unsed unamed train datasets!")
			self.dataset_path = r'/home/ubuntu/bo/data/uname/'
			self.pic_list,self.bboxes = decode_anno(self.dataset_path,'train')
		else:
			print('datasets type is error!')


	def __getitem__(self, item):
		if self.type_ == 'ctw':
			image = plt.imread(self.craft_train_list[(item % len(self.craft_train_list))][0])  # Read the image
			char_box = self.craft_train_list[item][1].copy()[0]
			for i in range(1, len(self.craft_train_list[item][1])):
				char_box = np.concatenate((char_box, self.craft_train_list[item][1].copy()[i]), axis=0)
			
			temp = []
			for box in char_box:
				if polygon_area(box) > 4*64*0 :
					temp.append(box.copy())
			big_box=np.array(temp)
		else:
			image = plt.imread(self.dataset_path + 'pics/' + self.pic_list[item])
			big_box = np.array(self.bboxes[item])

		if big_box.shape[0]==0:
			image = cv2.resize(image, (config.img_size,config.img_size))
			normal_image = image.astype(np.uint8).copy()
			image = normalize_mean_variance(image).transpose(2, 0, 1)
			weight_character = np.zeros([config.img_size, config.img_size], dtype=np.uint8)/255
		else:
			image, character = resize(image, big_box.transpose(2,1,0),side=config.img_size) 
			normal_image = image.astype(np.uint8).copy()
			image = normalize_mean_variance(image).transpose(2, 0, 1)
			# Generate character heatmap
			weight_character = generate_target(image.shape, character.copy())
			# Generate weight_affinity heatmap
			weight_affinity = generate_origin_cross_affinity_map(image.shape, character.copy())
			# plt.imsave("./weight_affinity.png",weight_affinity)
			# plt.imsave("./weight_character.png",weight_character)
			# np.savetxt("text",weight_affinity,fmt='%3f',delimiter=' ')

		return \
			image.astype(np.float32), \
			weight_character.astype(np.float32), \
			weight_affinity.astype(np.float32), \
			normal_image
			

	def __len__(self):

		if self.type_ == 'ctw':
			return int(len(self.craft_train_list))
		else:
			return int(len(self.pic_list))


class DataLoaderCTW_test(data.Dataset):
	def __init__(self, type_):
		self.type_ = type_

		if self.type_ == 'ctw':
			jsonl_path=r"/home/ubuntu/bo/data/ctw/test_cls.jsonl"
			# jsonl_path=r'/home/tml/boo/tml/bo/CRAFT/CRAFT-Remade/ctw/ctw-annotations/test_cls.jsonl'
			image_root=r'/home/ubuntu/bo/data/ctw/images-test'
			CTW_test_convertor = ctw_test(jsonl_path,image_root)
			self.craft_test_list = CTW_test_convertor.convert_to_craft()
		elif self.type_ == 'uname':
			print("unsed unamed test datasets!")
			self.dataset_path = r'/home/ubuntu/bo/data/uname/'
			self.pic_list,self.bboxes = decode_anno(self.dataset_path,'test')
		else:
			print("no state")

	def __getitem__(self, item):
		if self.type_ == 'ctw':
			image = plt.imread(self.craft_test_list[item][0])  # Read the image
			char_box = self.craft_test_list[item][1].copy()[0]
			for i in range(1, len(self.craft_test_list[item][1])):
				char_box = np.concatenate((char_box, self.craft_test_list[item][1].copy()[i]), axis=0)
			
			temp = []
			for box in char_box:
				if polygon_area(box) > 4*64*0 :
					temp.append(box.copy())
			big_box=np.array(temp)
		else:
			image = plt.imread(self.dataset_path + 'pics/' + self.pic_list[item])
			big_box = np.array(self.bboxes[item])

		if big_box.shape[0]==0:
			image = cv2.resize(image, (config.img_size,config.img_size))
			image = normalize_mean_variance(image).transpose(2, 0, 1)

			weight_character = np.zeros([config.img_size, config.img_size], dtype=np.uint8)/255
			weight_affinity = np.zeros([config.img_size, config.img_size], dtype=np.uint8)/255
			num_box = 0
			new_box = np.zeros([5000,], dtype = float, order = 'C')
		else:
			image, character = resize(image, big_box.transpose(2,1,0),side=config.img_size)  # Resize the image to (768, 768)
			# normal_image = image.astype(np.uint8).copy()
			image = normalize_mean_variance(image).transpose(2, 0, 1)
			# Generate character heatmap
			weight_character = generate_target(image.shape, character.copy())
			# return boxes
			num_box = character.shape[2]
			new_box = character.reshape(character.size)
			# print(character.size)
			new_box = np.pad(new_box,(0,5000-character.size),'constant',constant_values=(0,0))
			weight_affinity = generate_origin_cross_affinity_map(image.shape, character.copy())

		return \
			image.astype(np.float32), \
			weight_character.astype(np.float32), \
			weight_affinity, \
			new_box.astype(np.float32), \
			num_box
			

	def __len__(self):

		if self.type_ == 'ctw':
			return int(len(self.craft_test_list))
		else:
			return int(len(self.pic_list))

class DataLoaderEval(data.Dataset):

	"""
		DataLoader for evaluation on any custom folder given the path
	"""

	def __init__(self, path):

		self.base_path = path
		self.imnames = sorted(os.listdir(self.base_path))

	def __getitem__(self, item):

		image = plt.imread(self.base_path+'/'+self.imnames[item])  # Read the image

		if len(image.shape) == 2:
			image = np.repeat(image[:, :, None], repeats=3, axis=2)
		elif image.shape[2] == 1:
			image = np.repeat(image, repeats=3, axis=2)
		else:
			image = image[:, :, 0: 3]

		# ------ Resize the image to (768, 768) ---------- #

		height, width, channel = image.shape
		max_side = max(height, width)
		new_resize = (int(width / max_side * 768), int(height / max_side * 768))
		image = cv2.resize(image, new_resize)

		big_image = np.ones([768, 768, 3], dtype=np.float32) * np.mean(image)
		big_image[
			(768 - image.shape[0]) // 2: (768 - image.shape[0]) // 2 + image.shape[0],
			(768 - image.shape[1]) // 2: (768 - image.shape[1]) // 2 + image.shape[1]] = image
		big_image = normalize_mean_variance(big_image)
		big_image = big_image.transpose(2, 0, 1)

		return big_image.astype(np.float32), self.imnames[item], np.array([height, width])

	def __len__(self):

		return len(self.imnames)
