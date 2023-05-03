import train_ctw.config as config
from .utils import generate_word_bbox_batch
from .parallel import DataParallelModel, DataParallelCriterion
from .data_manipulation import normalize_mean_variance,denormalize_mean_variance
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import src.config as cct_cfg
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def test(model):
	model.eval()
	image = cv2.imread(r'/home/liuyingfeng/bo/data_bak_211_uname/vgg_cct_affinity/uname_test/100/1/image.png')
	image = cv2.resize(image, (config.img_size,config.img_size))
	image = normalize_mean_variance(image).transpose(2, 0, 1)
	image = torch.from_numpy(np.expand_dims(image,axis=0))
	if config.use_cuda:
		image = image.cuda()

	output = model(image)

	if type(output) == list:
		output = torch.cat(output, dim=0)
	output[output < 0] = 0
	output[output > 1] = 1


	output = output.data.cpu().numpy()
	image = image[0].data.cpu().numpy()
	base = config.single_test_save_path

	os.makedirs(base, exist_ok=True)
	character_bbox = output[0, 0, :, :]
	drawn_image = denormalize_mean_variance(image.transpose(1, 2, 0))
	predicted_bbox = generate_word_bbox_batch(
					output[:, 0, :, :],
					character_threshold=config.threshold_character,
					word_threshold=config.threshold_word,
				)
	# image = cv2.imread(r'/home/liuyingfeng/bo/data_bak_211_uname/vgg_cct_affinity/uname_test/100/1/image.png')
	cv2.drawContours(drawn_image, predicted_bbox[0], -1, (0, 255, 0), 2)
	cv2.imwrite(base + '/st_image.png', drawn_image)
	plt.imsave(base + '/pred_characters.png', character_bbox)
	plt.imsave(
		base + '/pred_characters_thresh.png',
		np.float32(character_bbox > config.threshold_character))


def main():
	model_name = 'craft'
	if config.model_architecture == 'craft':
		from .craft import CRAFT
		model = CRAFT()
	elif config.model_architecture == 'craft_eff':
		from .craft_eff import CRAFT_cct
		model = CRAFT_cct()
	elif config.model_architecture == 'craft_cct':
		from .craft_cct import CRAFT_cct
		config_vit = cct_cfg.get_CTranS_config()
		model = CRAFT_cct(config=config_vit)
	elif config.model_architecture == 'craft_vgg_cct':
		from .craft_vgg_cct import CRAFT_cct
		config_vit = cct_cfg.get_CTranS_config()
		model = CRAFT_cct(config=config_vit)
	else :
		print("no the model,exit!")
		exit()

	model = DataParallelModel(model)
	if config.use_cuda:
		model = model.cuda()

	saved_model = torch.load(config.single_test_model_path)
	model.load_state_dict(saved_model['state_dict'])
	print('Loaded the model')
	test(model)
