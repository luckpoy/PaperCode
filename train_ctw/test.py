import train_ctw.config as config
from train_ctw.dataloader import DataLoaderCTW_test
from .parallel import DataParallelModel, DataParallelCriterion
from .utils import calculate_batch_fscore, generate_word_bbox_batch, _init_fn
from .generic_model import Criterian
from .data_manipulation import denormalize_mean_variance

from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import src.config as cct_cfg
import cv2

# from PIL import Image
# import torchvision.transforms as transforms
import operator


os.environ['CUDA_VISIBLE_DEVICES'] = str(config.num_cuda)


def save(image, output, target, no, target_bbox, o_a):
	output = output.data.cpu().numpy()
	image = image.data.cpu().numpy()
	target = target.data.cpu().numpy()

	batch_size = output.shape[0]
	base = config.ctw_test_save_path+str(no)+'/'
	# base = '/home/tml/boo/tml/bo/CRAFT/CRAFT-Remade/model/ctw_test1/'+str(no)+'/'
	# save_path = config.single_test_save_path

	os.makedirs(base, exist_ok=True)

	for i in range(batch_size):
		os.makedirs(base+str(i), exist_ok=True)
		character_bbox = output[i, 0, :, :]
		if config.affine_flag:
			affine_output=output[i, 1, :, :]
		drawn_image = denormalize_mean_variance(image[i].transpose(1, 2, 0))
		predicted_bbox = generate_word_bbox_batch(
						output[i, :, :, :],
						character_threshold=config.threshold_character,
						word_threshold=config.threshold_word,
					)
		origin_image = drawn_image.copy()
		cv2.drawContours(drawn_image, predicted_bbox[0], -1, (0, 255, 0), 2)
		
		save_path=base+str(i)
		plt.imsave(save_path + '/image.png', origin_image)
		plt.imsave(save_path + '/output.png', drawn_image)
		plt.imsave(save_path + '/target_characters.png', target[i, :, :])
		plt.imsave(save_path + '/pred_characters.png', character_bbox)
		if config.affine_flag:
			plt.imsave(save_path + '/affine_output.png', affine_output)
		plt.imsave(
			save_path + '/pred_characters_thresh.png',
			np.float32(character_bbox > config.threshold_character))


def test(dataloader, loss_criterian, model):

	with torch.no_grad():  # For no gradient calculation

		model.eval()
		iterator = tqdm(dataloader)
		all_loss = []
		all_accuracy = []

		for no, (image, weight, weight_affinity, boxes, num_box) in enumerate(iterator):

			if config.use_cuda:
				image, weight, affinity = image.cuda(), weight.cuda(), weight_affinity.cuda()
			output,_ = model(image)
			loss = loss_criterian(output, weight, affinity).mean()

			all_loss.append(loss.item())
			if type(output) == list:
				output = torch.cat(output, dim=0)
			output[output < 0] = 0
			output[output > 1] = 1

			# output execute -
			if config.affine_flag:
				output[:, 0, :, :] = output[:, 0, :, :] - output[:, 1, :, :]
			else:
				output[:, 0, :, :] = output[:, 0, :, :]

			output[output < 0] = 0
			output[output > 1] = 1

			# unloader = transforms.ToPILImage()
			# img = unloader(output[0, 0, :, :].data.cpu())
			# img.convert('RGBA').save("debug/output1.png")
			# img = unloader(weight[0].data.cpu())
			# img.convert('RGBA').save("debug/weight.png")

			predicted_bbox = generate_word_bbox_batch(
				output[:, 0, :, :].data.cpu().numpy(),
				character_threshold=config.threshold_character,
				word_threshold=config.threshold_word,
			)
			target_bbox = []
			for i in range(boxes.shape[0]):
				if num_box[i] == 0:
					target_bbox.append(np.zeros(shape=(5,2)))
				else:
					target_bbox.append(boxes[i][:num_box[i]*8].reshape(2, 4, -1).numpy().transpose(2, 1, 0)[:,:,np.newaxis,:])
		
			o_a=calculate_batch_fscore(pred=predicted_bbox, target=target_bbox, threshold=config.threshold_fscore, text_target=None)
			o_f,o_p,o_r = o_a
			all_accuracy.append(o_a)

			iterator.set_description(
				# 'Loss:' + str(int(loss.item() * config.optimizer_iteration * 1000000) / 1000000) + 
				'Iterations:[' +
				str(no) + '/' + str(len(iterator)) +
				'| Score: ' + 
				'F:'+str(int(o_f*1000)/1000) + ' |' + 
				'P:'+str(int(o_p*1000)/1000) + ' |' +
				'R:'+str(int(o_r*1000)/1000)
			)
			if no % 20 == 0 and no != 0:
				if type(output) == list:
					output = torch.cat(output, dim=0)
				save(image, output, weight, no, target_bbox,o_a)
				# break
			# print("test")

		return all_accuracy


def main(model_path):

	if config.model_architecture == 'UNET_ResNet':
		from src.UNET_ResNet import UNetWithResnet50Encoder
		model = UNetWithResnet50Encoder()
	elif config.model_architecture == 'craft':
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
	loss_criterian = DataParallelCriterion(Criterian())

	test_dataloader = DataLoaderCTW_test(config.datasets)
	if config.use_cuda:
		model = model.cuda()

	print('Getting the Dataloader')

	test_dataloader = DataLoader(
		test_dataloader, batch_size=config.batch_size['test'],
		shuffle=False, num_workers=config.num_workers['test'], worker_init_fn=_init_fn)

	print('Got the dataloader')
	
	saved_model = torch.load(model_path)
	model.load_state_dict(saved_model['state_dict'])

	print('Loaded the model')

	all_accuracy = test(test_dataloader, loss_criterian, model)
	np.array(all_accuracy)
	eval=np.mean(all_accuracy,axis=0)
	print('Average eval  on the set is:', eval)
