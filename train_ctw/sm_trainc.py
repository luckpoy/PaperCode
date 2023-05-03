import email
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import os
from shutil import copyfile
import numpy as np
import matplotlib.pyplot as plt
import traceback  
# import craft_smalier

from .generic_model import Criterian
from .dataloader import DataLoaderCTW
from train_ctw.dataloader import DataLoaderCTW_test
from .data_manipulation import denormalize_mean_variance
import train_ctw.config as config
from .parallel import DataParallelModel, DataParallelCriterion
from .utils import calculate_batch_fscore, generate_word_bbox_batch, _init_fn
import src.config as cct_cfg
import cv2
from train_ctw.test import test 
from .craft_smalier import compute_emb_loss
import torch.nn.functional as F

from .send_email import send_error_notification,send_email_tome


os.environ['CUDA_VISIBLE_DEVICES'] = str(config.num_cuda)


def save(data, output, target, drawn_image, no):
	output = output.data.cpu().numpy()
	data = data.data.cpu().numpy()
	target = target.data.cpu().numpy()
	drawn_image = drawn_image.data.cpu().numpy()
	batch_size = output.shape[0]
	base = config.ctw_save_path+str(no)+'/'

	os.makedirs(base, exist_ok=True)

	for i in range(batch_size):
		os.makedirs(base+str(i), exist_ok=True)
		character_bbox = output[i, 0, :, :]

		plt.imsave(base+str(i) + '/image.png', denormalize_mean_variance(data[i].transpose(1, 2, 0)))
		plt.imsave(base+str(i) + '/target_characters.png', target[i, :, :])
		blob = np.logical_or(
			target[i, :, :] > config.threshold_character,
			False
		)
		blob = np.float32(blob)
		plt.imsave(base + str(i) + '/blob.png', blob)
		plt.imsave(base + str(i) + '/pred_characters.png', character_bbox)
		# Thresholding the character
		plt.imsave(
			base + str(i) + '/pred_characters_thresh.png',
			np.float32(character_bbox > config.threshold_character)
		)
		predicted_bbox = generate_word_bbox_batch(
						output[i, :, :, :],
						character_threshold=config.threshold_character,
						word_threshold=config.threshold_word,
					)
		cv2.drawContours(drawn_image[i], predicted_bbox[0], -1, (0, 255, 0), 2)
		plt.imsave(
			base + str(i) + '/drawn_image.png',
			drawn_image[i]
		)


def train(dataloader, loss_criterian, model, model_smalier, optimizer, starting_no, all_loss, all_accuracy,test_dataloader,best):
	def change_lr(no_i):
		for i in config.lr:
			if i == no_i:
				print('Learning Rate Changed to ', config.lr[i])
				for param_group in optimizer.param_groups:
					param_group['lr'] = config.lr[i]

	for epoch in range(config.epoch):
		model.train()
		optimizer.zero_grad()
		iterator = tqdm(dataloader)
		len_it=len(iterator)
		for i, (image, weight, affinity, drawn_image) in enumerate(iterator):
			no = i + epoch*len_it + starting_no
			change_lr(no)

			if config.use_cuda:
				image, weight, affinity = image.cuda(), weight.cuda(), affinity.cuda()

			output, feature_list = model(image)

			weight[weight>0.2]=1.0
			affinity[affinity>0.0]=1.0
			target=torch.cat([weight.unsqueeze(1), affinity.unsqueeze(1)], dim=1)
			feat = model_smalier(feature_list)
			
			b,c,h,w = feat.shape
			target = F.interpolate(target, size=(h, w), mode='bilinear', align_corners=False)
			emb_loss=compute_emb_loss(feat,target)

			loss = loss_criterian(output, weight, affinity).mean()/config.optimizer_iteration +0.01*emb_loss
			all_loss.append(loss.item()*config.optimizer_iteration)
			loss.backward()

			if (no + 1) % config.optimizer_iteration == 0:
				optimizer.step()
				optimizer.zero_grad()

			iterator.set_description(
				'epoch:' + str(int(epoch)) + 
				' Iterations:[' +
				str(no % len(iterator)) + '/' + str(len(iterator)) +
				'] Average Loss:' + 
				str(int(np.array(all_loss)[-min(1000, len(all_loss)):].mean()*100000)/100000)
			)
			# if no % 4 == 0:
			# 	break

		test_accuracy = test(test_dataloader, loss_criterian, model)
		np.array(test_accuracy)
		model_eval=np.mean(test_accuracy,axis=0)[0]
		print(np.mean(test_accuracy,axis=0))
		if model_eval > best :
			best = model_eval
			torch.save(
				{
					'state_dict': model.state_dict(),
					'optimizer': optimizer.state_dict(),
					'no' : no,
					'eval' : model_eval
				}, config.save_path + '/' + 'model.pkl')
			np.save(config.save_path + '/loss_plot_training.npy', all_loss)
			plt.plot(all_loss)
			plt.savefig(config.save_path + '/loss_plot_training.png')
			plt.clf()

			# all_eval = np.mean(test_accuracy,axis=0)
			# email_body = "The model have update! Details as follow:\n"+ \
			# 	"epoch: " 		 + str(int(epoch)) + \
			# 	"no :"    		 + str(no) + \
			# 	"eval: "  		 + all_eval.tostring() 
			# 	'Average Loss: ' + str(int(np.array(all_loss)[-min(1000, len(all_loss)):].mean()*100000)/100000)
			# send_email_tome("训练进度提醒",email_body)

			    # print("affine_flag = " + affine_flag)
				# TypeError: must be str, not bool

	return all_loss


def main():
	from .craft import CRAFT
	model = CRAFT()
	from .craft_smalier import smalier_model
	model_smalier = smalier_model()

	print("model:"+config.model_architecture)
	model = DataParallelModel(model)
	loss_criterian = DataParallelCriterion(Criterian(affinity_flag=config.affine_flag))

	if config.use_cuda:
		model = model.cuda()

	optimizer = torch.optim.Adam(model.parameters(), lr=config.lr[1])

	if config.pretrained:
		saved_model = torch.load(config.pretrained_path)
		model.load_state_dict(saved_model['state_dict'])
		optimizer.load_state_dict(saved_model['optimizer'])
		starting_no = saved_model['no']
		eval_model = saved_model['eval']
		all_loss = np.load(config.pretrained_loss_plot_training).tolist()
		print('Loaded the pre_model')
		print('eval:'+ str(eval_model) +' no:'+ str(starting_no))
	else:
		starting_no = 0
		all_loss = []
		eval_model = 0
	all_accuracy = []

	saved_model = torch.load('model/sm_model.pkl')
	model_smalier.load_state_dict(saved_model['state_dict'])

	print('Loading the dataloader')
	train_dataloader = DataLoaderCTW(config.datasets)
	train_dataloader = DataLoader(
		train_dataloader, batch_size=config.batch_size['train'],
		shuffle=True, num_workers=config.num_workers['train'], worker_init_fn=_init_fn)

	test_dataloader = DataLoaderCTW_test(config.datasets)
	test_dataloader = DataLoader(
		test_dataloader, batch_size=config.batch_size['test'],
		shuffle=False, num_workers=config.num_workers['test'], worker_init_fn=_init_fn)
	# print('Loaded the dataloader')

	try:
		all_loss = train(
			train_dataloader,loss_criterian, model, model_smalier, optimizer, starting_no=starting_no,
			all_loss=all_loss, all_accuracy=all_accuracy,test_dataloader=test_dataloader,
			best=eval_model)

		np.save(config.save_path + '/loss_plot_training.npy', all_loss)
		plt.plot(all_loss)
		plt.savefig(config.save_path + '/loss_plot_training.png')
		plt.clf()

		print("Saved Final Model")

	except:
		print("occur error and will send notification to notify master!")
		traceback.print_exc()
		send_error_notification(traceback.format_exc())
