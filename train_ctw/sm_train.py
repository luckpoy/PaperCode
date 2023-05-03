import train_ctw.config as config
from train_ctw.dataloader import DataLoaderCTW
from .parallel import DataParallelModel, DataParallelCriterion
from .utils import calculate_batch_fscore, generate_word_bbox_batch, _init_fn
from .generic_model import Criterian
from .craft_smalier import compute_emb_loss

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt


os.environ['CUDA_VISIBLE_DEVICES'] = str(config.num_cuda)

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]  #   2
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        #   256 x 512
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature   #   256 x   512
            anchor_count = contrast_count   #   2
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        #   print (anchor_dot_contrast.size())  256 x 256

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss


def test(dataloader, optimizer, model, model_smalier):

	# with torch.no_grad():  # For no gradient calculation
	optimizer_smalier=torch.optim.Adam(model_smalier.parameters(), lr=config.lr[1])

	model.eval()
	model_smalier.train()
	optimizer_smalier.zero_grad()
			
	iterator = tqdm(dataloader)
	all_loss = []
			
	def change_lr(no_i):
		for i in config.lr:
			if i == no_i:
				print('Learning Rate Changed to ', config.lr[i])
				for param_group in optimizer_smalier.param_groups:
					param_group['lr'] = 1e6

	for epoch in range(100):
		if epoch == 20:
			break

		for no, (image, weight, weight_affinity, drawn_image) in enumerate(iterator):
			change_lr(no)
			# weight[weight>0.2]=1
			# weight_affinity[weight_affinity>0]=1

			if config.use_cuda:
				image, weight, affinity = image.cuda(), weight.cuda(), weight_affinity.cuda()
			output, feature_list = model(image)
			
			feat_list = model_smalier(feature_list)

			# weight = nn.AdaptiveAvgPool2d((1,1))(weight.unsqueeze(1)).view(-1)
			# weight = torch.repeat_interleave(weight, 256, dim=2)
			# affinity = nn.AdaptiveAvgPool2d((1,1))(affinity.unsqueeze(1)).view(-1)
			# affinity = torch.repeat_interleave(affinity, 256, dim=2)
			# print(weight.unsqueeze(1).shape)

			# one = torch.ones_like(weight)
			# weight = torch.where(weight > 0.3, one, weight)
			# affinity = torch.where(affinity > 0, one, affinity)
			# weight = weight.cpu().numpy()
			# affinity = affinity.cpu().numpy()
			# print(weight.sum())
			weight[weight>0.2]=1.0
			# print(weight.sum())
			# print(affinity.sum())
			affinity[affinity>0.0]=1.0
			# print(affinity.sum())

			target=torch.cat([weight.unsqueeze(1), affinity.unsqueeze(1)], dim=1)
			feat = model_smalier(feature_list)
			
			b,c,h,w = feat.shape
			target = F.interpolate(target, size=(h, w), mode='bilinear', align_corners=False)
			
			# print(str(target.shape)+" "+str(feat.shape))

			loss=compute_emb_loss(feat,target)
			all_loss.append(loss)
			# print("epoch:{}".format(epoch)+"no:{}".format(no)+"emb_loss:{}".format(loss))
			# bsz = target.size(0)//2
			
			# print(bsz)
			# loss=0
			# for index in range(len(feat_list)):
			# 	features = feat_list[index]
			# 	# f1 = features
			# 	# f2 = features
			# 	f1, f2 = torch.split(features, (5, 5), dim=0)
			# 	features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
			# 	# print(features.shape)
			# 	# print(target.shape)
			# 	# loss += contra_criterion(features, labels=target) * 1e-1
			# 	loss += contra_criterion(features) * 1e-1
				

			loss.backward()
            
			if (no + 1) % config.optimizer_iteration == 0:
				optimizer_smalier.step()
				optimizer_smalier.zero_grad()

			iterator.set_description(
				'Loss:' + str(int(loss.item() * 10000) / 10000) + 
				' || Iterations:[' +
				str(no) + '/' + str(len(iterator)) 
			)
                        
		plt.plot(all_loss)
		plt.savefig(config.save_path + '/EMB_loss_plot_training.png')
		plt.clf()
		torch.save(
			{
				'state_dict': model_smalier.state_dict(),
				'optimizer': optimizer.state_dict(),
				'no' : no,
			}, config.save_path + '/' + 'model_epoch_'+str(epoch)+'.pkl')



def main(model_path):

	from .craft import CRAFT
	model = CRAFT()
	from .craft_smalier import smalier_model
	model_smalier = smalier_model()


	model = DataParallelModel(model)
	# loss_criterian = DataParallelCriterion(Criterian())
        
	optimizer = torch.optim.Adam(model.parameters(), lr=config.lr[1])


	print('Getting the Dataloader')
	train_dataloader = DataLoaderCTW(config.datasets)
	train_dataloader = DataLoader(
		train_dataloader, batch_size=config.batch_size['test'],
		shuffle=True, num_workers=config.num_workers['test'], worker_init_fn=_init_fn)
	
	if config.use_cuda:
		model = model.cuda()
		model_smalier = model_smalier.cuda()


	print('Got the dataloader')
	
	saved_model = torch.load(model_path)
	model.load_state_dict(saved_model['state_dict'])

	print('Loaded the model')

	test(train_dataloader, optimizer, model, model_smalier)


