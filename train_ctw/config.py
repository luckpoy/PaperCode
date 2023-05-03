from pickle import TRUE
from torch import affine_grid_generator
from config import *

num_cuda = "0"
use_cuda = True
datasets = 'ctw'

epoch = 50

if datasets == 'ctw':
	img_size = 1024
else:
	img_size = 512
print("dataset is " + datasets)
up_mode = "old"

batch_size_train = 6
batch_size_test = 10
num_workers_train = int(2*batch_size_train)
num_workers_test = int(1*batch_size_test)

batch_size = {
	'train': batch_size_train*len(num_cuda.split(',')),
	'test': batch_size_test*len(num_cuda.split(',')),
}

num_workers = {
	'train': num_workers_train,
	'test': num_workers_test
}
work_dir = '/home/ubuntu/bo/CCD/'
if datasets == 'uname':
	save_path = work_dir + 'model/uname/'
else:
	save_path = work_dir + 'model/ctw/'
pretrained = True
pretrained_path = work_dir + 'model/ctw/model.pkl'
pretrained_loss_plot_training = work_dir + 'model/ctw/loss_plot_training.npy'
optimizer_iteration = 4//len(num_cuda.split(','))

single_test_model_path ='/home/liuyingfeng/bo/data_bak_211_uname/vgg_cct_seg/uname/model.pkl'
single_test_save_path = work_dir + 'debug/st/'

lr = {
	1: 1e-4,
	8000*optimizer_iteration: 5e-5,
	16000*optimizer_iteration: 2e-5,
	32000*optimizer_iteration: 1e-5,
	48000*optimizer_iteration: 1e-6,
}

num_epochs_strong_supervision = 1.2

periodic_fscore = 200*optimizer_iteration
periodic_output = 1000*optimizer_iteration
periodic_save = 1000*optimizer_iteration

visualize_generated = True
visualize_freq = 21000

weight_threshold = 0.5
seg_threshold = 0.8
seg_ignore = 0.3
seg_flag = False

affine_flag = True

model_architecture = 'craft'
#UNET_ResNet craft craft_cct craft_eff craft_vgg_cct craft_smalier


print("affine_flag = " + str(affine_flag))
print("seg_flag = " + str(seg_flag))
with open('log.txt','w') as f:
	f.write("model: " + model_architecture + '\n')
	f.write("affine_flag = " + str(affine_flag)+'\n')
	f.write("seg_flag = " + str(seg_flag))