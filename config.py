import math

seed = 0

THRESHOLD_POSITIVE = 0.1
THRESHOLD_NEGATIVE = 0

threshold_point = 25
window = 120

sigma = 18.5
sigma_aff = 20

boundary_character = math.exp(-1/2*(threshold_point**2)/(sigma**2))
boundary_affinity = math.exp(-1/2*(threshold_point**2)/(sigma_aff**2))

threshold_character = boundary_character - 0.03
threshold_affinity = boundary_affinity + 0.03

threshold_character_upper = boundary_character + 0.2
threshold_affinity_upper = boundary_affinity + 0.2

scale_character = math.sqrt(math.log(boundary_character)/math.log(threshold_character_upper))
scale_affinity = math.sqrt(math.log(boundary_affinity)/math.log(threshold_affinity_upper))

dataset_name = 'ICDAR2015'
test_dataset_name = 'ICDAR2015'

print("loaded config flie!")
# print(
# 	'Boundary character value = ', boundary_character,
# 	'| Threshold character value = ', threshold_character,
# 	'| Threshold character upper value = ', threshold_character_upper
# )
# print(
# 	'Boundary affinity value = ', boundary_affinity,
# 	'| Threshold affinity value = ', threshold_affinity,
# 	'| Threshold affinity upper value = ', threshold_affinity_upper
# )
# print('Scale character value = ', scale_character, '| Scale affinity value = ', scale_affinity)
# print('Training Dataset = ', dataset_name, '| Testing Dataset = ', test_dataset_name)

DataLoaderSYNTH_base_path = '/home/tml/bo/CRAFT/CRAFT-pytorch/data'
DataLoaderSYNTH_mat = '/home/tml/bo/CRAFT/CRAFT-pytorch/data/gt.mat'
DataLoaderSYNTH_Train_Synthesis = '/home/tml/bo/CRAFT/CRAFT-Remade/model/synth/train_synthesis/'

#DataLoader_Other_Synthesis = '/home/tml/bo/CRAFT/CRAFT-Remade/model/sharedata/'+dataset_name+'/Save/'
Other_Dataset_Path = '/home/tml/bo/CRAFT/CRAFT-Remade/ic15/' #'/home/tml/bo/CRAFT/CRAFT-Remade/ic13/ic13_input'
save_path = '/home/tml/bo/CRAFT/CRAFT-Remade/model/ic15/' #'/home/tml/bo/CRAFT/CRAFT-Remade/ic13/ic13_input'
images_path = save_path+'/Images'
target_path = save_path+'/Generated'

# ctw_datasets_path = '/home/ubuntu/bo/CCD/ctw/'
ctw_save_path = '/home/ubuntu/bo/CCD/model/ctw/'

if dataset_name=='uname':
	ctw_test_save_path = '/home/ubuntu/bo/CCD/model/uname_test/'
else:
	ctw_test_save_path = '/home/ubuntu/bo/CCD/model/ctw_test/'

Test_Dataset_Path = Other_Dataset_Path

threshold_word = 0.1
threshold_fscore = 0.5

dataset_pre_process = {
	'ic13': {
		'train': {
			'target_json_path': None,
			'target_folder_path': None,
		},
		'test': {
			'target_json_path': None,
			'target_folder_path': None,
		}
	},
	'ic15': {
		'train': {
			'target_json_path': "/home/tml/bo/CRAFT/CRAFT-Remade/ic15/Images/train_gt",
			'target_folder_path': "/home/tml/bo/CRAFT/CRAFT-Remade/ic15/Images/train",
		},
		'test': {
			'target_json_path': "/home/tml/bo/CRAFT/CRAFT-Remade/ic15/Images/test_gt",
			'target_folder_path': "/home/tml/bo/CRAFT/CRAFT-Remade/ic15/Images/test",
		}
	}
}

start_iteration = 0
skip_iterations = []
