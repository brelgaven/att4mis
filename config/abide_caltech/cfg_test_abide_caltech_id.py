import numpy as np

data_identifier_source = 'abide_caltech'
data_identifier_target = 'abide_caltech'
experiment_name = 'id01'

save_images = True
path_to_save_images = './results/%s/test/%s'%(data_identifier_source, experiment_name)
path_to_load_pretrained_model = './pre_trained/abide_caltech_model_segmentation_id01.pth'

deterministic = True
seed = 42

deterministic = True
seed = 42

loss_mult = [0.5, 0.5]

n0 = 16
pbm = 0.0
batch_size = 8
num_classes = 15
path_to_save_trained_model = './pre_trained'

image_size = (256, 256, 256)
patch_size = (32,8,8)