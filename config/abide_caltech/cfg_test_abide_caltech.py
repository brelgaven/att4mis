import numpy as np

data_identifier_source = 'abide_caltech'
data_identifier_target = 'abide_caltech'
experiment_name = 'trans02'

save_images = True
save_path = './results/%s/test/%s'%(data_identifier_source, experiment_name)
model_path = './pre_trained/abide_caltech_model_segmentation_trans02.pth'

deterministic = True
seed = 42

loss_mult = [0.5, 0.5]

n0 = 16
pbm = 0.0
batch_size = 16
num_classes = 15
path_to_save_trained_model = './pre_trained'

image_size = (256, 256, 256)
patch_size = (32,8,8)

embedder = {
   'shape'     : tuple([batch_size, n0*8, image_size[1]//8, image_size[2]//8]),
   'size'      : patch_size,
}

transformer = {
    'num_layers'        : 12, 
    'd_model'           : np.prod(patch_size),
    'nhead'             : 8,  
    'dim_feedforward'   : 1024,
}