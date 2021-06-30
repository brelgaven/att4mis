import numpy as np

data_identifier_source = 'acdc'
data_identifier_target = 'acdc'
experiment_name = 'Att01_trans06'

save_images = False
save_path = './results/%s/test/%s'%(data_identifier_source, experiment_name)
model_path = './pre_trained/acdc_model_segmentation_Att01_trans06.pth'

train_id = experiment_name
batch_size = 1

use_attention = True

n0 = 64
image_size = (256, 256, 20)
no_slices = [8, 8, 9, 9, 8, 8, 10, 10, 14, 14, 8, 8, 9, 9, 10, 10, 7, 7, 13, 13, 6, 6, 14, 14, 6, 6, 10, 10, 10, 10, 9, 9, 8, 8, 15, 15, 9, 9, 10, 10] 
patch_size = (512, 1, 1)

embedder = {
   'shape'     : tuple([batch_size, n0*8, image_size[0]//8, image_size[1]//8]),
   'size'      : patch_size,
}