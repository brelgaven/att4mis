import numpy as np

data_identifier_source = 'acdc'
data_identifier_target = 'acdc'
experiment_name = 'Att01_noBtNk02'

save_images = False
save_path = './results/%s/test/%s' % (data_identifier_source, experiment_name)
model_path = './pre_trained/acdc_model_segmentation_Att01_noBtNk02.pth'

use_attention = True
noBtNk = True

train_id = experiment_name
batch_size = 1

n0 = 32
image_size = (256, 256, 20)
no_slices = [8, 8, 9, 9, 8, 8, 10, 10, 14, 14, 8, 8, 9, 9, 10, 10, 7, 7, 13, 13, 6, 6, 14, 14, 6, 6, 10, 10, 10, 10, 9, 9, 8, 8, 15, 15, 9, 9, 10, 10] 
