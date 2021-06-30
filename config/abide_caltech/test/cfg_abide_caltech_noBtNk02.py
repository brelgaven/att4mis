import numpy as np

data_identifier_source = 'abide_caltech'
data_identifier_target = 'abide_caltech'
experiment_name = 'noBtNk02'

save_images = False
save_path = './results/%s/test/%s' % (data_identifier_source, experiment_name)
model_path = './pre_trained/abide_caltech_model_segmentation_noBtNk02.pth'

use_attention = False
noBtNk = True

train_id = experiment_name
batch_size = 1

n0 = 32
image_size = (256, 256, 256) 

