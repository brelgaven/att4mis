import numpy as np

data_identifier_source = 'nci'
data_identifier_target = 'nci'
experiment_name = 'Att01_noBtNk02_0'

save_images = False
save_path = './results/%s/test/%s' % (data_identifier_source, experiment_name)
model_path = './pre_trained/nci_model_segmentation_Att01_noBtNk02_0.pth'

use_attention = True
noBtNk = True

train_id = experiment_name
batch_size = 1

n0 = 32
image_size = (256, 256, 20)
no_slices = [20, 20, 20, 20, 19, 20, 20, 15, 20, 20]