import numpy as np

data_identifier_source = 'nci'
data_identifier_target = 'nci'
experiment_name = 'Att01_trans02_1'

save_images = False
save_path = './results/%s/test/%s' % (data_identifier_source, experiment_name)
model_path = './pre_trained/nci_model_segmentation_Att01_trans02_1.pth'

train_id = experiment_name
batch_size = 1

use_attention = True

n0 = 16
image_size = (256, 256, 20)
no_slices = [20, 20, 20, 20, 19, 20, 20, 15, 20, 20]
patch_size = (32, 8, 8)

embedder = {
    'shape':
    tuple([batch_size, n0 * 8, image_size[0] // 8, image_size[1] // 8]),
    'size': patch_size,
}