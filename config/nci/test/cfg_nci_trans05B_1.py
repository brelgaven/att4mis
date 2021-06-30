import numpy as np

data_identifier_source = 'nci'
data_identifier_target = 'nci'
experiment_name = 'trans05B_1'

save_images = False
save_path = './results/%s/test/%s'%(data_identifier_source, experiment_name)
model_path = './pre_trained/nci_model_segmentation_trans05B_1.pth'

train_id = experiment_name
batch_size = 1

n0 = 32
image_size = (256, 256, 20)
no_slices = [20, 20, 20, 20, 19, 20, 20, 15, 20, 20]
patch_size = (256, 4, 4)

embedder = {
   'shape'     : tuple([batch_size, n0*8, image_size[0]//8, image_size[1]//8]),
   'size'      : patch_size,
}