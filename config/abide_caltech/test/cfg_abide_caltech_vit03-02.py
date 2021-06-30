import numpy as np

data_identifier_source = 'abide_caltech'
data_identifier_target = 'abide_caltech'
experiment_name = 'vit03-02'

save_images = False
save_path = './results/%s/test/%s'%(data_identifier_source, experiment_name)
model_path = './pre_trained/abide_caltech_model_segmentation_vit03-02.pth'

train_id = experiment_name
batch_size = 1

n0 = 64
image_size = (256, 256, 256) 

patch_size = (512, 4, 4)

embedder = {
   'shape'     : tuple([batch_size, n0*8, image_size[0]//8, image_size[1]//8]),
   'size'      : patch_size,
}