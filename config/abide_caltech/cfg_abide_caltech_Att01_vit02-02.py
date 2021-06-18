import numpy as np

train_id = 'Att01_vit02-02'
data_identifier_source = 'abide_caltech'

number_of_epoch = 1000

deterministic = True
seed = 42

loss_mult = [0.5, 0.5]  # CE, Dice

n0 = 32
pbm = 0.0
batch_size = 8
num_classes = 15
path_to_save_trained_model = './pre_trained'

image_size = (256, 256, 256)  #XYZ
patch_size = (256, 4, 4)  #ZXY (16->2, 32->4)

use_attention = True

embedder = {
    'shape':
    tuple([batch_size, n0 * 8, image_size[0] // 8, image_size[1] // 8]),
    'size': patch_size,
}

bnViT = {
    'model'         : "vit_base_patch16_224",
    'input_size'    : np.prod(patch_size),
    'pretrained'    : True,
    'pth_path'      : "pre_trained/vit/jx_vit_base_p16_224-80ecf9dd.pth",
    'vit_size'      : 768    
}

test = {
    'data_identifier_source': data_identifier_source,
    'data_identifier_target': data_identifier_source,
    'experiment_name': train_id,
    'save_images': False,
}