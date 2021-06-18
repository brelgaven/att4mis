import numpy as np

train_id = 'Att01_vit01-02'
data_identifier_source = 'nci'

number_of_epoch = 2000

deterministic = True
seed = 57

loss_mult = [0.5, 0.5]  # CE, Dice

n0 = 16
pbm = 0.0
batch_size = 8
num_classes = 3
path_to_save_trained_model = './pre_trained'

image_size = (256, 256, 20)  #XYZ
patch_size = (128, 4, 4)  #ZXY (16->2, 32->4)

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

test_batch_size = 1

test_embedder = {
    'shape':
    tuple([test_batch_size, n0 * 8, image_size[0] // 8, image_size[1] // 8]),
    'size':
    patch_size,
}

test = {
    'data_identifier_source': data_identifier_source,
    'data_identifier_target': data_identifier_source,
    'experiment_name': train_id,
    'save_images': False,
    'batch_size': test_batch_size,
    'embedder': test_embedder,
    'no_slices': [20, 20, 20, 20, 19, 20, 20, 15, 20, 20],
}