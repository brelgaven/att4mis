import numpy as np

train_id = 'Att01_trans06B'
data_identifier_source = 'abide_caltech'

number_of_epoch = 1000

deterministic = True
seed = 42

loss_mult = [0.5, 0.5]

n0 = 64
pbm = 0.0
batch_size = 2
num_classes = 15
path_to_save_trained_model = './pre_trained'

image_size = (256, 256, 256)  #XYZ
patch_size = (512, 2, 2)  #ZXY

use_attention = True
noBtNk = False

embedder = {
    'shape':
    tuple([batch_size, n0 * 8, image_size[0] // 8, image_size[1] // 8]),
    'size': patch_size,
}

transformer = {
    'num_layers': 6,
    'd_model': np.prod(patch_size),
    'nhead': 8,
    'dim_feedforward': 1024,
}

test_batch_size = batch_size

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
}