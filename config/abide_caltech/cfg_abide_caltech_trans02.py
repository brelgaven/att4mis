import numpy as np

train_id = 'trans02'
data_identifier_source = 'abide_caltech'

number_of_epoch = 1000

deterministic = True
seed = 42

loss_mult = [0.5, 0.5]

n0 = 16
pbm = 0.0
batch_size = 8
num_classes = 15
path_to_save_trained_model = './pre_trained'

use_attention = False

image_size = (256, 256, 256)
patch_size = (32, 8, 8)  #ZXY

embedder = {
    'shape':
    tuple([batch_size, n0 * 8, image_size[0] // 8, image_size[1] // 8]),
    'size': patch_size,
}

transformer = {
    'num_layers': 12,
    'd_model': np.prod(patch_size),
    'nhead': 8,
    'dim_feedforward': 1024,
}

test = {
    'data_identifier_source': data_identifier_source,
    'data_identifier_target': data_identifier_source,
    'experiment_name': train_id,
    'save_images': False,
}