import numpy as np

train_id = 'CSEtrans04'
data_identifier_source = 'nci'

number_of_epoch = 2000

deterministic = True
seed = 42

loss_mult = [1, 0]  # CE, Dice

n0 = 16
pbm = 0.0
batch_size = 8
num_classes = 3
path_to_save_trained_model = './pre_trained'

image_size = (256, 256, 20)  #XYZ
patch_size = (64, 8, 8)  #ZXY

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