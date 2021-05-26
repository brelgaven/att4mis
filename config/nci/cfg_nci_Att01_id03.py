import numpy as np

train_id = 'Att01_id03'
data_identifier_source = 'nci'

number_of_epoch = 2000

deterministic = True
seed = 42

loss_mult = [0.5, 0.5]  # CE, Dice

n0 = 64
pbm = 0.0
batch_size = 8
num_classes = 3
path_to_save_trained_model = './pre_trained'

image_size = (256, 256, 20)  #XYZ
patch_size = (32, 8, 8)  #ZXY

use_attention = True

test_batch_size = 1

test = {
    'data_identifier_source': data_identifier_source,
    'data_identifier_target': data_identifier_source,
    'experiment_name': train_id,
    'save_images': False,
    'batch_size': test_batch_size,
    'no_slices': [20, 20, 20, 20, 19, 20, 20, 15, 20, 20],
}