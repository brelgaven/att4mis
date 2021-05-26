import numpy as np

train_id = 'Att01_noBtNk02'
data_identifier_source = 'acdc'

number_of_epoch = 1000

deterministic = True
seed = 42

loss_mult = [0.5, 0.5]  # CE, Dice

n0 = 32
pbm = 0.0
batch_size = 8
num_classes = 4
path_to_save_trained_model = './pre_trained'

image_size = (256, 256, 20)  #XYZ
patch_size = (32, 8, 8)  #ZXY

use_attention = True
noBtNk = True

test_batch_size = 1

test = {
    'data_identifier_source': data_identifier_source,
    'data_identifier_target': data_identifier_source,
    'experiment_name': train_id,
    'save_images': False,
    'batch_size': test_batch_size,
    'no_slices': [ 8,  8,  9,  9,  8,  8, 10, 10, 14, 14,  8,  8,  9,  9, 10, 10,  7,
                   7, 13, 13,  6,  6, 14, 14,  6,  6, 10, 10, 10, 10,  9,  9,  8,  8,
                  15, 15,  9,  9, 10, 10],
}