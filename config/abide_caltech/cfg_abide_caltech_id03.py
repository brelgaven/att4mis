import numpy as np

train_id = 'id03'
data_identifier_source = 'abide_caltech'

number_of_epoch = 1000

deterministic = True
seed = 42

loss_mult = [0.5, 0.5]

n0 = 64
pbm = 0.0
batch_size = 8
num_classes = 15
path_to_save_trained_model = './pre_trained'

use_attention = False

image_size = (256, 256, 256)
patch_size = (32, 8, 8)

test = {
    'data_identifier_source': data_identifier_source,
    'data_identifier_target': data_identifier_source,
    'experiment_name': train_id,
    'save_images': False,
}