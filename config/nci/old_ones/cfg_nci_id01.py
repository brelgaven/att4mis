import numpy as np

train_id = 'id01'
data_identifier_source = 'nci'

number_of_epoch = 2000

deterministic = True
seed = 42

loss_mult = [0.5, 0.5]

n0 = 16
pbm = 0.0
batch_size = 8
num_classes = 3
path_to_save_trained_model = './pre_trained'

image_size = (194, 256, 256)
patch_size = (32, 8 ,8)

test = {
    'data_identifier_source': data_identifier_source,
    'data_identifier_target': data_identifier_source,
    'experiment_name'       : train_id,
    'save_images'           : True, 
}