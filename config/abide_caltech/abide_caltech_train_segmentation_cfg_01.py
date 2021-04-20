# =======================
# Config file to test segmentation network using train_segmentation_source.py
# =======================
# Training parameters

import numpy as np
import torch.nn as nn

train_id = '01_0'
data_identifier_source = 'abide_caltech'

number_of_epoch = 200

n0 = 16
pbm = 0.0
batch_size = 16
num_classes = 15
path_to_save_trained_model = './pre_trained'

image_size = (256, 256, 256)
patch_size = (2,32,32)

embedder = {
    'shape'     : tuple([batch_size, n0*8, image_size[1]//8, image_size[2]//8]),
    'size'      : patch_size,
    #'stride': ,
    #'pe_fac': ,
}

performer = {
    'dim'   	: np.prod(patch_size),
    'depth' 	: 4,
    'heads' 	: 4,
    'dim_head'	: 256,
    'causal'	: False,    
}

# transLayer = {
#     'd_model'           : np.prod(patch_size),
#     'nhead'             : 4,  
#     'dim_feedforward'   : 4,
# }

# transformer = {
#     'encoder_layer' : nn.TransformerEncoderLayer(**config.transLayer),
#     'num_layers'    : 6,  
# }







