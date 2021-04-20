# =======================
# Config file to test segmentation network using test_segmentation_source.py
# =======================
# Test parameters

import torch.nn as nn
import numpy as np

data_identifier_source = 'abide_caltech'
data_identifier_target = 'abide_caltech'
experiment_name = '%s_test_segmentation_cfg_01'%data_identifier_source
image_size = (256, 256, 256)
batch_size = 16
num_classes = 15

save_images = True
path_to_save_images = './results/%s/%s'%(data_identifier_source, experiment_name)
path_to_load_pretrained_model = './pre_trained/abide_caltech_model_segmentation01_0.pth'


# encoder/decoder parameters

n0 = 16

# bottleneck architecture

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
#     'dim_feedforward'   : 2048,
# }

# transformer = {
#     'encoder_layer' : nn.TransformerEncoderLayer(**transLayer),
#     'num_layers'    : 6,
# }
