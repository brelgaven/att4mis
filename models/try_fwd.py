#%% Imports

from ctun import *

import ml_collections

if torch.cuda.is_available():  
    dev = "cuda:0" 
else:  
    dev = "cpu"
    
device = torch.device(dev) 

#%% Example Configuration

config = ml_collections.ConfigDict()

config.num_classes = 4
config.n0 = 16
config.batch_size = 3
patch_size = (4, 2, 2)

input_shape = (config.batch_size, 1, 256, 256)
#================
x = torch.rand(input_shape).to(device)

enc = Encoder(config.n0).cuda()
encout_shape = enc(x)[-1].shape

print('Input shape is', input_shape)
print('Encoder output shape is', encout_shape)

#================

config.embedder = {
    'shape'     : encout_shape,
    'size'      : patch_size,
    #'stride': ,
    #'pe_fac': ,
}

# config.performer = {
#     'dim'   : np.prod(patch_size),
#     'depth' : 2,
#     'heads' : 4,
#     'causal': False,    
# }

config.transLayer = {
    'd_model'           : np.prod(patch_size),
    'nhead'             : 4,  
    'dim_feedforward'   : 4,
}

config.transformer = {
    'encoder_layer' : nn.TransformerEncoderLayer(**config.transLayer),
    'num_layers'    : 6,  
}

#%% Try the Model

x = torch.rand(input_shape).to(device)

model = Ctun(config).cuda()
model.cuda()
y, y_sigmoid, y_argmax = model(x)
print('Model forward works.') 
