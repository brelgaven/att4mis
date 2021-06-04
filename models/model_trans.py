#%% Imports

try:
    from models.model_embed import *
    from models.model_bnViT import BnViT
except Exception:
    from model_embed import *
    from model_bnViT import BnViT

import torch
import torch.nn as nn
from performer_pytorch import Performer

import math
import torch.nn.functional as F
import copy

#%% Tranformer Reconfiguration

def transConfig(cfg):
    cfg_layer = copy.deepcopy(cfg.transformer)
    num_layers = cfg_layer.pop('num_layers')
    cfg_trans = {
        'encoder_layer' : nn.TransformerEncoderLayer(**cfg_layer),
        'num_layers'    : num_layers,  
    }
    return cfg_trans

#%% BottleNeck Class

class BottleNeck(nn.Module):
    def __init__(self, cfg):
        self.noFlag = False
        self.idFlag = False
        super(BottleNeck, self).__init__()
        if hasattr(cfg, 'noBtNk'):
            self.noFlag = cfg.noBtNk
        if hasattr(cfg, 'performer'):
            self.embedder = Embedder(**cfg.embedder)
            self.transformer = Performer(**cfg.performer)
        elif hasattr(cfg, 'transformer'):
            self.embedder = Embedder(**cfg.embedder)
            self.transformer = nn.TransformerEncoder(**transConfig(cfg))
        elif hasattr(cfg, 'bnViT'):
            self.embedder = Embedder(**cfg.embedder)
            self.transformer = BnViT(**cfg.bnViT)
            model_name = cfg.bnViT['model']
            print(f'ViT Transformer: {model_name}')
        elif not self.noFlag:
            self.idFlag = True
            
        self.informater()
        
    def forward(self, x):
        if self.noFlag:
            y = 0.0*x
        elif self.idFlag:
            y = x
        else:
            z = self.embedder(x)
            t = self.transformer(z)
            y = depatcher(t, self.embedder.unfold_shape)
        return y
    
    def informater(self):
        
        if self.noFlag == True:
            print('No Bottleneck Mode')
        
        if self.idFlag == True:
            print('ID Mode')