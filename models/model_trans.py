#%% Imports

try:
    from models.model_embed import *
except Exception:
    from model_embed import *

import torch
import torch.nn as nn
from performer_pytorch import Performer

import math
import torch.nn.functional as F

#%% BottleNeck Class

class BottleNeck(nn.Module):
    def __init__(self, cfg):
        super(BottleNeck, self).__init__()
        if hasattr(cfg, 'performer'):
            self.embedder = Embedder(**cfg.embedder)
            self.transformer = Performer(**cfg.performer)
        elif hasattr(cfg, 'transformer'):
            self.embedder = Embedder(**cfg.embedder)
            self.transformer = nn.TransformerEncoder(**cfg.transformer)
        else:
            self.embedder = nn.Identity()
            self.transformer = nn.Identity()
        
    def forward(self, x):
        z = self.embedder(x)
        t = self.transformer(z)
        y = depatcher(t, self.embedder.unfold_shape)
        return y
