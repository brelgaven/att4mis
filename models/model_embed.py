#%% Imports

import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from positional_encodings import PositionalEncodingPermute1D, \
    PositionalEncodingPermute2D, PositionalEncodingPermute3D

PEs = [PositionalEncodingPermute1D,
       PositionalEncodingPermute2D, 
       PositionalEncodingPermute3D]

#%% Patcher Methods

def patcher(x, size, stride=None, return_us=False):
    # modified version of 
    # https://discuss.pytorch.org/t/creating-nonoverlapping-patches
    # -from-3d-data-and-reshape-them-back-to-the-image/51210/5
    
    stride = stride or size 
    
    b = x.shape[0]
    
    kc, kh, kw = size   # kernel size
    dc, dh, dw = stride # stride

    # REVIEW Changed here (0,1,2) -> (1,2,3)
    x_pad = F.pad(x, (x.size(3)%kw // 2, x.size(3)%kw // 2,
                      x.size(2)%kh // 2, x.size(2)%kh // 2,
                      x.size(1)%kc // 2, x.size(1)%kc // 2))

    patches = x_pad.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
    if return_us == True:
        unfold_shape = patches.size()
    patches = patches.contiguous().view(b, -1, kc, kh, kw)

    if return_us == True:
        return patches, unfold_shape, x_pad
    else:
        return patches
    
def depatcher(patches, unfold_shape):
    
    # for last iteration, last batch might be different
    if np.prod(unfold_shape) != np.prod(patches.shape):
        unfold_shape = list(unfold_shape) #torchSize to List
        unfold_shape[0] = np.prod(patches.shape) // np.prod(unfold_shape[1:])
    
    patches_orig = patches.view(unfold_shape)
    output_c = unfold_shape[1] * unfold_shape[4]
    output_h = unfold_shape[2] * unfold_shape[5]
    output_w = unfold_shape[3] * unfold_shape[6]
    patches_orig = patches_orig.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    patches_orig = patches_orig.view(-1, output_c, output_h, output_w)
    
    return patches_orig

def check_patcher(shape, size):
    
    x = torch.randn(shape)

    z, unfold_shape, x_pad = patcher(x, size, return_us=True)
    y = depatcher(z, unfold_shape)

    output_c = unfold_shape[1] * unfold_shape[4]
    output_h = unfold_shape[2] * unfold_shape[5]
    output_w = unfold_shape[3] * unfold_shape[6]

    if np.allclose(y, x_pad[:, :output_c, :output_h, :output_w]):
        if y.shape == x.shape:
            if np.allclose(y, x_pad):
                flag = 0
            else:        
                flag = 1
        else:
            flag = 1
    else:
        flag = -1
        
    if flag == 1:
        warnings.warn("Patcher dimensions work only with padding!")
    elif flag == -1:
        warnings.warn("Patcher is wrong!!! Risky to use!!!")
    
    return flag

#%% Embedder Class

class Embedder(nn.Module):
    def __init__(self, shape, size, stride=None, pe_fac=1.0, slp=False):
        super(Embedder, self).__init__()
        check_patcher(shape, size)
        self.shape = shape # input shape
        self.pos_enc = PEs[len(shape)-3](shape[1])
        self.size = size
        self.stride = stride or size
        self.pe_fac = pe_fac
        
        p, us, x_pad = patcher(torch.zeros(shape), self.size, 
                               self.stride, return_us=True)
        
        self.unfold_shape = us
        self.padded_shape = x_pad.shape # not used, only info
        n_dims = np.prod(p.shape[2:])
        if slp:
            self.outF = nn.Linear(n_dims, n_dims, bias=True)
        else:
            self.outF = nn.Identity()
        
    def forward(self, x):
        pe = self.pe_fac * self.pos_enc(0*x)
        x = (x + pe)
        
        p = patcher(x, self.size, self.stride)
        # p.shape (b, n, c, h, w)
        q = p.view(tuple(p.shape[:2]) + (-1,))
        # q.shape (b, n, c*h*w)
        y = self.outF(q)
        
        return y