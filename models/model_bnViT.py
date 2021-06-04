import timm
import torch
import torch.nn as nn


class BnViT(nn.Module):
    
    def __init__(self, model, input_size, 
                 output_size=None, pretrained=False,
                 pth_path=None, vit_size=768):

        output_size = output_size or input_size
        
        super(BnViT, self).__init__()

        model = timm.create_model(model, pretrained=False)
        if pretrained:
            model.load_state_dict(torch.load(pth_path))
            
        self.blocks = model.blocks
        
        self.in_norm = nn.LayerNorm(input_size)
        self.vit_norm = nn.LayerNorm(vit_size)
        self.out_norm = nn.LayerNorm(output_size)
        
        self.inF = nn.Linear(input_size, vit_size) 
        self.outF = nn.Linear(vit_size, output_size)

    def forward(self, x):
        
        x = self.in_norm(x)
        x = self.inF(x)
        
        x = self.vit_norm(x)
        
        for blk in self.blocks:
            x = blk(x)
        
        x = self.vit_norm(x)
        
        x = self.outF(x)
        x = self.out_norm(x)
        
        return x