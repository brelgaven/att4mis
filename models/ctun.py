#%% Imports

try:
    from models.model_zoo import *
    from models.model_trans import *
except Exception:
    from model_zoo import *
    from model_trans import *

import pdb


def defaulting_config(cfg):

    if not hasattr(cfg, 'use_attention'):
        cfg.use_attention = False

    if not hasattr(cfg, 'noBtNk'):
        cfg.noBtNk = False

    return cfg


#%% Ctun Class


class Ctun(nn.Module):
    def __init__(self, cfg):
        super(Ctun, self).__init__()
        cfg = defaulting_config(cfg)
        self.pbm = cfg.pbm if hasattr(cfg, 'pbm') else 0.0
        self.encoder_unet = Encoder(cfg.n0)
        self.bottleneck_unet = BottleNeck(cfg)
        self.decoder_unet = Decoder(cfg.n0, cfg.num_classes, cfg.use_attention)

    def forward(self, x):
        zs = self.encoder_unet(x)

        bno = self.bottleneck_unet(zs[-1]) + self.pbm * zs[-1]

        zs = zs[:-1] + (bno, )
        y = self.decoder_unet(zs)

        y_sigmoid = nn.Sigmoid()(y)
        y_argmax = torch.argmax(y_sigmoid, axis=1)

        return y, y_sigmoid, y_argmax
