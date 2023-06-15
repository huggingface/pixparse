import timm
from torch import nn as nn

from pixparse.models.config import ImageEncoderCfg


def create_image_encoder(cfg: ImageEncoderCfg) -> nn.Module:
    assert cfg.name
    extra_kwargs = {}
    if cfg.img_size is not None:
        extra_kwargs['img_size'] = cfg.img_size
    model = timm.create_model(
        cfg.name,
        pretrained=cfg.pretrained,
        num_classes=0,
        global_pool='',
        **extra_kwargs
    )

    # FIXME need to add support for changing input resolution / attn window sizes for models like swin,
    #  the original Donut added some hacks to resize rel-pos bias

    return model


class ImageEncoderTimm(nn.Module):
    def __init__(self, cfg: ImageEncoderCfg):
        super().__init__()
        self.trunk = create_image_encoder(cfg)
        self.pool = None   # TBD possible attention pooling w/ pos embed
        self.head = None   # TBD extra projection?

    def forward(self, x):
        x = self.trunk(x)
        if self.pool is not None:
            x = self.pool(x)
        if self.head is not None:
            x = self.head(x)
        # flatten?
        return x
