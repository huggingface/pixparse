import timm
from torch import nn as nn

from pixparse.data import ImageInputCfg, image_fmt_to_chs
from pixparse.models import ImageEncoderCfg


def create_image_encoder(cfg: ImageEncoderCfg) -> nn.Module:
    assert cfg.name
    extra_kwargs = {}
    if cfg.image_size is not None and cfg.needs_image_size:
        extra_kwargs['img_size'] = cfg.image_size
    model = timm.create_model(
        cfg.name,
        pretrained=cfg.pretrained,
        in_chans=image_fmt_to_chs(cfg.image_fmt),
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

        mean = self.trunk.pretrained_cfg.get('mean', [0.5])
        std = self.trunk.pretrained_cfg.get('std', [0.5])
        self.traits = dict(
            input=ImageInputCfg(
                image_size=cfg.image_size,
                image_mean=mean,
                image_std=std,
            )
        )

    def forward(self, x):
        x = self.trunk(x)
        if self.pool is not None:
            x = self.pool(x)
        if self.head is not None:
            x = self.head(x)
        # flatten?
        return x
