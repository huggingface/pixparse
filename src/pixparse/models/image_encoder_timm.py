from collections import OrderedDict

import timm
import torch
from torch import nn as nn

from pixparse.data import ImageInputCfg, image_fmt_to_chs
from pixparse.models import ImageEncoderCfg


class ImageEncoderTimm(nn.Module):
    def __init__(self, cfg: ImageEncoderCfg):
        super().__init__()
        assert cfg.name

        timm_kwargs = {}
        if cfg.image_size is not None and cfg.needs_image_size:
            timm_kwargs['img_size'] = cfg.image_size
        if cfg.patch_size is not None:
            timm_kwargs['patch_size'] = cfg.patch_size
        if cfg.window_size is not None:
            timm_kwargs['window_size'] = cfg.patch_size
        if cfg.drop_path_rate is not None:
            timm_kwargs['drop_path_rate'] = cfg.drop_path_rate
        if cfg.patch_drop_rate is not None:
            timm_kwargs['patch_drop_rate'] = cfg.patch_drop_rate
        if cfg.drop_rate and not cfg.head_type:
            # drop rate set in timm model only if external out projection not active
            timm_kwargs['drop_rate'] = cfg.drop_rate

        self.trunk = timm.create_model(
            cfg.name,
            pretrained=cfg.pretrained,
            in_chans=image_fmt_to_chs(cfg.image_fmt),
            num_classes=0,  # classifier removed
            global_pool='',  # pooling removed
            **timm_kwargs
        )

        prev_dim = self.trunk.num_features
        if cfg.timm_out_dim:
            # re-add classifier (random init) if timm_out_dim set
            self.trunk.reset_classifier(cfg.timm_out_dim)
            prev_dim = cfg.timm_out_dim

        assert not cfg.pool_type, 'pooling support not added'
        self.pool = None   # TBD possible attention pooling w/ pos embed

        out_dim = cfg.out_dim or prev_dim
        if cfg.head_type:
            assert cfg.head_type in ('linear', 'mlp')
            if cfg.head_type == 'mlp':
                self.head = timm.layers.Mlp(
                    in_features=prev_dim,
                    hidden_features=int(prev_dim * 2),  # fixed 2x mlp ratio, could make config
                    drop=(cfg.drop_rate, 0.),
                    out_features=out_dim,
                )
            else:
                self.head = nn.Sequential(OrderedDict([
                    ('drop', nn.Identity() if cfg.drop_rate is None else nn.Dropout(cfg.drop_rate)),
                    ('proj', nn.Linear(prev_dim, out_dim)),
                ]))
        else:
            self.head = nn.Identity()

        mean = self.trunk.pretrained_cfg.get('mean', [0.5])
        std = self.trunk.pretrained_cfg.get('std', [0.5])
        self.traits = dict(
            input=ImageInputCfg(
                image_size=cfg.image_size,
                image_mean=mean,
                image_std=std,
            )
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.trunk.set_grad_checkpointing(enable)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'trunk.' + n for n in self.trunk.no_weight_decay()}

    @torch.jit.ignore
    def get_wrap_layers(self):
        # FIXME make more generic
        if isinstance(self.trunk, timm.models.VisionTransformer):
            from timm.models.vision_transformer import Block
            return {Block}
        elif isinstance(self.trunk, timm.models.SwinTransformer):
            from timm.models.swin_transformer import SwinTransformerBlock
            return {SwinTransformerBlock}
        else:
            assert False

    def forward(self, x):
        x = self.trunk(x)
        if self.pool is not None:
            x = self.pool(x)
        x = self.head(x)

        # FIXME flatten for BCHW/BHWC outs?
        return x
