from dataclasses import dataclass
from typing import Optional


@dataclass
class ImageEncoderCfg:
    name: str = 'vit_base_patch16_224'
    img_size: Optional[int] = 448
    pretrained: bool = True


@dataclass
class TextDecoderCfg:
    name: str = 'facebook/bart-base'
    pretrained: bool = True
    num_decoder_layers: Optional[int] = 4
    max_length: Optional[int] = 1024
    pad_token_id: Optional[int] = None  # not sure that this is needed or a good idea?


@dataclass
class ModelCfg:
    image_encoder: ImageEncoderCfg = ImageEncoderCfg()
    text_decoder: TextDecoderCfg = TextDecoderCfg()
