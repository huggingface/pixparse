import torch.nn as nn
import torch

import PIL

from typing import Optional

from .config import ModelCfg
from .image_encoder_timm import ImageEncoderTimm
from .text_decoder_hf import TextDecoderHf

import re

from transformers.file_utils import ModelOutput

class Cruller(nn.Module):
    def __init__(self, cfg: ModelCfg, tokenizer): #FIXME we need to pass along something like a TokenizerCfg
        super().__init__()
        self.image_encoder = ImageEncoderTimm(cfg.image_encoder)
        self.text_decoder = TextDecoderHf(cfg.text_decoder, tokenizer)

    def forward(self, image_input, text_input):
        encoder_output = self.image_encoder(image_input)
        decoder_output = self.text_decoder(
            text_input,
            encoder_hidden_states=encoder_output,
            return_dict=True,
        )
        return decoder_output