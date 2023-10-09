import torch
import torch.nn as nn

from .config import ModelCfg
from .image_encoder_timm import ImageEncoderTimm
from .text_decoder_hf import TextDecoderHf


class Cruller(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.image_encoder = ImageEncoderTimm(cfg.image_encoder)
        self.text_decoder = TextDecoderHf(cfg.text_decoder)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.image_encoder.set_grad_checkpointing(enable)
        self.text_decoder.set_grad_checkpointing(enable)

    def forward(self, image_input, text_input):
        encoder_output = self.image_encoder(image_input)
        decoder_output = self.text_decoder(
            text_input,
            encoder_hidden_states=encoder_output,
            return_dict=True,
        )
        return decoder_output

