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

    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd = set()
        no_wd |= {'image_encoder.' + n for n in self.image_encoder.no_weight_decay()}
        no_wd |= {'text_decoder.' + n for n in self.text_decoder.no_weight_decay()}
        return no_wd

    @torch.jit.ignore
    def get_wrap_layers(self):
        wrap_layers = set()
        wrap_layers |= self.image_encoder.get_wrap_layers()
        wrap_layers |= self.text_decoder.get_wrap_layers()
        return wrap_layers

    def forward(self, image_input, text_input):
        encoder_output = self.image_encoder(image_input)
        decoder_output = self.text_decoder(
            text_input,
            encoder_hidden_states=encoder_output,
            return_dict=True,
        )
        return decoder_output

