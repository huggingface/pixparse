import torch.nn as nn

from .config import ModelCfg
from .image_encoder_timm import ImageEncoderTimm
from .text_decoder_hf import TextDecoderHf



class Cruller(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.image_encoder = ImageEncoderTimm(cfg.image_encoder)
        self.text_decoder = TextDecoderHf(cfg.text_decoder)

    def forward(self, image_input, text_input):
        encoder_output = self.image_encoder(image_input)
        decoder_output = self.text_decoder(
            text_input,
            encoder_hidden_states=encoder_output,
            return_dict=True,
        )
        return decoder_output